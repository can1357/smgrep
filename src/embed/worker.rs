//! Multi-threaded embedding worker pool with timeout management
//!
//! Provides concurrent embedding processing with automatic worker lifecycle
//! management and timeout-based cleanup.

use std::{
   sync::{
      Arc,
      atomic::{AtomicU64, Ordering},
   },
   time::Duration,
};

use futures::{StreamExt, stream::FuturesUnordered};
use smallvec::SmallVec;
use tokio::{
   sync::oneshot,
   task::JoinHandle,
   time::{Instant, MissedTickBehavior},
};
use tokio_util::sync::CancellationToken;

use crate::{
   Str, config,
   embed::{CandleEmbedder, Embedder, HybridEmbedding, QueryEmbedding, candle::EmbeddingError},
   error::Result,
};

struct WorkerMessage {
   chunk: SmallVec<[Str; 4]>,
   tx:    oneshot::Sender<Result<Vec<HybridEmbedding>>>,
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct RelativeClock(Instant);

impl RelativeClock {
   fn new() -> Self {
      Self(Instant::now())
   }

   fn time(&self) -> u64 {
      self.0.elapsed().as_millis() as u64
   }
}

/// Multi-threaded embedding worker pool
///
/// Distributes embedding work across multiple async workers with automatic
/// timeout management and graceful shutdown. Workers idle out after inactivity.
pub struct EmbedWorker {
   workers:      Option<Vec<JoinHandle<()>>>,
   sender:       flume::Sender<WorkerMessage>,
   cancel_token: CancellationToken,
   batch_size:   usize,
   embedder:     Arc<CandleEmbedder>,
}

impl EmbedWorker {
   /// Creates a worker pool with configured number of threads
   pub fn new() -> Result<Self> {
      let cfg = config::get();
      let num_threads = cfg.default_threads();
      let batch_sz = cfg.batch_size();
      let timeout = Duration::from_millis(cfg.worker_timeout_ms);
      let embedder = Arc::new(CandleEmbedder::new()?);

      let (tx, rx) = flume::bounded(num_threads * 2);

      let mut workers = Vec::with_capacity(num_threads);

      let t0 = RelativeClock::new();
      let cancel_token = CancellationToken::new();
      let last_message = Arc::new(AtomicU64::new(0));

      for worker_id in 0..num_threads {
         let timeout_ms = cfg.worker_timeout_ms;
         let cancel_token = cancel_token.clone();
         let last_message = last_message.clone();
         let embedder = embedder.clone();
         let mut rx = rx.clone().into_stream();
         let handle = tokio::spawn(async move {
            let mut timer = tokio::time::interval(timeout);
            timer.set_missed_tick_behavior(MissedTickBehavior::Skip);
            tracing::debug!(worker_id, "embedding worker started");

            let mut ops = FuturesUnordered::new();

            loop {
               // if we have more than 10 operations, wait for one to complete
               if ops.len() > 10 {
                  ops.next().await;
                  continue;
               }

               // Receive a message from the channel
               let msg: WorkerMessage = tokio::select! {
                  _ = ops.next() => {
                     // An operation completed
                     continue;
                  }
                  Some(msg) = rx.next() => {
                     // Update the last message time
                     last_message.fetch_max(t0.time(), Ordering::Release);
                     msg
                  }
                  _ = timer.tick() => {
                     // Check if the timeout has been exceeded
                     let now = t0.time();
                     let last_msg = last_message.load(Ordering::Acquire);
                     if now - last_msg > timeout_ms {
                        // Cancel the token if the timeout has been exceeded
                        cancel_token.cancel();
                     }
                     continue;
                  }
                  () = cancel_token.cancelled() => {
                     // Cancel the token if the worker has been cancelled
                     tracing::debug!(worker_id, "embedding worker cancelled");
                     break;
                  }
                  else => {
                     // The channel has been disconnected, shut down the worker
                     tracing::error!(worker_id, "channel disconnected, shutting down");
                     break;
                  }
               };

               // Compute the embeddings
               let embedder = embedder.clone();
               ops.push(async move {
                  let result = embedder.compute_hybrid(&msg.chunk).await;
                  _ = msg.tx.send(result);
               });
            }

            tracing::debug!(worker_id, "worker shut down");
         });

         workers.push(handle);
      }

      Ok(Self { workers: Some(workers), sender: tx, batch_size: batch_sz, cancel_token, embedder })
   }

   /// Computes hybrid embeddings using the worker pool
   pub async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>> {
      if texts.is_empty() {
         return Ok(Vec::new());
      }

      let rxs: Vec<oneshot::Receiver<_>> = texts
         .chunks(self.batch_size)
         .map(|chunk| {
            let (tx, rx) = oneshot::channel();
            self
               .sender
               .send(WorkerMessage { chunk: chunk.iter().cloned().collect(), tx })
               .map_err(|_| EmbeddingError::WorkerClosed)?;
            Ok(rx)
         })
         .collect::<Result<Vec<_>>>()?;

      let mut messages = Vec::with_capacity(texts.len());
      for rx in rxs {
         messages.extend(rx.await.map_err(|_| EmbeddingError::WorkCancelled)??);
      }
      Ok(messages)
   }
}

impl Drop for EmbedWorker {
   fn drop(&mut self) {
      self.cancel_token.cancel();
   }
}

#[async_trait::async_trait]
impl Embedder for EmbedWorker {
   async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>> {
      Self::compute_hybrid(self, texts).await
   }

   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding> {
      self.embedder.encode_query(text).await
   }

   fn is_ready(&self) -> bool {
      self.workers.is_some()
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[tokio::test]
   async fn test_worker_creation() {
      let worker = EmbedWorker::new();
      assert!(worker.is_ok());
   }

   #[tokio::test]
   async fn test_compute_empty() {
      let worker = EmbedWorker::new().unwrap();
      let result = worker.compute_hybrid(&[]).await;
      assert!(result.is_ok());
      assert_eq!(result.unwrap().len(), 0);
   }
}
