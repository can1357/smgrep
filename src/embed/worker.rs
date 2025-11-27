use std::{sync::Arc, thread, time::Duration};

use crossbeam_channel::{Sender, bounded};
use parking_lot::Mutex;

use crate::{
   config::{batch_size, threads, worker_timeout_ms},
   embed::{CandleEmbedder, Embedder, HybridEmbedding},
   error::{Result, RsgrepError},
};

enum WorkerMessage {
   Compute { texts: Vec<String>, response: Sender<Result<Vec<HybridEmbedding>>> },
   Shutdown,
}

pub struct EmbedWorker {
   workers:    Option<Vec<thread::JoinHandle<()>>>,
   sender:     Sender<WorkerMessage>,
   batch_size: usize,
}

impl EmbedWorker {
   pub fn new() -> Result<Self> {
      let num_threads = threads();
      let batch_sz = batch_size();
      let timeout = Duration::from_millis(worker_timeout_ms());

      let (tx, rx) = bounded::<WorkerMessage>(num_threads * 2);
      let rx = Arc::new(Mutex::new(rx));

      let mut workers = Vec::with_capacity(num_threads);

      for worker_id in 0..num_threads {
         let rx = Arc::clone(&rx);
         let timeout_duration = timeout;

         let handle = thread::spawn(move || {
            let embedder = match CandleEmbedder::new() {
               Ok(e) => e,
               Err(e) => {
                  tracing::error!(worker_id, "failed to initialize embedder: {}", e);
                  return;
               },
            };

            tracing::debug!(worker_id, "embedding worker started");

            loop {
               let msg = {
                  let guard = rx.lock();
                  match guard.recv_timeout(timeout_duration) {
                     Ok(msg) => msg,
                     Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        continue;
                     },
                     Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        tracing::debug!(worker_id, "channel disconnected, shutting down");
                        break;
                     },
                  }
               };

               match msg {
                  WorkerMessage::Compute { texts, response } => {
                     let result = tokio::runtime::Handle::try_current()
                        .ok()
                        .map(|handle| {
                           handle.block_on(async { embedder.compute_hybrid(&texts).await })
                        })
                        .unwrap_or_else(|| {
                           tokio::runtime::Runtime::new()
                              .map_err(|e| {
                                 RsgrepError::Embedding(format!("failed to create runtime: {}", e))
                              })
                              .and_then(|rt| {
                                 rt.block_on(async { embedder.compute_hybrid(&texts).await })
                              })
                        });

                     if response.send(result).is_err() {
                        tracing::warn!(worker_id, "failed to send result, receiver dropped");
                     }
                  },
                  WorkerMessage::Shutdown => {
                     tracing::debug!(worker_id, "received shutdown signal");
                     break;
                  },
               }
            }

            tracing::debug!(worker_id, "worker shut down");
         });

         workers.push(handle);
      }

      Ok(Self { workers: Some(workers), sender: tx, batch_size: batch_sz })
   }

   pub fn compute_batch(&self, texts: Vec<String>) -> Result<Vec<HybridEmbedding>> {
      if texts.is_empty() {
         return Ok(Vec::new());
      }

      let chunks: Vec<Vec<String>> = texts
         .chunks(self.batch_size)
         .map(|chunk| chunk.to_vec())
         .collect();

      let mut all_results = Vec::with_capacity(texts.len());

      for chunk in chunks {
         let (response_tx, response_rx) = bounded(1);

         self
            .sender
            .send(WorkerMessage::Compute { texts: chunk, response: response_tx })
            .map_err(|e| RsgrepError::Embedding(format!("failed to send work: {}", e)))?;

         let result = response_rx
            .recv()
            .map_err(|e| RsgrepError::Embedding(format!("failed to receive result: {}", e)))??;

         all_results.extend(result);
      }

      Ok(all_results)
   }

   pub fn shutdown(mut self) {
      if let Some(workers) = self.workers.take() {
         for _ in 0..workers.len() {
            let _ = self.sender.send(WorkerMessage::Shutdown);
         }

         for handle in workers {
            let _ = handle.join();
         }
      }
   }
}

impl Drop for EmbedWorker {
   fn drop(&mut self) {
      if let Some(workers) = &self.workers {
         for _ in 0..workers.len() {
            let _ = self.sender.send(WorkerMessage::Shutdown);
         }
      }
   }
}

#[async_trait::async_trait]
impl Embedder for EmbedWorker {
   async fn compute_hybrid(&self, texts: &[String]) -> Result<Vec<HybridEmbedding>> {
      self.compute_batch(texts.to_vec())
   }

   async fn encode_query(&self, text: &str) -> Result<crate::embed::QueryEmbedding> {
      let embedder = CandleEmbedder::new()?;
      embedder.encode_query(text).await
   }

   fn is_ready(&self) -> bool {
      self.workers.is_some()
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn test_worker_creation() {
      let worker = EmbedWorker::new();
      assert!(worker.is_ok());
   }

   #[tokio::test]
   async fn test_compute_empty() {
      let worker = EmbedWorker::new().unwrap();
      let result = worker.compute_batch(vec![]);
      assert!(result.is_ok());
      assert_eq!(result.unwrap().len(), 0);
   }
}
