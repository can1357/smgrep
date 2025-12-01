//! Long-running daemon server command.
//!
//! Starts a background server that maintains an index, watches for file
//! changes, and responds to search requests over Unix domain sockets.
//! Automatically shuts down after a period of inactivity.

use std::{
   path::{Path, PathBuf},
   sync::{
      Arc,
      atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
   },
   time::{Duration, Instant},
};

use console::style;
use futures::stream::{self, StreamExt};
use parking_lot::Mutex;
use tokio::{signal, sync::watch, time};

use crate::{
   Result, Str,
   chunker::Chunker,
   config,
   embed::Embedder,
   file::{FileSystem, FileWatcher, IgnorePatterns, LocalFileSystem, WatchAction},
   git,
   index_lock::IndexLock,
   ipc::{self, Request, Response, ServerStatus},
   meta::{FileHash, MetaStore},
   store::{LanceStore, SearchParams, Store},
   types::{PreparedChunk, SearchResponse, SearchResult, SearchStatus, VectorRecord},
   usock, version,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::embed::candle::CandleEmbedder;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use crate::embed::worker::EmbedWorker;

/// The main server state managing indexing, search, and file watching.
struct Server {
   store:         Arc<dyn Store>,
   embedder:      Arc<dyn Embedder>,
   chunker:       Chunker,
   meta_store:    Mutex<MetaStore>,
   store_id:      String,
   root:          PathBuf,
   indexing:      AtomicBool,
   progress:      AtomicU8,
   launch_time:   Instant,
   last_activity: AtomicU64,
   shutdown:      watch::Sender<bool>,
}

impl Server {
   fn clock(&self) -> u64 {
      self.launch_time.elapsed().as_millis() as u64
   }

   fn touch(&self) {
      self
         .last_activity
         .fetch_max(self.clock(), Ordering::Relaxed);
   }

   fn idle_duration(&self) -> Duration {
      let timestamp = self
         .clock()
         .saturating_sub(self.last_activity.load(Ordering::Relaxed));
      Duration::from_millis(timestamp)
   }
}

/// Executes the serve command, starting a long-running daemon server.
pub async fn execute(path: Option<PathBuf>, store_id: Option<String>) -> Result<()> {
   let root = std::env::current_dir()?;
   let serve_path = path.unwrap_or_else(|| root.clone());

   let resolved_store_id = store_id.map_or_else(|| git::resolve_store_id(&serve_path), Ok)?;

   let listener = match usock::Listener::bind(&resolved_store_id).await {
      Ok(l) => l,
      Err(e) if e.to_string().contains("already running") => {
         println!("{}", style("Server already running").yellow());
         return Ok(());
      },
      Err(e) => return Err(e),
   };

   println!("{}", style("Starting smgrep server...").green().bold());
   println!("Listening: {}", style(listener.local_addr()).cyan());
   println!("Path: {}", style(serve_path.display()).dim());
   println!("Store ID: {}", style(&resolved_store_id).cyan());

   let store: Arc<dyn Store> = Arc::new(LanceStore::new()?);
   // EmbedWorker's parallel workers cause hangs on Metal. Use CandleEmbedder directly.
   // This matches the single-threaded pattern used by huggingface/text-embeddings-inference.
   #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
   let embedder: Arc<dyn Embedder> = Arc::new(CandleEmbedder::new()?);
   #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
   let embedder: Arc<dyn Embedder> = Arc::new(EmbedWorker::new()?);

   if !embedder.is_ready() {
      println!("{}", style("Waiting for embedder to initialize...").yellow());
      time::sleep(Duration::from_millis(500)).await;
   }

   let mut meta_store = MetaStore::load(&resolved_store_id)?;
   let model_changed = meta_store.model_mismatch();

   if model_changed {
      store.delete_store(&resolved_store_id).await?;
      meta_store.reset_for_model_change();
      meta_store.save()?;
   }

   let is_empty = store.is_empty(&resolved_store_id).await?;
   let needs_initial_index = is_empty || model_changed;

   let (shutdown_tx, shutdown_rx) = watch::channel(false);

   let server = Arc::new(Server {
      store,
      embedder,
      chunker: Chunker::default(),
      meta_store: Mutex::new(meta_store),
      store_id: resolved_store_id,
      root: serve_path,
      indexing: AtomicBool::new(needs_initial_index),
      progress: AtomicU8::new(0),
      last_activity: AtomicU64::new(0),
      launch_time: Instant::now(),
      shutdown: shutdown_tx.clone(),
   });

   if needs_initial_index {
      let reason = if model_changed {
         "Embedding models changed; rebuilding index..."
      } else {
         "Store empty, performing initial index..."
      };

      println!("{}", style(reason).yellow());
      let server_clone = Arc::clone(&server);
      tokio::spawn(async move {
         if let Err(e) = server_clone.initial_sync().await {
            tracing::error!("Initial sync failed: {}", e);
         }
      });
   }

   let _watcher = server.start_watcher()?;

   let idle_server = Arc::clone(&server);
   let idle_shutdown = shutdown_tx.clone();
   let cfg = config::get();
   let idle_timeout = Duration::from_secs(cfg.idle_timeout_secs);
   let idle_check_interval = Duration::from_secs(cfg.idle_check_interval_secs);
   tokio::spawn(async move {
      loop {
         time::sleep(idle_check_interval).await;
         if idle_server.idle_duration() > idle_timeout {
            println!("{}", style("Idle timeout reached, shutting down...").yellow());
            let _ = idle_shutdown.send(true);
            break;
         }
      }
   });

   println!("\n{}", style("Server listening").green());
   println!("{}", style("Press Ctrl+C to stop").dim());

   let accept_server = Arc::clone(&server);
   let mut accept_shutdown = shutdown_rx.clone();
   let accept_handle = tokio::spawn(async move {
      loop {
         tokio::select! {
            result = listener.accept() => {
               match result {
                  Ok(stream) => {
                     let client_server = Arc::clone(&accept_server);
                     tokio::spawn(async move { client_server.handle_client(stream).await });
                  }
                  Err(e) => {
                     tracing::error!("Accept error: {}", e);
                  }
               }
            }
            _ = accept_shutdown.changed() => {
               if *accept_shutdown.borrow() {
                  break;
               }
            }
         }
      }
   });

   tokio::select! {
      _ = signal::ctrl_c() => {
         println!("\n{}", style("Shutting down...").yellow());
         let _ = shutdown_tx.send(true);
      }
      () = async {
         let mut rx = shutdown_rx.clone();
         loop {
            rx.changed().await.ok();
            if *rx.borrow() {
               break;
            }
         }
      } => {}
   }

   accept_handle.abort();

   println!("{}", style("Server stopped").green());
   Ok(())
}

impl Server {
   async fn handle_client(self: &Arc<Self>, mut stream: usock::Stream) {
      self.touch();

      let mut buffer = ipc::SocketBuffer::new();
      let mut shutting_down = false;

      loop {
         let request: Request = match buffer.recv(&mut stream).await {
            Ok(req) => req,
            Err(e) => {
               if e.to_string().contains("failed to read length") {
                  break;
               }
               tracing::debug!("Client read error: {}", e);
               break;
            },
         };

         self.touch();

         let response = match request {
            Request::Hello { .. } => Response::Hello { git_hash: version::GIT_HASH.to_string() },
            Request::Search { query, limit, path, rerank } => {
               self.handle_search(query, limit, path, rerank).await
            },
            Request::Health => Response::Health {
               status: ServerStatus {
                  indexing: self.indexing.load(Ordering::Relaxed),
                  progress: self.progress.load(Ordering::Relaxed),
                  files:    0,
               },
            },
            Request::Shutdown => {
               shutting_down = true;
               Response::Shutdown { success: true }
            },
         };

         if let Err(e) = buffer.send(&mut stream, &response).await {
            tracing::debug!("Client write error: {}", e);
            break;
         }

         if shutting_down {
            let _ = self.shutdown.send(true);
            break;
         }
      }
   }

   async fn handle_search(
      &self,
      query: String,
      limit: usize,
      path: Option<PathBuf>,
      rerank: bool,
   ) -> Response {
      if query.is_empty() {
         return Response::Error { message: "query is required".to_string() };
      }

      let search_path = path.as_ref().map(|p| {
         if p.is_absolute() {
            p.clone()
         } else {
            self.root.join(p)
         }
      });

      let query_emb = match self.embedder.encode_query(&query).await {
         Ok(emb) => emb,
         Err(e) => return Response::Error { message: format!("embedding failed: {e}") },
      };

      let search_result = self
         .store
         .search(SearchParams {
            store_id: &self.store_id,
            query_text: &query,
            query_vector: &query_emb.dense,
            query_colbert: &query_emb.colbert,
            limit,
            path_filter: search_path.as_deref(),
            rerank,
         })
         .await;

      match search_result {
         Ok(response) => {
            let results = response
               .results
               .into_iter()
               .map(|r| {
                  let rel_path = r
                     .path
                     .strip_prefix(&self.root)
                     .map(PathBuf::from)
                     .unwrap_or(r.path);

                  SearchResult {
                     path:       rel_path,
                     content:    r.content,
                     score:      r.score,
                     start_line: r.start_line,
                     num_lines:  r.num_lines,
                     chunk_type: r.chunk_type,
                     is_anchor:  r.is_anchor,
                  }
               })
               .collect();

            let is_indexing = self.indexing.load(Ordering::Relaxed);
            let progress_val = self.progress.load(Ordering::Relaxed);

            Response::Search(SearchResponse {
               results,
               status: if is_indexing {
                  SearchStatus::Indexing
               } else {
                  SearchStatus::Ready
               },
               progress: if is_indexing {
                  Some(progress_val)
               } else {
                  None
               },
            })
         },
         Err(e) => Response::Error { message: format!("search failed: {e}") },
      }
   }

   async fn initial_sync(self: &Arc<Self>) -> Result<()> {
      let _lock = IndexLock::acquire(&self.store_id)?;

      let fs = LocalFileSystem::new();
      let files: Vec<PathBuf> = fs.get_files(&self.root)?.collect();

      let total = files.len();
      if total == 0 {
         self.indexing.store(false, Ordering::Relaxed);
         self.progress.store(100, Ordering::Relaxed);
         tracing::info!("Initial sync complete: 0/0 files indexed");
         return Ok(());
      }

      let results: Vec<_> = stream::iter(files.into_iter().enumerate())
         .map(|(i, file_path)| {
            let server = Arc::clone(self);
            async move {
               let result = server.process_file(&file_path).await;
               (i, file_path, result)
            }
         })
         .buffer_unordered(8)
         .collect()
         .await;

      let mut indexed = 0;
      for (completed, (_i, file_path, result)) in results.into_iter().enumerate() {
         match result {
            Ok(()) => indexed += 1,
            Err(e) => tracing::warn!("Failed to index {}: {}", file_path.display(), e),
         }

         let pct = ((completed + 1) * 100 / total).min(100) as u8;
         self.progress.store(pct, Ordering::Relaxed);
      }

      self.indexing.store(false, Ordering::Relaxed);
      self.progress.store(100, Ordering::Relaxed);

      tracing::info!("Initial sync complete: {}/{} files indexed", indexed, total);
      Ok(())
   }

   async fn process_file(&self, file_path: &Path) -> Result<()> {
      let content = tokio::fs::read(file_path).await?;

      if content.is_empty() {
         return Ok(());
      }
      let content_str = Str::from_utf8_lossy(&content);

      let hash = FileHash::sum(&content);

      {
         let meta = self.meta_store.lock();
         if let Some(existing_hash) = meta.get_hash(file_path)
            && existing_hash == hash
         {
            return Ok(());
         }
      }

      let chunks = self.chunker.chunk(&content_str, file_path).await?;
      if chunks.is_empty() {
         return Ok(());
      }

      let path_arc = std::sync::Arc::new(file_path.to_path_buf());
      let prepared: Vec<PreparedChunk> = chunks
         .iter()
         .enumerate()
         .map(|(i, chunk)| {
            let context_prev = if i > 0 {
               Some(chunks[i - 1].content.clone())
            } else {
               None
            };
            let context_next = if i < chunks.len() - 1 {
               Some(chunks[i + 1].content.clone())
            } else {
               None
            };
            PreparedChunk {
               id: format!("{}:{}", file_path.display(), i),
               path: std::sync::Arc::clone(&path_arc),
               hash,
               content: chunk.content.clone(),
               start_line: chunk.start_line as u32,
               end_line: chunk.end_line as u32,
               chunk_index: Some(i as u32),
               is_anchor: chunk.is_anchor,
               chunk_type: chunk.chunk_type,
               context_prev,
               context_next,
            }
         })
         .collect();

      let texts: Vec<Str> = prepared.iter().map(|c| c.content.clone()).collect();
      let embeddings = self.embedder.compute_hybrid(&texts).await?;

      let records: Vec<VectorRecord> = prepared
         .into_iter()
         .zip(embeddings)
         .map(|(prep, emb)| VectorRecord {
            id:            prep.id,
            path:          prep.path,
            hash:          prep.hash,
            content:       prep.content,
            start_line:    prep.start_line,
            end_line:      prep.end_line,
            chunk_index:   prep.chunk_index,
            is_anchor:     prep.is_anchor,
            chunk_type:    prep.chunk_type,
            context_prev:  prep.context_prev,
            context_next:  prep.context_next,
            vector:        emb.dense,
            colbert:       emb.colbert,
            colbert_scale: emb.colbert_scale,
         })
         .collect();

      self.store.insert_batch(&self.store_id, records).await?;

      {
         let mut meta = self.meta_store.lock();
         meta.set_hash(file_path, hash);
      }
      self.meta_store.lock().save()?;

      Ok(())
   }

   fn start_watcher(self: &Arc<Self>) -> Result<FileWatcher> {
      let ignore_patterns = IgnorePatterns::new(&self.root);
      let server = Arc::clone(self);
      let watcher = FileWatcher::new(self.root.clone(), ignore_patterns, move |changes| {
         let server = Arc::clone(&server);
         tokio::spawn(async move {
            let _lock = match IndexLock::acquire(&server.store_id) {
               Ok(lock) => lock,
               Err(e) => {
                  tracing::error!("Failed to acquire index lock: {e}");
                  return;
               },
            };

            let results: Vec<_> = stream::iter(changes)
               .map(|(path, action)| {
                  let server = Arc::clone(&server);
                  async move {
                     let result = match action {
                        WatchAction::Delete => {
                           if let Err(e) = server.store.delete_file(&server.store_id, &path).await {
                              tracing::error!("Failed to delete file from store: {}", e);
                           }
                           {
                              let mut meta = server.meta_store.lock();
                              meta.remove(&path);
                           }
                           let value = server.meta_store.lock().save();
                           if let Err(e) = value {
                              tracing::error!("Failed to save meta after delete: {}", e);
                           }
                           Ok(())
                        },
                        WatchAction::Upsert => server.process_file(&path).await,
                     };
                     (path, action, result)
                  }
               })
               .buffer_unordered(8)
               .collect()
               .await;

            for (path, action, result) in results {
               if let Err(e) = result {
                  match action {
                     WatchAction::Delete => {
                        tracing::error!("Failed to handle delete for {}: {}", path.display(), e);
                     },
                     WatchAction::Upsert => {
                        tracing::error!("Failed to process changed file {}: {}", path.display(), e);
                     },
                  }
               }
            }
         });
      })?;

      Ok(watcher)
   }
}
