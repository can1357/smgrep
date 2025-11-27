use std::{
   path::{Path, PathBuf},
   sync::{
      Arc,
      atomic::{AtomicBool, AtomicU8, Ordering},
   },
   time::{Duration, Instant},
};

use anyhow::{Context, Result};
use console::style;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio::{
   net::{UnixListener, UnixStream},
   signal,
   sync::watch,
};

use crate::{
   chunker,
   embed::Embedder,
   file::{FileSystem, FileWatcher, LocalFileSystem, WatchAction},
   ipc::{self, Request, Response, ServerStatus},
   meta::MetaStore,
   store::Store,
   types::{PreparedChunk, SearchResponse, SearchStatus, VectorRecord},
};

const IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(60);

struct ServerState {
   store:         Arc<dyn Store>,
   embedder:      Arc<dyn Embedder>,
   store_id:      String,
   root:          PathBuf,
   indexing:      Arc<AtomicBool>,
   progress:      Arc<AtomicU8>,
   last_activity: Arc<Mutex<Instant>>,
}

impl ServerState {
   fn touch(&self) {
      *self.last_activity.lock() = Instant::now();
   }

   fn idle_duration(&self) -> Duration {
      self.last_activity.lock().elapsed()
   }
}

pub async fn execute(path: Option<PathBuf>, store_id: Option<String>) -> Result<()> {
   let root = std::env::current_dir()?;
   let serve_path = path.unwrap_or_else(|| root.clone());

   let resolved_store_id = store_id
      .map(Ok)
      .unwrap_or_else(|| crate::git::resolve_store_id(&serve_path))?;

   let socket_path = ipc::socket_path(&resolved_store_id);
   if let Some(parent) = socket_path.parent() {
      std::fs::create_dir_all(parent).context("failed to create socks directory")?;
   }

   if socket_path.exists() {
      if try_connect(&socket_path).await {
         println!("{}", style("Server already running").yellow());
         return Ok(());
      }
      std::fs::remove_file(&socket_path).context("failed to remove stale socket")?;
   }

   let listener = UnixListener::bind(&socket_path).context("failed to bind socket")?;

   println!("{}", style("Starting rsgrep server...").green().bold());
   println!("Socket: {}", style(socket_path.display()).cyan());
   println!("Path: {}", style(serve_path.display()).dim());
   println!("Store ID: {}", style(&resolved_store_id).cyan());

   let store: Arc<dyn Store> = Arc::new(crate::store::LanceStore::new()?);
   let embedder: Arc<dyn Embedder> = Arc::new(crate::embed::candle::CandleEmbedder::new()?);

   if !embedder.is_ready() {
      println!("{}", style("Waiting for embedder to initialize...").yellow());
      tokio::time::sleep(Duration::from_millis(500)).await;
   }

   let meta_store = MetaStore::load(&resolved_store_id)?;
   let is_empty = store.is_empty(&resolved_store_id).await?;

   let indexing = Arc::new(AtomicBool::new(false));
   let progress = Arc::new(AtomicU8::new(0));
   let meta_store_arc = Arc::new(parking_lot::Mutex::new(meta_store));
   let last_activity = Arc::new(Mutex::new(Instant::now()));

   if is_empty {
      println!("{}", style("Store empty, performing initial index...").yellow());
      indexing.store(true, Ordering::Relaxed);

      let store_clone = Arc::clone(&store);
      let embedder_clone = Arc::clone(&embedder);
      let store_id_clone = resolved_store_id.clone();
      let root_clone = serve_path.clone();
      let indexing_clone = Arc::clone(&indexing);
      let progress_clone = Arc::clone(&progress);
      let meta_clone = Arc::clone(&meta_store_arc);

      tokio::spawn(async move {
         if let Err(e) = initial_sync(
            store_clone,
            embedder_clone,
            &store_id_clone,
            &root_clone,
            indexing_clone,
            progress_clone,
            meta_clone,
         )
         .await
         {
            tracing::error!("Initial sync failed: {}", e);
         }
      });
   }

   let state = Arc::new(ServerState {
      store: Arc::clone(&store),
      embedder: Arc::clone(&embedder),
      store_id: resolved_store_id.clone(),
      root: serve_path.clone(),
      indexing: Arc::clone(&indexing),
      progress: Arc::clone(&progress),
      last_activity,
   });

   let _watcher = start_watcher(
      serve_path.clone(),
      Arc::clone(&store),
      Arc::clone(&embedder),
      resolved_store_id.clone(),
      Arc::clone(&meta_store_arc),
   )?;

   let (shutdown_tx, shutdown_rx) = watch::channel(false);

   let idle_state = Arc::clone(&state);
   let idle_shutdown = shutdown_tx.clone();
   tokio::spawn(async move {
      loop {
         tokio::time::sleep(IDLE_CHECK_INTERVAL).await;
         if idle_state.idle_duration() > IDLE_TIMEOUT {
            println!("{}", style("Idle timeout reached, shutting down...").yellow());
            let _ = idle_shutdown.send(true);
            break;
         }
      }
   });

   println!("\n{}", style("Server listening").green());
   println!("{}", style("Press Ctrl+C to stop").dim());

   let accept_state = Arc::clone(&state);
   let mut accept_shutdown = shutdown_rx.clone();
   let accept_handle = tokio::spawn(async move {
      loop {
         tokio::select! {
            result = listener.accept() => {
               match result {
                  Ok((stream, _)) => {
                     let client_state = Arc::clone(&accept_state);
                     tokio::spawn(handle_client(stream, client_state));
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
      _ = async {
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
   let _ = std::fs::remove_file(&socket_path);

   println!("{}", style("Server stopped").green());
   Ok(())
}

async fn try_connect(socket_path: &Path) -> bool {
   UnixStream::connect(socket_path).await.is_ok()
}

async fn handle_client(mut stream: UnixStream, state: Arc<ServerState>) {
   state.touch();

   let mut buffer = ipc::SocketBuffer::new();

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

      state.touch();

      let response = match request {
         Request::Search { query, limit, path, rerank } => {
            handle_search(&state, query, limit, path, rerank).await
         },
         Request::Health => Response::Health {
            status: ServerStatus {
               indexing: state.indexing.load(Ordering::Relaxed),
               progress: state.progress.load(Ordering::Relaxed),
               files:    0,
            },
         },
         Request::Shutdown => {
            let _ = buffer
               .send(&mut stream, &Response::Shutdown { success: true })
               .await;
            std::process::exit(0);
         },
      };

      if let Err(e) = buffer.send(&mut stream, &response).await {
         tracing::debug!("Client write error: {}", e);
         break;
      }
   }
}

async fn handle_search(
   state: &ServerState,
   query: String,
   limit: usize,
   path: Option<String>,
   rerank: bool,
) -> Response {
   if query.is_empty() {
      return Response::Error { message: "query is required".to_string() };
   }

   let search_path = path.as_ref().map(|p| {
      if Path::new(p).is_absolute() {
         PathBuf::from(p)
      } else {
         state.root.join(p)
      }
   });

   let query_emb = match state.embedder.encode_query(&query).await {
      Ok(emb) => emb,
      Err(e) => return Response::Error { message: format!("embedding failed: {}", e) },
   };

   let path_filter = search_path
      .as_ref()
      .map(|p| p.to_string_lossy().to_string());

   let search_result = state
      .store
      .search(
         &state.store_id,
         &query,
         &query_emb.dense,
         &query_emb.colbert,
         limit,
         path_filter.as_deref(),
         rerank,
      )
      .await;

   match search_result {
      Ok(response) => {
         let results = response
            .results
            .into_iter()
            .map(|r| {
               let rel_path = r
                  .path
                  .strip_prefix(&state.root.to_string_lossy().to_string())
                  .unwrap_or(&r.path)
                  .trim_start_matches('/')
                  .to_string();

               crate::types::SearchResult {
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

         let is_indexing = state.indexing.load(Ordering::Relaxed);
         let progress_val = state.progress.load(Ordering::Relaxed);

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
      Err(e) => Response::Error { message: format!("search failed: {}", e) },
   }
}

async fn initial_sync(
   store: Arc<dyn Store>,
   embedder: Arc<dyn Embedder>,
   store_id: &str,
   root: &Path,
   indexing: Arc<AtomicBool>,
   progress: Arc<AtomicU8>,
   meta_store: Arc<parking_lot::Mutex<MetaStore>>,
) -> Result<()> {
   let fs = LocalFileSystem::new();
   let files: Vec<PathBuf> = fs.get_files(root)?.collect();

   let total = files.len();
   let mut indexed = 0;

   for (i, file_path) in files.iter().enumerate() {
      if let Err(e) = process_file(&store, &embedder, store_id, file_path, &meta_store).await {
         tracing::warn!("Failed to index {}: {}", file_path.display(), e);
      } else {
         indexed += 1;
      }

      let pct = ((i + 1) * 100 / total).min(100) as u8;
      progress.store(pct, Ordering::Relaxed);
   }

   indexing.store(false, Ordering::Relaxed);
   progress.store(100, Ordering::Relaxed);

   tracing::info!("Initial sync complete: {}/{} files indexed", indexed, total);
   Ok(())
}

async fn process_file(
   store: &Arc<dyn Store>,
   embedder: &Arc<dyn Embedder>,
   store_id: &str,
   file_path: &Path,
   meta_store: &Arc<parking_lot::Mutex<MetaStore>>,
) -> Result<()> {
   let content = tokio::fs::read(file_path)
      .await
      .context("failed to read file")?;

   if content.is_empty() {
      return Ok(());
   }

   let content_str = String::from_utf8_lossy(&content).to_string();
   let hash = compute_hash(&content);

   {
      let meta = meta_store.lock();
      if let Some(existing_hash) = meta.get_hash(&file_path.to_string_lossy())
         && existing_hash == &hash
      {
         return Ok(());
      }
   }

   let chunker = chunker::create_chunker(file_path);
   let chunks = chunker.chunk(&content_str, file_path)?;

   if chunks.is_empty() {
      return Ok(());
   }

   let prepared: Vec<PreparedChunk> = chunks
      .into_iter()
      .enumerate()
      .map(|(i, chunk)| PreparedChunk {
         id:           format!("{}:{}", file_path.display(), i),
         path:         file_path.to_string_lossy().to_string(),
         hash:         hash.clone(),
         content:      chunk.content,
         start_line:   chunk.start_line as u32,
         end_line:     chunk.end_line as u32,
         chunk_index:  Some(i as u32),
         is_anchor:    chunk.is_anchor,
         chunk_type:   chunk.chunk_type,
         context_prev: chunk.context.first().cloned(),
         context_next: chunk.context.last().cloned(),
      })
      .collect();

   let texts: Vec<String> = prepared.iter().map(|c| c.content.clone()).collect();
   let embeddings = embedder
      .compute_hybrid(&texts)
      .await
      .context("failed to compute embeddings")?;

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

   store
      .insert_batch(store_id, records)
      .await
      .context("failed to insert batch")?;

   {
      let mut meta = meta_store.lock();
      meta.set_hash(file_path.to_string_lossy().to_string(), hash);
      meta.save()?;
   }

   Ok(())
}

fn compute_hash(content: &[u8]) -> String {
   let mut hasher = Sha256::new();
   hasher.update(content);
   hex::encode(hasher.finalize())
}

fn start_watcher(
   root: PathBuf,
   store: Arc<dyn Store>,
   embedder: Arc<dyn Embedder>,
   store_id: String,
   meta_store: Arc<parking_lot::Mutex<MetaStore>>,
) -> Result<FileWatcher> {
   let ignore_patterns = crate::file::IgnorePatterns::new(&root);
   let watcher = FileWatcher::new(root.clone(), ignore_patterns, move |changes| {
      let store = Arc::clone(&store);
      let embedder = Arc::clone(&embedder);
      let store_id = store_id.clone();
      let meta_store = Arc::clone(&meta_store);

      tokio::spawn(async move {
         for (path, action) in changes {
            match action {
               WatchAction::Delete => {
                  if let Err(e) = store.delete_file(&store_id, &path.to_string_lossy()).await {
                     tracing::error!("Failed to delete file from store: {}", e);
                  }
                  let mut meta = meta_store.lock();
                  meta.remove(&path.to_string_lossy());
                  if let Err(e) = meta.save() {
                     tracing::error!("Failed to save meta after delete: {}", e);
                  }
               },
               WatchAction::Upsert => {
                  if let Err(e) =
                     process_file(&store, &embedder, &store_id, &path, &meta_store).await
                  {
                     tracing::error!("Failed to process changed file: {}", e);
                  }
               },
            }
         }
      });
   })?;

   Ok(watcher)
}
