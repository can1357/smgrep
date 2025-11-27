use std::{
   path::{Path, PathBuf},
   sync::{
      Arc,
      atomic::{AtomicBool, AtomicU8, Ordering},
   },
};

use anyhow::{Context, Result};
use axum::{
   Json, Router,
   extract::State,
   http::{HeaderMap, StatusCode},
   response::{IntoResponse, Response},
   routing::{get, post},
};
use console::style;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::signal;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::{
   chunker,
   embed::Embedder,
   file::{FileSystem, FileWatcher, LocalFileSystem, WatchAction},
   meta::MetaStore,
   store::Store,
   types::{PreparedChunk, VectorRecord},
};

const MAX_REQUEST_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerLock {
   pub port:       u16,
   pub pid:        u32,
   pub auth_token: String,
}

fn server_lock_path(root: &Path) -> PathBuf {
   root.join(".rsgrep").join("server.json")
}

pub fn read_server_lock(root: &Path) -> Result<Option<ServerLock>> {
   let lock_path = server_lock_path(root);
   if !lock_path.exists() {
      return Ok(None);
   }

   let content = std::fs::read_to_string(&lock_path).context("failed to read server lock file")?;
   let lock: ServerLock =
      serde_json::from_str(&content).context("failed to parse server lock file")?;

   Ok(Some(lock))
}

pub fn write_server_lock(root: &Path, lock: &ServerLock) -> Result<()> {
   let lock_path = server_lock_path(root);
   if let Some(parent) = lock_path.parent() {
      std::fs::create_dir_all(parent).context("failed to create .rsgrep directory")?;
   }

   let content = serde_json::to_string_pretty(lock).context("failed to serialize server lock")?;
   std::fs::write(&lock_path, content).context("failed to write server lock file")?;

   Ok(())
}

pub fn remove_server_lock(root: &Path) -> Result<()> {
   let lock_path = server_lock_path(root);
   if lock_path.exists() {
      std::fs::remove_file(&lock_path).context("failed to remove server lock file")?;
   }
   Ok(())
}

#[derive(Clone)]
struct ServerState {
   store:      Arc<dyn Store>,
   embedder:   Arc<dyn Embedder>,
   store_id:   String,
   auth_token: String,
   root:       PathBuf,
   indexing:   Arc<AtomicBool>,
   progress:   Arc<AtomicU8>,
   meta_store: Arc<parking_lot::Mutex<MetaStore>>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
   query:  String,
   #[serde(default)]
   limit:  Option<usize>,
   #[serde(default)]
   path:   Option<String>,
   #[serde(default = "default_rerank")]
   rerank: bool,
}

fn default_rerank() -> bool {
   true
}

#[derive(Debug, Serialize)]
struct SearchResponseJson {
   results:  Vec<JsonResult>,
   status:   String,
   #[serde(skip_serializing_if = "Option::is_none")]
   progress: Option<u8>,
}

#[derive(Debug, Serialize)]
struct JsonResult {
   path:       String,
   score:      f32,
   content:    String,
   #[serde(skip_serializing_if = "Option::is_none")]
   chunk_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
   status: String,
}

async fn health_handler() -> Json<HealthResponse> {
   Json(HealthResponse { status: "ready".to_string() })
}

async fn search_handler(
   State(state): State<ServerState>,
   headers: HeaderMap,
   Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponseJson>, ApiError> {
   let auth_header = headers
      .get("authorization")
      .and_then(|v| v.to_str().ok())
      .unwrap_or("");

   let provided_token = auth_header.strip_prefix("Bearer ").unwrap_or(auth_header);

   if provided_token != state.auth_token {
      return Err(ApiError::Unauthorized);
   }

   if req.query.is_empty() {
      return Err(ApiError::BadRequest("query is required".to_string()));
   }

   let limit = req.limit.unwrap_or(25);
   let search_path = req.path.as_ref().map(|p| {
      if Path::new(p).is_absolute() {
         PathBuf::from(p)
      } else {
         state.root.join(p)
      }
   });

   let query_emb = state
      .embedder
      .encode_query(&req.query)
      .await
      .map_err(|e| ApiError::Internal(format!("embedding failed: {}", e)))?;

   let path_filter = search_path
      .as_ref()
      .map(|p| p.to_string_lossy().to_string());

   let response = state
      .store
      .search(
         &state.store_id,
         &req.query,
         &query_emb.dense,
         &query_emb.colbert,
         limit,
         path_filter.as_deref(),
         req.rerank,
      )
      .await
      .map_err(|e| ApiError::Internal(format!("search failed: {}", e)))?;

   let results: Vec<JsonResult> = response
      .results
      .into_iter()
      .map(|r| {
         let rel_path = r
            .path
            .strip_prefix(&state.root.to_string_lossy().to_string())
            .unwrap_or(&r.path)
            .trim_start_matches('/')
            .to_string();

         JsonResult {
            path:       rel_path,
            score:      r.score,
            content:    format_dense_snippet(&r.content),
            chunk_type: Some(format!("{:?}", r.chunk_type).to_lowercase()),
         }
      })
      .collect();

   let is_indexing = state.indexing.load(Ordering::Relaxed);
   let progress_val = state.progress.load(Ordering::Relaxed);

   Ok(Json(SearchResponseJson {
      results,
      status: if is_indexing { "indexing" } else { "ready" }.to_string(),
      progress: if is_indexing {
         Some(progress_val)
      } else {
         None
      },
   }))
}

fn format_dense_snippet(content: &str) -> String {
   let lines: Vec<&str> = content.lines().take(5).collect();
   if lines.len() < content.lines().count() {
      format!("{}\n...", lines.join("\n"))
   } else {
      lines.join("\n")
   }
}

#[derive(Debug)]
enum ApiError {
   Unauthorized,
   BadRequest(String),
   PayloadTooLarge,
   Internal(String),
}

impl IntoResponse for ApiError {
   fn into_response(self) -> Response {
      let (status, message) = match self {
         ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized".to_string()),
         ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
         ApiError::PayloadTooLarge => {
            (StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large".to_string())
         },
         ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
      };

      let body = serde_json::json!({ "error": message });
      (status, Json(body)).into_response()
   }
}

pub async fn execute(port: u16, path: Option<PathBuf>, store_id: Option<String>) -> Result<()> {
   let root = std::env::current_dir()?;
   let serve_path = path.unwrap_or_else(|| root.clone());

   let resolved_store_id = store_id.unwrap_or_else(|| {
      serve_path
         .file_name()
         .and_then(|s| s.to_str())
         .unwrap_or("default")
         .to_string()
   });

   let auth_token = Uuid::new_v4().to_string();
   let pid = std::process::id();

   let lock = ServerLock { port, pid, auth_token: auth_token.clone() };

   write_server_lock(&serve_path, &lock)?;

   println!("{}", style("Starting rsgrep server...").green().bold());
   println!("Port: {}", style(port.to_string()).cyan());
   println!("Path: {}", style(serve_path.display()).dim());
   println!("Store ID: {}", style(&resolved_store_id).cyan());

   let store: Arc<dyn Store> = Arc::new(crate::store::LanceStore::new()?);
   let embedder: Arc<dyn Embedder> = Arc::new(crate::embed::candle::CandleEmbedder::new()?);

   if !embedder.is_ready() {
      println!("{}", style("Waiting for embedder to initialize...").yellow());
      tokio::time::sleep(std::time::Duration::from_millis(500)).await;
   }

   let meta_store = MetaStore::load(&resolved_store_id)?;
   let is_empty = store.is_empty(&resolved_store_id).await?;

   let indexing = Arc::new(AtomicBool::new(false));
   let progress = Arc::new(AtomicU8::new(0));
   let meta_store_arc = Arc::new(parking_lot::Mutex::new(meta_store));

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

   let state = ServerState {
      store:      Arc::clone(&store),
      embedder:   Arc::clone(&embedder),
      store_id:   resolved_store_id.clone(),
      auth_token: auth_token.clone(),
      root:       serve_path.clone(),
      indexing:   Arc::clone(&indexing),
      progress:   Arc::clone(&progress),
      meta_store: Arc::clone(&meta_store_arc),
   };

   let _watcher = start_watcher(
      serve_path.clone(),
      Arc::clone(&store),
      Arc::clone(&embedder),
      resolved_store_id.clone(),
      Arc::clone(&meta_store_arc),
   )?;

   let app = Router::new()
      .route("/health", get(health_handler))
      .route("/search", post(search_handler))
      .layer(CorsLayer::permissive())
      .with_state(state);

   let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
      .await
      .context("failed to bind to port")?;

   println!("\n{}", style(format!("Server listening on http://127.0.0.1:{}", port)).green());
   println!("{}", style("Press Ctrl+C to stop").dim());

   let server = axum::serve(listener, app);

   tokio::select! {
      result = server => {
         result.context("server error")?;
      }
      _ = signal::ctrl_c() => {
         println!("\n{}", style("Shutting down...").yellow());
      }
   }

   remove_server_lock(&serve_path)?;
   println!("{}", style("Server stopped").green());

   Ok(())
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
