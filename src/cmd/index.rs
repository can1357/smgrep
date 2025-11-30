//! Index creation and management command.
//!
//! Scans source files, chunks them, computes embeddings, and stores them in the
//! vector database. Supports dry-run mode and index reset operations.

use std::{
   path::{Path, PathBuf},
   sync::Arc,
};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use walkdir::WalkDir;

use crate::{
   Result,
   chunker::Chunker,
   embed::Embedder,
   file::LocalFileSystem,
   git,
   index_lock::IndexLock,
   meta::MetaStore,
   store::{LanceStore, Store},
   sync::{SyncEngine, SyncProgressCallback},
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::embed::candle::CandleEmbedder;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use crate::embed::worker::EmbedWorker;

/// Executes the index command to create or update a code index.
pub async fn execute(
   path: Option<PathBuf>,
   dry_run: bool,
   reset: bool,
   store_id: Option<String>,
) -> Result<()> {
   let root = std::env::current_dir()?;
   let index_path = path.unwrap_or_else(|| root.clone());

   let resolved_store_id = store_id.map_or_else(|| git::resolve_store_id(&index_path), Ok)?;

   if reset {
      println!("{}", style(format!("Resetting index for store: {resolved_store_id}")).yellow());
      delete_store(&resolved_store_id, &index_path).await?;
      println!("{}", style("Existing index removed. Re-indexing...").dim());
   }

   let spinner = ProgressBar::new_spinner();
   spinner.set_style(
      ProgressStyle::default_spinner()
         .template("{spinner:.green} {msg}")
         .unwrap(),
   );

   if dry_run {
      spinner.set_message("Scanning files (dry run)...");
      let file_count = scan_files(&index_path);
      spinner.finish_with_message(format!("Dry run complete: would index {file_count} files"));
      println!("\nWould index files in: {}", index_path.display());
      println!("Store ID: {resolved_store_id}");
      return Ok(());
   }

   let mut pb = ProgressBar::new(0);
   pb.set_style(
      ProgressStyle::default_bar()
         .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
         .unwrap()
         .progress_chars("█▓░"),
   );
   pb.set_message("...");
   pb.set_prefix("Indexing: ");

   let result = index_files(&index_path, &resolved_store_id, &mut |u| {
      pb.progress(u);
      spinner.tick();
      pb.tick();
   })
   .await?;

   pb.finish_with_message(format!("Indexing complete: {} files indexed", result.indexed));

   println!("\n{}", style("Index created successfully!").green().bold());
   println!("Store ID: {}", style(&resolved_store_id).cyan());
   println!("Path: {}", style(index_path.display()).dim());
   println!("Files indexed: {}", result.indexed);
   println!("Total chunks: {}", style(result.total_chunks.to_string()).bold());

   Ok(())
}

/// Deletes an existing store and its associated metadata.
async fn delete_store(store_id: &str, index_path: &Path) -> Result<()> {
   let _lock = IndexLock::acquire(store_id)?;

   let store = LanceStore::new()?;

   store.delete_store(store_id).await?;

   let mut meta_store = MetaStore::load(store_id)?;
   meta_store.delete_by_prefix(index_path);
   meta_store.save()?;

   Ok(())
}

/// Scans the directory tree and counts indexable source files.
fn scan_files(path: &Path) -> usize {
   let mut count = 0;
   if path.is_dir() {
      for entry in WalkDir::new(path)
         .follow_links(false)
         .into_iter()
         .filter_map(|e| e.ok())
      {
         if entry.file_type().is_file()
            && let Some(ext) = entry.path().extension()
         {
            let ext_str = ext.to_string_lossy();
            if matches!(ext_str.as_ref(), "rs" | "ts" | "js" | "py" | "go" | "tsx" | "jsx") {
               count += 1;
            }
         }
      }
   }
   count
}

/// Result of an indexing operation.
#[derive(Debug)]
struct IndexResult {
   indexed:      usize,
   total_chunks: usize,
}

/// Performs the actual file indexing using the sync engine.
async fn index_files(
   path: &Path,
   store_id: &str,
   callback: &mut dyn SyncProgressCallback,
) -> Result<IndexResult> {
   let file_system = LocalFileSystem::new();
   // EmbedWorker's parallel workers cause hangs on Metal. Use CandleEmbedder directly.
   // This matches the single-threaded pattern used by huggingface/text-embeddings-inference.
   #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
   let embedder: Arc<dyn Embedder> = Arc::new(CandleEmbedder::new()?);
   #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
   let embedder: Arc<dyn Embedder> = Arc::new(EmbedWorker::new()?);
   let store: Arc<dyn Store> = Arc::new(LanceStore::new()?);

   let sync_engine = SyncEngine::new(file_system, Chunker::default(), embedder, store);

   let result = sync_engine
      .initial_sync(store_id, path, false, callback)
      .await?;

   Ok(IndexResult { indexed: result.indexed, total_chunks: result.indexed })
}
