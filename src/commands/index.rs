use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{
   chunker::fallback::FallbackChunker,
   embed::Embedder,
   file::LocalFileSystem,
   meta::MetaStore,
   store::{LanceStore, Store},
   sync::SyncEngine,
};

pub async fn execute(
   path: Option<PathBuf>,
   dry_run: bool,
   reset: bool,
   store_id: Option<String>,
) -> Result<()> {
   let root = std::env::current_dir().context("failed to get current directory")?;
   let index_path = path.unwrap_or_else(|| root.clone());

   let resolved_store_id = store_id.unwrap_or_else(|| {
      index_path
         .file_name()
         .and_then(|s| s.to_str())
         .unwrap_or("default")
         .to_string()
   });

   if reset {
      println!("{}", style(format!("Resetting index for store: {}", resolved_store_id)).yellow());
      delete_store(&resolved_store_id, &index_path).await?;
      println!("{}", style("Existing index removed. Re-indexing...").dim());
   }

   let spinner = ProgressBar::new_spinner();
   spinner.set_style(
      ProgressStyle::default_spinner()
         .template("{spinner:.green} {msg}")
         .unwrap(),
   );
   spinner.enable_steady_tick(Duration::from_millis(100));

   if dry_run {
      spinner.set_message("Scanning files (dry run)...");
      let file_count = scan_files(&index_path).await?;
      spinner.finish_with_message(format!("Dry run complete: would index {} files", file_count));
      println!("\nWould index files in: {}", index_path.display());
      println!("Store ID: {}", resolved_store_id);
      return Ok(());
   }

   let pb = ProgressBar::new(0);
   pb.set_style(
      ProgressStyle::default_bar()
         .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
         .unwrap()
         .progress_chars("█▓░"),
   );
   pb.enable_steady_tick(Duration::from_millis(100));
   pb.set_message("Indexing files...");

   let pb_clone = pb.clone();
   let progress_callback: Box<dyn Fn(crate::types::SyncProgress) + Send> =
      Box::new(move |progress: crate::types::SyncProgress| {
         pb_clone.set_length(progress.total as u64);
         pb_clone.set_position(progress.processed as u64);
         if let Some(file) = &progress.current_file {
            let short = file
               .rsplit('/')
               .next()
               .unwrap_or(file);
            pb_clone.set_message(format!("Indexing: {}", short));
         }
      });

   let result = index_files(&index_path, &resolved_store_id, Some(progress_callback)).await?;

   pb.finish_with_message(format!("Indexing complete: {} files indexed", result.indexed));

   println!("\n{}", style("Index created successfully!").green().bold());
   println!("Store ID: {}", style(&resolved_store_id).cyan());
   println!("Path: {}", style(index_path.display()).dim());
   println!("Files indexed: {}", result.indexed);
   println!("Total chunks: {}", style(result.total_chunks.to_string()).bold());

   Ok(())
}

async fn delete_store(store_id: &str, index_path: &PathBuf) -> Result<()> {
   let store = LanceStore::new()?;

   store.delete_store(store_id).await?;

   let mut meta_store = MetaStore::load(store_id)?;
   let prefix = index_path.to_string_lossy().to_string();
   meta_store.delete_by_prefix(&prefix);
   meta_store.save()?;

   Ok(())
}

async fn scan_files(path: &PathBuf) -> Result<usize> {
   let mut count = 0;
   if path.is_dir() {
      for entry in walkdir::WalkDir::new(path)
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
   Ok(count)
}

#[derive(Debug)]
struct IndexResult {
   indexed:      usize,
   total_chunks: usize,
}

async fn index_files(
   path: &PathBuf,
   store_id: &str,
   progress_callback: Option<Box<dyn Fn(crate::types::SyncProgress) + Send>>,
) -> Result<IndexResult> {
   let file_system = LocalFileSystem::new();
   let chunker = FallbackChunker::new();
   let embedder: Arc<dyn Embedder> = Arc::new(crate::embed::candle::CandleEmbedder::new()?);
   let store: Arc<dyn Store> = Arc::new(LanceStore::new()?);

   let sync_engine = SyncEngine::new(file_system, chunker, embedder, store);

   let result = sync_engine
      .initial_sync(store_id, path, false, progress_callback)
      .await?;

   Ok(IndexResult { indexed: result.indexed, total_chunks: 0 })
}
