use std::{
   collections::HashSet,
   path::{Path, PathBuf},
};

use indicatif::ProgressBar;
use rayon::prelude::*;

pub use crate::types::SyncProgress;
use crate::{
   Result, Str,
   chunker::{Chunker, anchor::create_anchor_chunk},
   config,
   embed::Embedder,
   file::FileSystem,
   meta::{FileHash, MetaStore},
   store::Store,
   types::{PreparedChunk, VectorRecord},
};

fn get_mtime(path: &Path) -> u64 {
   path
      .metadata()
      .and_then(|m| m.modified())
      .ok()
      .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
      .map_or(0, |d| d.as_secs())
}

pub struct SyncEngine<F: FileSystem, E: Embedder, S: Store> {
   file_system: F,
   chunker:     Chunker,
   embedder:    E,
   store:       S,
}

#[derive(Debug, Clone)]
pub struct SyncResult {
   pub processed: usize,
   pub indexed:   usize,
   pub skipped:   usize,
   pub deleted:   usize,
}

pub trait SyncProgressCallback: Send {
   fn progress(&mut self, progress: SyncProgress);
}

impl<F: FnMut(SyncProgress) + Send> SyncProgressCallback for F {
   fn progress(&mut self, progress: SyncProgress) {
      self(progress);
   }
}

impl SyncProgressCallback for () {
   fn progress(&mut self, _progress: SyncProgress) {}
}

impl SyncProgressCallback for ProgressBar {
   fn progress(&mut self, progress: SyncProgress) {
      self.update(|state| {
         state.set_len(progress.total as u64);
         state.set_pos(progress.processed as u64);
      });
      if let Some(file) = &progress.current_file {
         let short = file.rsplit('/').next().unwrap_or(&**file);
         self.set_message(short.to_string());
      }
   }
}

impl<F, E, S> SyncEngine<F, E, S>
where
   F: FileSystem + Sync,
   E: Embedder + Send + Sync,
   S: Store + Send + Sync,
{
   pub const fn new(file_system: F, chunker: Chunker, embedder: E, store: S) -> Self {
      Self { file_system, chunker, embedder, store }
   }

   pub async fn initial_sync(
      &self,
      store_id: &str,
      root: &Path,
      dry_run: bool,
      callback: &mut dyn SyncProgressCallback,
   ) -> Result<SyncResult> {
      const SAVE_INTERVAL: usize = 25;

      let mut meta_store = MetaStore::load(store_id)?;
      let batch_size = config::get().batch_size();

      let files = self.file_system.get_files(root)?.collect::<HashSet<_>>();

      let mut processed = 0;
      let mut indexed = 0;
      let mut skipped = 0;

      let meta_paths = meta_store.all_paths().cloned().collect::<HashSet<_>>();

      let deleted_paths = meta_paths.difference(&files).cloned().collect::<Vec<_>>();

      if !dry_run && !deleted_paths.is_empty() {
         self.store.delete_files(store_id, &deleted_paths).await?;
         for path in &deleted_paths {
            meta_store.remove(path);
         }
      }

      let deleted_count = deleted_paths.len();

      let hash_results: Vec<_> = files
         .into_iter()
         .filter_map(|file_path| {
            let current_mtime = get_mtime(&file_path);

            if let Some(stored_mtime) = meta_store.get_mtime(&file_path)
               && stored_mtime == current_mtime
            {
               return None;
            }

            let content = std::fs::read(&file_path).ok()?;
            let hash = FileHash::sum(&content);

            let existing_hash = meta_store.get_hash(file_path.as_path());
            let needs_indexing = existing_hash != Some(hash);

            Some((file_path, hash, content, current_mtime, needs_indexing))
         })
         .collect();

      let changed_files: Vec<PathBuf> = hash_results
         .iter()
         .filter_map(|(path, _, _, _, needs_indexing)| {
            let has_existing_hash = meta_store.get_hash(path).is_some();
            if *needs_indexing && has_existing_hash {
               Some(path.clone())
            } else {
               None
            }
         })
         .collect();

      if !dry_run && !changed_files.is_empty() {
         self.store.delete_files(store_id, &changed_files).await?;
      }

      let files_to_index: Vec<_> = hash_results
         .into_iter()
         .filter_map(|(path_str, hash, content, mtime, needs_indexing)| {
            processed += 1;
            if !needs_indexing {
               skipped += 1;
               None
            } else if dry_run {
               indexed += 1;
               None
            } else {
               Some((path_str, hash, content, mtime))
            }
         })
         .collect();

      let chunked_files: Vec<_> = files_to_index
         .par_iter()
         .filter_map(|(path, hash, content, mtime)| {
            let content_str = Str::from_utf8_lossy(content);

            let chunks = self.chunker.chunk(&content_str, path).ok()?;
            let anchor_chunk = create_anchor_chunk(&content_str, path);

            let mut prepared_chunks = Vec::new();

            let anchor_prepared = PreparedChunk {
               id:           format!("{}:anchor", path.display()),
               path:         path.clone(),
               hash:         *hash,
               content:      anchor_chunk.content.clone(),
               start_line:   anchor_chunk.start_line as u32,
               end_line:     anchor_chunk.end_line as u32,
               chunk_index:  Some(0),
               is_anchor:    Some(true),
               chunk_type:   anchor_chunk.chunk_type,
               context_prev: None,
               context_next: None,
            };
            prepared_chunks.push(anchor_prepared);

            for (idx, chunk) in chunks.iter().enumerate() {
               let context_prev: Option<Str> = if idx > 0 {
                  Some(chunks[idx - 1].content.clone())
               } else {
                  None
               };

               let context_next: Option<Str> = if idx < chunks.len() - 1 {
                  Some(chunks[idx + 1].content.clone())
               } else {
                  None
               };

               let prepared = PreparedChunk {
                  id: format!("{}:{}", path.display(), idx),
                  path: path.clone(),
                  hash: *hash,
                  content: chunk.content.clone(),
                  start_line: chunk.start_line as u32,
                  end_line: chunk.end_line as u32,
                  chunk_index: Some(idx as u32 + 1),
                  is_anchor: Some(false),
                  chunk_type: chunk.chunk_type,
                  context_prev,
                  context_next,
               };
               prepared_chunks.push(prepared);
            }

            Some((path.clone(), *hash, *mtime, prepared_chunks))
         })
         .collect();

      let mut embed_queue: Vec<(PathBuf, FileHash, u64, Vec<PreparedChunk>)> = Vec::new();
      let mut since_save = 0;
      let total_to_embed = chunked_files.len();
      let mut embedded = 0;

      for (path, hash, mtime, prepared_chunks) in chunked_files {
         embed_queue.push((path, hash, mtime, prepared_chunks));

         if embed_queue.len() >= batch_size {
            callback.progress(SyncProgress {
               processed: embedded,
               indexed,
               total: total_to_embed,
               current_file: Some(
                  format!("Embedding batch ({} files)...", embed_queue.len()).into(),
               ),
            });

            let batch = std::mem::take(&mut embed_queue);
            let batch_count = batch.len();
            let batch_indexed = self
               .process_embed_batch(store_id, batch, &mut meta_store)
               .await?;
            indexed += batch_indexed;
            embedded += batch_count;
            since_save += batch_count;

            if since_save >= SAVE_INTERVAL {
               meta_store.save()?;
               since_save = 0;
            }

            callback.progress(SyncProgress {
               processed: embedded,
               indexed,
               total: total_to_embed,
               current_file: None,
            });
         }
      }

      if !dry_run && !embed_queue.is_empty() {
         callback.progress(SyncProgress {
            processed: embedded,
            indexed,
            total: total_to_embed,
            current_file: Some(
               format!("Embedding final batch ({} files)...", embed_queue.len()).into(),
            ),
         });

         let batch = std::mem::take(&mut embed_queue);
         let batch_count = batch.len();
         let batch_indexed = self
            .process_embed_batch(store_id, batch, &mut meta_store)
            .await?;
         indexed += batch_indexed;
         embedded += batch_count;
      }

      if !dry_run {
         callback.progress(SyncProgress {
            processed: embedded,
            indexed,
            total: total_to_embed,
            current_file: Some("Creating indexes...".into()),
         });

         meta_store.save()?;

         if indexed > 0 {
            self.store.create_fts_index(store_id).await?;
            self.store.create_vector_index(store_id).await?;
         }
      }

      callback.progress(SyncProgress {
         processed: total_to_embed,
         indexed,
         total: total_to_embed,
         current_file: None,
      });

      Ok(SyncResult { processed, indexed, skipped, deleted: deleted_count })
   }

   async fn process_embed_batch(
      &self,
      store_id: &str,
      batch: Vec<(PathBuf, FileHash, u64, Vec<PreparedChunk>)>,
      meta_store: &mut MetaStore,
   ) -> Result<usize> {
      let mut all_chunks = Vec::new();
      let mut file_paths = Vec::new();

      for (path, _hash, _mtime, chunks) in &batch {
         for chunk in chunks {
            all_chunks.push(chunk.clone());
         }
         file_paths.push(path.clone());
      }

      if all_chunks.is_empty() {
         return Ok(0);
      }

      let texts: Vec<Str> = all_chunks.iter().map(|c| c.content.clone()).collect();

      let embeddings = self.embedder.compute_hybrid(&texts).await?;

      let records: Vec<VectorRecord> = all_chunks
         .into_iter()
         .zip(embeddings.into_iter())
         .map(|(chunk, embedding)| VectorRecord {
            id:            chunk.id,
            path:          chunk.path,
            hash:          chunk.hash,
            content:       chunk.content,
            start_line:    chunk.start_line,
            end_line:      chunk.end_line,
            chunk_index:   chunk.chunk_index,
            is_anchor:     chunk.is_anchor,
            chunk_type:    chunk.chunk_type,
            context_prev:  chunk.context_prev,
            context_next:  chunk.context_next,
            vector:        embedding.dense,
            colbert:       embedding.colbert,
            colbert_scale: embedding.colbert_scale,
         })
         .collect();

      self.store.insert_batch(store_id, records).await?;

      for (path, hash, mtime, _) in batch {
         meta_store.set_meta(path, hash, mtime);
      }

      Ok(file_paths.len())
   }
}
