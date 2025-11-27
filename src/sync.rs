use std::{collections::HashSet, path::Path};

use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::{
   Result,
   chunker::{Chunker, anchor::create_anchor_chunk},
   config,
   embed::Embedder,
   file::FileSystem,
   meta::MetaStore,
   store::Store,
   types::{PreparedChunk, SyncProgress, VectorRecord},
};

fn get_mtime(path: &Path) -> u64 {
   path.metadata()
      .and_then(|m| m.modified())
      .ok()
      .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
      .map(|d| d.as_secs())
      .unwrap_or(0)
}

pub struct SyncEngine<F: FileSystem, C: Chunker, E: Embedder, S: Store> {
   file_system: F,
   chunker:     C,
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

impl<F, C, E, S> SyncEngine<F, C, E, S>
where
   F: FileSystem + Sync,
   C: Chunker,
   E: Embedder + Send + Sync,
   S: Store + Send + Sync,
{
   pub fn new(file_system: F, chunker: C, embedder: E, store: S) -> Self {
      Self { file_system, chunker, embedder, store }
   }

   pub async fn initial_sync(
      &self,
      store_id: &str,
      root: &Path,
      dry_run: bool,
      progress_callback: Option<Box<dyn Fn(SyncProgress) + Send>>,
   ) -> Result<SyncResult> {
      const SAVE_INTERVAL: usize = 25;

      let mut meta_store = MetaStore::load(store_id)?;
      let batch_size = config::batch_size();

      let files: Vec<_> = self.file_system.get_files(root)?.collect::<Vec<_>>();

      let mut processed = 0;
      let mut indexed = 0;
      let mut skipped = 0;

      let files_on_disk: HashSet<String> = files
         .iter()
         .map(|p| p.to_string_lossy().to_string())
         .collect();

      let meta_paths: HashSet<String> = meta_store.all_paths().cloned().collect();

      let deleted_paths: Vec<String> = meta_paths.difference(&files_on_disk).cloned().collect();

      if !dry_run && !deleted_paths.is_empty() {
         self.store.delete_files(store_id, &deleted_paths).await?;
         for path in &deleted_paths {
            meta_store.remove(path);
         }
      }

      let deleted_count = deleted_paths.len();

      let store_hashes = self.store.get_file_hashes(store_id).await?;

      let hash_results: Vec<_> = files
         .par_iter()
         .filter_map(|file_path| {
            let path_str = file_path.to_string_lossy().to_string();
            let current_mtime = get_mtime(file_path);

            if let Some(stored_mtime) = meta_store.get_mtime(&path_str) {
               if stored_mtime == current_mtime {
                  return None;
               }
            }

            let content = std::fs::read(file_path).ok()?;
            let hash = compute_hash(&content);

            let existing_hash = meta_store
               .get_hash(&path_str)
               .map(|s| s.as_str())
               .or_else(|| store_hashes.get(&path_str).map(|s| s.as_str()));
            let needs_indexing = existing_hash != Some(hash.as_str());

            Some((path_str, hash, content, current_mtime, needs_indexing))
         })
         .collect();

      let changed_files: Vec<String> = hash_results
         .iter()
         .filter_map(|(path, _, _, _, needs_indexing)| {
            let has_existing_hash =
               meta_store.get_hash(path).is_some() || store_hashes.contains_key(path);
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
         .filter_map(|(path_str, hash, content, mtime)| {
            let file_path = Path::new(path_str);
            let content_str = String::from_utf8_lossy(content);

            let chunks = self.chunker.chunk(&content_str, file_path).ok()?;
            let anchor_chunk = create_anchor_chunk(&content_str, file_path);

            let mut prepared_chunks = Vec::new();

            let anchor_prepared = PreparedChunk {
               id:           format!("{}:anchor", path_str),
               path:         path_str.clone(),
               hash:         hash.clone(),
               content:      anchor_chunk.content,
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
               let context_prev = if idx > 0 {
                  Some(chunks[idx - 1].content.clone())
               } else {
                  None
               };

               let context_next = if idx < chunks.len() - 1 {
                  Some(chunks[idx + 1].content.clone())
               } else {
                  None
               };

               let prepared = PreparedChunk {
                  id:           format!("{}:{}", path_str, idx),
                  path:         path_str.clone(),
                  hash:         hash.clone(),
                  content:      chunk.content.clone(),
                  start_line:   chunk.start_line as u32,
                  end_line:     chunk.end_line as u32,
                  chunk_index:  Some(idx as u32 + 1),
                  is_anchor:    Some(false),
                  chunk_type:   chunk.chunk_type,
                  context_prev,
                  context_next,
               };
               prepared_chunks.push(prepared);
            }

            Some((path_str.clone(), hash.clone(), *mtime, prepared_chunks))
         })
         .collect();

      let mut embed_queue: Vec<(String, String, u64, Vec<PreparedChunk>)> = Vec::new();
      let mut since_save = 0;
      let total_to_embed = chunked_files.len();
      let mut embedded = 0;

      for (path_str, hash, mtime, prepared_chunks) in chunked_files {
         embed_queue.push((path_str.clone(), hash, mtime, prepared_chunks));

         if embed_queue.len() >= batch_size {
            if let Some(callback) = &progress_callback {
               callback(SyncProgress {
                  processed: embedded,
                  indexed,
                  total: total_to_embed,
                  current_file: Some(format!("Embedding batch ({} files)...", embed_queue.len())),
               });
            }

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

            if let Some(callback) = &progress_callback {
               callback(SyncProgress {
                  processed: embedded,
                  indexed,
                  total: total_to_embed,
                  current_file: None,
               });
            }
         }
      }

      if !dry_run && !embed_queue.is_empty() {
         if let Some(callback) = &progress_callback {
            callback(SyncProgress {
               processed: embedded,
               indexed,
               total: total_to_embed,
               current_file: Some(format!("Embedding final batch ({} files)...", embed_queue.len())),
            });
         }

         let batch = std::mem::take(&mut embed_queue);
         let batch_count = batch.len();
         let batch_indexed = self
            .process_embed_batch(store_id, batch, &mut meta_store)
            .await?;
         indexed += batch_indexed;
         embedded += batch_count;
      }

      if !dry_run {
         if let Some(callback) = &progress_callback {
            callback(SyncProgress {
               processed: embedded,
               indexed,
               total: total_to_embed,
               current_file: Some("Creating indexes...".to_string()),
            });
         }

         meta_store.save()?;

         if indexed > 0 {
            self.store.create_fts_index(store_id).await?;
            self.store.create_vector_index(store_id).await?;
         }
      }

      if let Some(callback) = &progress_callback {
         callback(SyncProgress {
            processed: total_to_embed,
            indexed,
            total: total_to_embed,
            current_file: None,
         });
      }

      Ok(SyncResult { processed, indexed, skipped, deleted: deleted_count })
   }

   async fn process_embed_batch(
      &self,
      store_id: &str,
      batch: Vec<(String, String, u64, Vec<PreparedChunk>)>,
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

      let texts: Vec<String> = all_chunks.iter().map(|c| c.content.clone()).collect();

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

fn compute_hash(content: &[u8]) -> String {
   let mut hasher = Sha256::new();
   hasher.update(content);
   hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn compute_hash_consistent() {
      let data = b"hello world";
      let hash1 = compute_hash(data);
      let hash2 = compute_hash(data);
      assert_eq!(hash1, hash2);
   }

   #[test]
   fn compute_hash_different_data() {
      let hash1 = compute_hash(b"hello");
      let hash2 = compute_hash(b"world");
      assert_ne!(hash1, hash2);
   }
}
