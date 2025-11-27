pub mod lance;

use std::{
   collections::HashMap,
   path::{Path, PathBuf},
   sync::Arc,
};

use crate::{
   error::Result,
   meta::FileHash,
   types::{SearchResponse, StoreInfo, VectorRecord},
};

pub fn path_to_store_key(path: &Path) -> String {
   match path.to_str() {
      Some(s) => s.replace('\'', "''"),
      None => hex::encode(path.as_os_str().as_encoded_bytes()),
   }
}

pub struct SearchParams<'a> {
   pub store_id:      &'a str,
   pub query_text:    &'a str,
   pub query_vector:  &'a [f32],
   pub query_colbert: &'a [Vec<f32>],
   pub limit:         usize,
   pub path_filter:   Option<&'a Path>,
   pub rerank:        bool,
}

#[async_trait::async_trait]
pub trait Store: Send + Sync {
   async fn insert_batch(&self, store_id: &str, records: Vec<VectorRecord>) -> Result<()>;

   async fn search(&self, params: SearchParams<'_>) -> Result<SearchResponse>;

   async fn delete_file(&self, store_id: &str, file_path: &Path) -> Result<()>;

   async fn delete_files(&self, store_id: &str, file_paths: &[PathBuf]) -> Result<()>;

   async fn delete_store(&self, store_id: &str) -> Result<()>;

   async fn get_info(&self, store_id: &str) -> Result<StoreInfo>;

   async fn list_files(&self, store_id: &str) -> Result<Vec<PathBuf>>;

   async fn is_empty(&self, store_id: &str) -> Result<bool>;

   async fn create_fts_index(&self, store_id: &str) -> Result<()>;

   async fn create_vector_index(&self, store_id: &str) -> Result<()>;

   async fn get_file_hashes(&self, store_id: &str) -> Result<HashMap<PathBuf, FileHash>>;
}

#[async_trait::async_trait]
impl<T: Store + ?Sized> Store for Arc<T> {
   async fn insert_batch(&self, store_id: &str, records: Vec<VectorRecord>) -> Result<()> {
      (**self).insert_batch(store_id, records).await
   }

   async fn search(&self, params: SearchParams<'_>) -> Result<SearchResponse> {
      (**self).search(params).await
   }

   async fn delete_file(&self, store_id: &str, file_path: &Path) -> Result<()> {
      (**self).delete_file(store_id, file_path).await
   }

   async fn delete_files(&self, store_id: &str, file_paths: &[PathBuf]) -> Result<()> {
      (**self).delete_files(store_id, file_paths).await
   }

   async fn delete_store(&self, store_id: &str) -> Result<()> {
      (**self).delete_store(store_id).await
   }

   async fn get_info(&self, store_id: &str) -> Result<StoreInfo> {
      (**self).get_info(store_id).await
   }

   async fn list_files(&self, store_id: &str) -> Result<Vec<PathBuf>> {
      (**self).list_files(store_id).await
   }

   async fn is_empty(&self, store_id: &str) -> Result<bool> {
      (**self).is_empty(store_id).await
   }

   async fn create_fts_index(&self, store_id: &str) -> Result<()> {
      (**self).create_fts_index(store_id).await
   }

   async fn create_vector_index(&self, store_id: &str) -> Result<()> {
      (**self).create_vector_index(store_id).await
   }

   async fn get_file_hashes(&self, store_id: &str) -> Result<HashMap<PathBuf, FileHash>> {
      (**self).get_file_hashes(store_id).await
   }
}

pub use lance::LanceStore;
