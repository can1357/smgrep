pub mod lance;

use std::{collections::HashMap, sync::Arc};

use crate::{
   error::Result,
   types::{SearchResponse, StoreInfo, VectorRecord},
};

#[async_trait::async_trait]
pub trait Store: Send + Sync {
   async fn insert_batch(&self, store_id: &str, records: Vec<VectorRecord>) -> Result<()>;

   async fn search(
      &self,
      store_id: &str,
      query_text: &str,
      query_vector: &[f32],
      query_colbert: &[Vec<f32>],
      limit: usize,
      path_filter: Option<&str>,
      rerank: bool,
   ) -> Result<SearchResponse>;

   async fn delete_file(&self, store_id: &str, file_path: &str) -> Result<()>;

   async fn delete_files(&self, store_id: &str, file_paths: &[String]) -> Result<()>;

   async fn delete_store(&self, store_id: &str) -> Result<()>;

   async fn get_info(&self, store_id: &str) -> Result<StoreInfo>;

   async fn list_files(&self, store_id: &str) -> Result<Vec<String>>;

   async fn is_empty(&self, store_id: &str) -> Result<bool>;

   async fn create_fts_index(&self, store_id: &str) -> Result<()>;

   async fn create_vector_index(&self, store_id: &str) -> Result<()>;

   async fn get_file_hashes(&self, store_id: &str) -> Result<HashMap<String, String>>;
}

#[async_trait::async_trait]
impl<T: Store + ?Sized> Store for Arc<T> {
   async fn insert_batch(&self, store_id: &str, records: Vec<VectorRecord>) -> Result<()> {
      (**self).insert_batch(store_id, records).await
   }

   async fn search(
      &self,
      store_id: &str,
      query_text: &str,
      query_vector: &[f32],
      query_colbert: &[Vec<f32>],
      limit: usize,
      path_filter: Option<&str>,
      rerank: bool,
   ) -> Result<SearchResponse> {
      (**self)
         .search(store_id, query_text, query_vector, query_colbert, limit, path_filter, rerank)
         .await
   }

   async fn delete_file(&self, store_id: &str, file_path: &str) -> Result<()> {
      (**self).delete_file(store_id, file_path).await
   }

   async fn delete_files(&self, store_id: &str, file_paths: &[String]) -> Result<()> {
      (**self).delete_files(store_id, file_paths).await
   }

   async fn delete_store(&self, store_id: &str) -> Result<()> {
      (**self).delete_store(store_id).await
   }

   async fn get_info(&self, store_id: &str) -> Result<StoreInfo> {
      (**self).get_info(store_id).await
   }

   async fn list_files(&self, store_id: &str) -> Result<Vec<String>> {
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

   async fn get_file_hashes(&self, store_id: &str) -> Result<HashMap<String, String>> {
      (**self).get_file_hashes(store_id).await
   }
}

pub use lance::LanceStore;
