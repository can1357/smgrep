pub mod colbert;
pub mod ranking;

use std::sync::Arc;

use crate::{
   embed::Embedder,
   error::{Result, RsgrepError},
   store::Store,
   types::SearchResponse,
};

pub struct SearchEngine {
   store:    Arc<dyn Store>,
   embedder: Arc<dyn Embedder>,
}

impl SearchEngine {
   pub fn new(store: Arc<dyn Store>, embedder: Arc<dyn Embedder>) -> Self {
      Self { store, embedder }
   }

   pub async fn search(
      &self,
      store_id: &str,
      query: &str,
      limit: usize,
      per_file_limit: usize,
      path_filter: Option<&str>,
      rerank: bool,
   ) -> Result<SearchResponse> {
      let query_enc = self
         .embedder
         .encode_query(query)
         .await
         .map_err(|e| RsgrepError::Embedding(format!("failed to encode query: {}", e)))?;

      let mut response = self
         .store
         .search(
            store_id,
            query,
            &query_enc.dense,
            &query_enc.colbert,
            limit * 2,
            path_filter,
            rerank,
         )
         .await?;

      ranking::apply_structural_boost(&mut response.results);

      response.results.sort_by(|a, b| {
         b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
      });

      if per_file_limit > 0 {
         response.results = ranking::apply_per_file_limit(response.results, per_file_limit);
      }

      response.results.truncate(limit);

      Ok(response)
   }
}
