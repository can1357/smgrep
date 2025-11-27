pub mod colbert;
pub mod ranking;

use std::{cmp::Ordering, path::Path, sync::Arc};

use crate::{
   embed::Embedder,
   error::Result,
   store::{SearchParams, Store},
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
      path_filter: Option<&Path>,
      rerank: bool,
   ) -> Result<SearchResponse> {
      let query_enc = self.embedder.encode_query(query).await?;
      let mut response = self
         .store
         .search(SearchParams {
            store_id,
            query_text: query,
            query_vector: &query_enc.dense,
            query_colbert: &query_enc.colbert,
            limit: limit * 2,
            path_filter,
            rerank,
         })
         .await?;

      ranking::apply_structural_boost(&mut response.results);

      response
         .results
         .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

      if per_file_limit > 0 {
         response.results = ranking::apply_per_file_limit(response.results, per_file_limit);
      }

      response.results.truncate(limit);

      Ok(response)
   }
}
