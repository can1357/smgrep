pub mod candle;
pub mod worker;

use std::sync::Arc;

pub use candle::CandleEmbedder;
pub use worker::EmbedWorker;

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct HybridEmbedding {
   pub dense:         Vec<f32>,
   pub colbert:       Vec<u8>,
   pub colbert_scale: f64,
}

#[derive(Debug, Clone)]
pub struct QueryEmbedding {
   pub dense:   Vec<f32>,
   pub colbert: Vec<Vec<f32>>,
}

#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
   async fn compute_hybrid(&self, texts: &[String]) -> Result<Vec<HybridEmbedding>>;
   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding>;
   fn is_ready(&self) -> bool;
}

#[async_trait::async_trait]
impl<T: Embedder + ?Sized> Embedder for Arc<T> {
   async fn compute_hybrid(&self, texts: &[String]) -> Result<Vec<HybridEmbedding>> {
      (**self).compute_hybrid(texts).await
   }

   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding> {
      (**self).encode_query(text).await
   }

   fn is_ready(&self) -> bool {
      (**self).is_ready()
   }
}
