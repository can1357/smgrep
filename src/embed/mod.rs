//! Text embedding module for semantic code search
//!
//! Provides hybrid embedding functionality combining dense and `ColBERT` sparse
//! vectors for improved retrieval accuracy.

pub mod candle;
pub mod worker;

use std::sync::Arc;

pub use candle::CandleEmbedder;
use ndarray::Array2;
pub use worker::EmbedWorker;

use crate::{Str, error::Result};

/// Hybrid embedding representation combining dense and sparse vectors
///
/// Dense embeddings capture semantic meaning while `ColBERT` embeddings
/// preserve token-level information for fine-grained matching.
#[derive(Debug, Clone)]
pub struct HybridEmbedding {
   /// Dense semantic embedding vector
   pub dense:         Vec<f32>,
   /// Quantized `ColBERT` token embeddings
   pub colbert:       Vec<u8>,
   /// Scale factor for dequantizing `ColBERT` embeddings
   pub colbert_scale: f64,
}

/// Query embedding with both dense and `ColBERT` representations
#[derive(Debug, Clone)]
pub struct QueryEmbedding {
   /// Dense semantic embedding vector
   pub dense:   Vec<f32>,
   /// `ColBERT` token embeddings matrix (rows = tokens, cols = dim)
   pub colbert: Array2<f32>,
}

/// Text embedding trait for generating hybrid embeddings
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
   /// Computes hybrid embeddings for multiple texts
   async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>>;
   /// Encodes a query with optional prefix
   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding>;
   /// Returns whether the embedder models are loaded and ready
   fn is_ready(&self) -> bool;
}

#[async_trait::async_trait]
impl<T: Embedder + ?Sized> Embedder for Arc<T> {
   async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>> {
      (**self).compute_hybrid(texts).await
   }

   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding> {
      (**self).encode_query(text).await
   }

   fn is_ready(&self) -> bool {
      (**self).is_ready()
   }
}
