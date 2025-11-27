use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkType {
   Function,
   Class,
   Interface,
   Method,
   TypeAlias,
   Block,
   Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
   pub content:     String,
   pub start_line:  usize,
   pub end_line:    usize,
   pub chunk_type:  Option<ChunkType>,
   pub context:     Vec<String>,
   pub chunk_index: Option<i32>,
   pub is_anchor:   Option<bool>,
}

impl Chunk {
   pub fn new(
      content: String,
      start_line: usize,
      end_line: usize,
      chunk_type: ChunkType,
      context: Vec<String>,
   ) -> Self {
      Self {
         content,
         start_line,
         end_line,
         chunk_type: Some(chunk_type),
         context,
         chunk_index: None,
         is_anchor: Some(false),
      }
   }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedChunk {
   pub id:           String,
   pub path:         String,
   pub hash:         String,
   pub content:      String,
   pub start_line:   u32,
   pub end_line:     u32,
   pub chunk_index:  Option<u32>,
   pub is_anchor:    Option<bool>,
   pub chunk_type:   Option<ChunkType>,
   pub context_prev: Option<String>,
   pub context_next: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VectorRecord {
   pub id:            String,
   pub path:          String,
   pub hash:          String,
   pub content:       String,
   pub start_line:    u32,
   pub end_line:      u32,
   pub chunk_index:   Option<u32>,
   pub is_anchor:     Option<bool>,
   pub chunk_type:    Option<ChunkType>,
   pub context_prev:  Option<String>,
   pub context_next:  Option<String>,
   pub vector:        Vec<f32>,
   pub colbert:       Vec<u8>,
   pub colbert_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
   pub path:       String,
   pub content:    String,
   pub score:      f32,
   pub start_line: u32,
   pub num_lines:  u32,
   pub chunk_type: Option<ChunkType>,
   pub is_anchor:  Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchStatus {
   Ready,
   Indexing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
   pub results:  Vec<SearchResult>,
   pub status:   SearchStatus,
   pub progress: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreInfo {
   pub store_id:  String,
   pub row_count: u64,
   pub path:      PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncProgress {
   pub processed:    usize,
   pub indexed:      usize,
   pub total:        usize,
   pub current_file: Option<String>,
}
