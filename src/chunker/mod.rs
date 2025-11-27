pub mod anchor;
pub mod fallback;
pub mod treesitter;

use std::path::Path;

use crate::{error::Result, types::Chunk};

pub trait Chunker: Send + Sync {
   fn chunk(&self, content: &str, path: &Path) -> Result<Vec<Chunk>>;
}

pub fn create_chunker(path: &Path) -> Box<dyn Chunker> {
   let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

   match ext {
      "ts" | "tsx" | "js" | "jsx" | "py" | "go" => Box::new(treesitter::TreeSitterChunker::new()),
      _ => Box::new(fallback::FallbackChunker::new()),
   }
}

pub const MAX_LINES: usize = 75;
pub const MAX_CHARS: usize = 2000;
pub const OVERLAP_LINES: usize = 10;
pub const OVERLAP_CHARS: usize = 200;
