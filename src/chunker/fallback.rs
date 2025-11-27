use std::path::Path;

use crate::{
   chunker::{Chunker, MAX_CHARS, MAX_LINES, OVERLAP_LINES},
   error::Result,
   types::{Chunk, ChunkType},
};

pub struct FallbackChunker;

impl FallbackChunker {
   pub fn new() -> Self {
      Self
   }
}

impl Default for FallbackChunker {
   fn default() -> Self {
      Self::new()
   }
}

impl Chunker for FallbackChunker {
   fn chunk(&self, content: &str, path: &Path) -> Result<Vec<Chunk>> {
      let lines: Vec<&str> = content.lines().collect();
      let mut chunks = Vec::new();
      let stride = (MAX_LINES - OVERLAP_LINES).max(1);
      let context = vec![format!("File: {}", path.display())];

      let mut i = 0;
      while i < lines.len() {
         let end = (i + MAX_LINES).min(lines.len());
         let sub_lines = &lines[i..end];

         if sub_lines.is_empty() {
            break;
         }

         let sub_content = sub_lines.join("\n");

         if sub_content.len() <= MAX_CHARS {
            chunks.push(Chunk::new(sub_content, i, end, ChunkType::Block, context.clone()));
            i += stride;
         } else {
            let split_chunks = split_by_chars(&sub_content, i, &context);
            chunks.extend(split_chunks);
            i += stride;
         }
      }

      Ok(chunks)
   }
}

fn split_by_chars(content: &str, start_line: usize, context: &[String]) -> Vec<Chunk> {
   let mut chunks = Vec::new();
   let stride = (MAX_CHARS - 200).max(1);

   let mut i = 0;
   while i < content.len() {
      let end = content.ceil_char_boundary(i + MAX_CHARS).min(content.len());
      let sub = &content[i..end];

      if sub.trim().is_empty() {
         break;
      }

      let prefix_lines = content[..i].lines().count();
      let sub_line_count = sub.lines().count();

      chunks.push(Chunk::new(
         sub.to_string(),
         start_line + prefix_lines,
         start_line + prefix_lines + sub_line_count,
         ChunkType::Block,
         context.to_vec(),
      ));

      i = content.ceil_char_boundary(i + stride);
   }

   chunks
}
