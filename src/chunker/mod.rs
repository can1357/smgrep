//! Code chunking with tree-sitter language-aware parsing.
//!
//! Splits source files into semantic chunks for code search and analysis.
//! Uses tree-sitter grammars when available, falls back to line-based
//! splitting.

pub mod anchor;

use std::{borrow::Cow, path::Path, slice, sync::Arc};

use memchr::memchr_iter;
use tree_sitter::Language;

use crate::{
   Str,
   chunker::anchor::CONST_EXPORT_REGEX,
   error::{ChunkerError, Result},
   grammar::GrammarManager,
   types::{Chunk, ChunkType},
};

/// Maximum number of lines per chunk.
pub const MAX_LINES: usize = 75;

/// Maximum number of characters per chunk.
pub const MAX_CHARS: usize = 2000;

/// Number of lines to overlap between consecutive chunks.
pub const OVERLAP_LINES: usize = 10;

/// Number of characters to overlap between consecutive chunks.
pub const OVERLAP_CHARS: usize = 200;

/// Number of characters to advance between chunks (`MAX_CHARS` -
/// `OVERLAP_CHARS`).
pub const STRIDE_CHARS: usize = MAX_CHARS - OVERLAP_CHARS;

/// Number of lines to advance between chunks (`MAX_LINES` - `OVERLAP_LINES`).
pub const STRIDE_LINES: usize = MAX_LINES - OVERLAP_LINES;

/// Splits source code into semantic chunks using tree-sitter grammars.
///
/// Extracts definitions (functions, classes, types) from parsed syntax trees
/// and splits them into manageable chunks for code search.
#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Chunker(Arc<GrammarManager>);

impl Chunker {
   async fn get_language(&self, path: &Path) -> Result<Option<Language>> {
      self.0.get_language_for_path(path).await.inspect_err(|e| {
         tracing::warn!("failed to load language for {}: {}", path.display(), e);
      })
   }

   fn line_range_to_byte_range(
      content: &str,
      start_line: usize,
      end_line: usize,
   ) -> (usize, usize) {
      let bytes = content.as_bytes();
      let mut start_byte = 0;
      let mut end_byte = bytes.len();
      let mut current_line = 0;

      for idx in memchr_iter(b'\n', bytes) {
         current_line += 1;
         if current_line == end_line {
            end_byte = idx;
            break;
         }
         if current_line == start_line {
            start_byte = idx + 1;
         }
      }

      (start_byte, end_byte)
   }

   fn simple_chunk(content: &Str, path: &Path) -> Vec<Chunk> {
      let lines: Vec<&str> = content.lines().collect();
      let mut chunks = Vec::new();
      let stride = (MAX_LINES - OVERLAP_LINES).max(1);
      let context: Str = format!("File: {}", path.display()).into();
      let stack = slice::from_ref(&context);

      let mut i = 0;
      while i < lines.len() {
         let end = (i + MAX_LINES).min(lines.len());
         let sub_lines = &lines[i..end];

         if sub_lines.is_empty() {
            break;
         }

         let (start_byte, end_byte) = Self::line_range_to_byte_range(content, i, end);
         let sub_content = content.slice(start_byte..end_byte);

         if sub_content.len() <= MAX_CHARS {
            chunks.push(Chunk::new(sub_content, i, end, ChunkType::Block, stack));
         } else {
            let split_chunks = Self::split_content_by_chars(&sub_content, i, stack);
            chunks.extend(split_chunks);
         }
         i += stride;
      }

      chunks
   }

   async fn chunk_with_tree_sitter(
      &self,
      content: &Str,
      path: &Path,
   ) -> Result<Option<Vec<Chunk>>> {
      let Some(language) = self.get_language(path).await? else {
         return Ok(None);
      };

      let (mut parser, store) = self.0.create_parser_with_store()?;
      parser
         .set_wasm_store(store)
         .map_err(ChunkerError::SetWasmStore)?;

      parser
         .set_language(&language)
         .map_err(ChunkerError::SetLanguage)?;

      let tree = parser
         .parse(content.as_str(), None)
         .ok_or(ChunkerError::ParseFailed)?;

      let root = tree.root_node();
      let file_context: Str = format!("File: {}", path.display()).into();

      let mut chunks = Vec::new();
      let mut block_chunks = Vec::new();
      let mut cursor_index = 0;
      let mut cursor_row = 0;
      let mut saw_definition = false;

      let mut cursor = root.walk();
      for child in root.named_children(&mut cursor) {
         Self::visit_node(
            &child,
            content,
            slice::from_ref(&file_context),
            &mut chunks,
            &mut saw_definition,
         );

         let effective = Self::unwrap_export(&child);
         let is_definition = Self::is_definition_node(&effective, content.as_str());

         if is_definition {
            if child.start_byte() > cursor_index {
               let gap_text = content.slice(cursor_index..child.start_byte());
               if !gap_text.trim().is_empty() {
                  block_chunks.push(Chunk::new(
                     gap_text,
                     cursor_row,
                     child.start_position().row,
                     ChunkType::Block,
                     slice::from_ref(&file_context),
                  ));
               }
            }

            cursor_index = child.end_byte();
            cursor_row = child.end_position().row;
         }
      }

      if cursor_index < content.len() {
         let tail_text = content.slice(cursor_index..);
         if !tail_text.trim().is_empty() {
            block_chunks.push(Chunk::new(
               tail_text,
               cursor_row,
               root.end_position().row,
               ChunkType::Block,
               &[file_context],
            ));
         }
      }

      if !saw_definition {
         return Ok(Some(Vec::new()));
      }

      let mut combined = block_chunks;
      combined.extend(chunks);
      combined.sort_by(|a, b| {
         a.start_line
            .cmp(&b.start_line)
            .then(a.end_line.cmp(&b.end_line))
      });

      Ok(Some(combined))
   }

   fn visit_node(
      node: &tree_sitter::Node,
      content: &Str,
      stack: &[Str],
      chunks: &mut Vec<Chunk>,
      saw_definition: &mut bool,
   ) {
      let effective = Self::unwrap_export(node);
      let is_definition = Self::is_definition_node(&effective, content.as_str());
      let mut stack = Cow::Borrowed(stack);

      if is_definition {
         *saw_definition = true;
         let label = Self::label_for_node(&effective, content.as_str());
         if let Some(label) = label {
            stack.to_mut().push(label.into());
         }

         let node_text = content.slice(effective.start_byte()..effective.end_byte());
         chunks.push(Chunk::new(
            node_text,
            effective.start_position().row,
            effective.end_position().row,
            Self::classify_node(&effective),
            stack.as_ref(),
         ));
      }

      let mut cursor = effective.walk();
      for child in effective.named_children(&mut cursor) {
         Self::visit_node(&child, content, &stack, chunks, saw_definition);
      }
   }

   fn unwrap_export<'a>(node: &'a tree_sitter::Node) -> tree_sitter::Node<'a> {
      if node.kind() == "export_statement" && node.named_child_count() > 0 {
         return node.named_child(0).unwrap();
      }
      *node
   }

   fn is_definition_node(node: &tree_sitter::Node, content: &str) -> bool {
      let kind = node.kind();
      matches!(
         kind,
         // JavaScript/TypeScript/Go/C/C++/Java
         "function_declaration"
            | "function_definition"
            | "method_definition"
            | "method_declaration"
            | "class_declaration"
            | "class_definition"
            | "interface_declaration"
            | "type_alias_declaration"
            | "type_declaration"
            // Rust
            | "function_item"
            | "impl_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "mod_item"
            | "type_item"
            | "const_item"
            | "static_item"
            // Python
            | "function_def"
            | "class_def"
            | "async_function_def"
      ) || Self::is_top_level_value_def(node, content)
   }

   fn is_top_level_value_def(node: &tree_sitter::Node, content: &str) -> bool {
      let kind = node.kind();
      if kind != "lexical_declaration" && kind != "variable_declaration" {
         return false;
      }

      if let Some(parent) = node.parent() {
         let parent_kind = parent.kind();
         if !matches!(parent_kind, "program" | "module" | "source_file" | "class_body") {
            return false;
         }
      }

      let text = &content[node.start_byte()..node.end_byte()];

      if text.contains("=>") {
         return true;
      }
      if text.contains("function ") {
         return true;
      }
      if text.contains("class ") {
         return true;
      }

      if CONST_EXPORT_REGEX.is_match(text) {
         return true;
      }

      false
   }

   fn classify_node(node: &tree_sitter::Node) -> ChunkType {
      let kind = node.kind();
      if kind.contains("class") {
         ChunkType::Class
      } else if kind.contains("interface") {
         ChunkType::Interface
      } else if kind.contains("type_alias") || kind.contains("type_declaration") {
         ChunkType::TypeAlias
      } else {
         ChunkType::Other
      }
   }

   fn get_node_name<'a>(node: &tree_sitter::Node, content: &'a str) -> Option<&'a str> {
      if let Some(name_node) = node.child_by_field_name("name") {
         return Some(&content[name_node.start_byte()..name_node.end_byte()]);
      }

      if let Some(property_node) = node.child_by_field_name("property") {
         return Some(&content[property_node.start_byte()..property_node.end_byte()]);
      }

      if let Some(identifier_node) = node.child_by_field_name("identifier") {
         return Some(&content[identifier_node.start_byte()..identifier_node.end_byte()]);
      }

      let mut cursor = node.walk();
      for child in node.named_children(&mut cursor) {
         let child_kind = child.kind();
         if matches!(
            child_kind,
            "identifier" | "property_identifier" | "type_identifier" | "field_identifier"
         ) {
            return Some(&content[child.start_byte()..child.end_byte()]);
         }

         if child_kind == "variable_declarator"
            && let Some(name) = Self::get_node_name(&child, content)
         {
            return Some(name);
         }
      }

      None
   }

   fn label_for_node(node: &tree_sitter::Node, content: &str) -> Option<String> {
      let name = Self::get_node_name(node, content);
      let kind = node.kind();

      let (prefix, default) = if kind.contains("class") {
         ("Class: ", "<anonymous class>")
      } else if kind.contains("method") {
         ("Method: ", "<anonymous method>")
      } else if kind.contains("interface") {
         ("Interface: ", "<anonymous interface>")
      } else if kind.contains("type_alias") || kind.contains("type_declaration") {
         ("Type: ", "<anonymous type>")
      } else if kind.contains("function") || Self::is_top_level_value_def(node, content) {
         ("Function: ", "<anonymous function>")
      } else {
         return name.map(|n| format!("Symbol: {n}"));
      };
      Some(format!("{prefix}{}", name.unwrap_or(default)))
   }

   fn split_if_too_big(chunk: Chunk) -> Vec<Chunk> {
      let char_count = chunk.content.len();
      let lines: Vec<&str> = chunk.content.lines().collect();
      let line_count = lines.len();

      if line_count <= MAX_LINES && char_count <= MAX_CHARS {
         return vec![chunk];
      }

      if char_count > MAX_CHARS && line_count <= MAX_LINES {
         return Self::split_by_chars(chunk);
      }

      let mut sub_chunks = Vec::new();
      let stride = (MAX_LINES - OVERLAP_LINES).max(1);
      let header = Self::extract_header_line(&chunk.content);

      let mut i = 0;
      while i < lines.len() {
         let end = (i + MAX_LINES).min(lines.len());
         let sub_lines = &lines[i..end];

         if sub_lines.len() < 3 && i > 0 {
            i += stride;
            continue;
         }

         let (start_byte, end_byte) = Self::line_range_to_byte_range(&chunk.content, i, end);
         let mut content = chunk.content.slice(start_byte..end_byte);

         if let Some(h) = header
            && i > 0
            && chunk.chunk_type != Some(ChunkType::Block)
         {
            content = Str::from_string(format!("{h}\n{content}"));
         }

         sub_chunks.push(Chunk::new(
            content,
            chunk.start_line + i,
            chunk.start_line + end,
            chunk.chunk_type.unwrap_or(ChunkType::Other),
            &chunk.context,
         ));

         i += stride;
      }

      sub_chunks
         .into_iter()
         .flat_map(|sc| {
            if sc.content.len() > MAX_CHARS {
               Self::split_by_chars(sc)
            } else {
               vec![sc]
            }
         })
         .collect()
   }

   fn split_by_chars_impl(
      content: &Str,
      start_line: usize,
      chunk_type: ChunkType,
      context: &[Str],
   ) -> Vec<Chunk> {
      let mut chunks = Vec::new();
      let mut iter = content.as_str();
      let mut ln = start_line;
      loop {
         iter = iter.trim_start();
         if iter.is_empty() {
            break;
         }
         let lim = iter.floor_char_boundary(MAX_CHARS);
         let (pre, post) = iter.split_at(lim);
         iter = post;
         let trimmed = pre.trim_end();
         if trimmed.is_empty() {
            continue;
         }
         let lines = trimmed.lines().count();
         chunks.push(Chunk::new(content.slice_ref(trimmed), ln, ln + lines, chunk_type, context));
         ln += lines;
      }
      chunks
   }

   fn split_content_by_chars(input: &Str, start_line: usize, context: &[Str]) -> Vec<Chunk> {
      Self::split_by_chars_impl(input, start_line, ChunkType::Block, context)
   }

   fn split_by_chars(chunk: Chunk) -> Vec<Chunk> {
      Self::split_by_chars_impl(
         &chunk.content,
         chunk.start_line,
         chunk.chunk_type.unwrap_or(ChunkType::Other),
         &chunk.context,
      )
   }

   fn extract_header_line(text: &str) -> Option<&str> {
      text.lines().map(str::trim).find(|s| !s.is_empty())
   }

   /// Splits source code into semantic chunks.
   ///
   /// Attempts tree-sitter parsing first, falls back to line-based chunking if
   /// parsing fails. Ensures all chunks satisfy [`MAX_LINES`] and
   /// [`MAX_CHARS`] constraints.
   pub async fn chunk(&self, content: &Str, path: &Path) -> Result<Vec<Chunk>> {
      let raw_chunks = match self.chunk_with_tree_sitter(content, path).await {
         Ok(Some(c)) => c,
         Ok(None) => Self::simple_chunk(content, path),
         Err(e) => {
            tracing::warn!(
               error = %e,
               path = %path.display(),
               "tree-sitter chunk failed, falling back to simple chunk"
            );
            Self::simple_chunk(content, path)
         },
      };

      let chunks: Vec<Chunk> = raw_chunks
         .into_iter()
         .flat_map(Self::split_if_too_big)
         .collect();

      Ok(chunks)
   }
}

#[cfg(test)]
mod tests {
   use std::fmt::Write;

   use super::*;

   #[test]
   fn split_by_chars_preserves_chunk_type() {
      let content = Str::from_string("a".repeat(MAX_CHARS + 10));
      let chunk = Chunk::new(content, 0, 1, ChunkType::Function, &[]);

      let pieces = Chunker::split_by_chars(chunk);

      assert!(!pieces.is_empty());
      assert!(
         pieces
            .iter()
            .all(|c| c.chunk_type == Some(ChunkType::Function))
      );
   }

   #[test]
   fn split_if_too_big_keeps_trailing_lines() {
      let content = Str::from_string((0..131).fold(String::new(), |mut s, i| {
         let _ = writeln!(s, "line {i}");
         s
      }));
      let chunk = Chunk::new(content, 0, 131, ChunkType::Block, &[]);

      let sub_chunks = Chunker::split_if_too_big(chunk);

      assert!(
         sub_chunks
            .iter()
            .any(|c| c.content.as_str().contains("line 130"))
      );
   }
}
