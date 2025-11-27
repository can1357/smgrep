use std::{path::Path, sync::Arc};

use regex::Regex;
use tree_sitter::Language;

use crate::{
   chunker::{
      Chunker, MAX_CHARS, MAX_LINES, OVERLAP_CHARS, OVERLAP_LINES, fallback::FallbackChunker,
   },
   error::{Result, RsgrepError},
   grammar::GrammarManager,
   types::{Chunk, ChunkType},
};

pub struct TreeSitterChunker {
   grammar_manager:         Arc<GrammarManager>,
   screaming_const_pattern: Regex,
}

impl TreeSitterChunker {
   pub fn new() -> Self {
      Self::with_grammar_manager(Arc::new(GrammarManager::default()))
   }

   pub fn with_grammar_manager(grammar_manager: Arc<GrammarManager>) -> Self {
      Self {
         grammar_manager,
         screaming_const_pattern: Regex::new(r"(?:^|\n)\s*(?:export\s+)?const\s+[A-Z0-9_]+\s*=")
            .unwrap(),
      }
   }

   fn get_language(&self, path: &Path) -> Option<Language> {
      match self.grammar_manager.get_language_for_path(path) {
         Ok(Some(lang)) => Some(lang),
         Ok(None) => None,
         Err(e) => {
            tracing::warn!("failed to load language for {}: {}", path.display(), e);
            None
         },
      }
   }

   fn chunk_with_tree_sitter(&self, content: &str, path: &Path) -> Result<Vec<Chunk>> {
      let language = match self.get_language(path) {
         Some(lang) => lang,
         None => {
            return FallbackChunker::new().chunk(content, path);
         },
      };

      let (mut parser, store) = self.grammar_manager.create_parser_with_store()?;
      parser
         .set_wasm_store(store)
         .map_err(|e| RsgrepError::Chunker(format!("failed to set WASM store: {:?}", e)))?;

      parser
         .set_language(&language)
         .map_err(|e| RsgrepError::Chunker(format!("failed to set language: {}", e)))?;

      let tree = parser
         .parse(content, None)
         .ok_or_else(|| RsgrepError::Chunker("failed to parse file".to_string()))?;

      let root = tree.root_node();
      let file_context = format!("File: {}", path.display());

      let mut chunks = Vec::new();
      let mut block_chunks = Vec::new();
      let mut cursor_index = 0;
      let mut cursor_row = 0;
      let mut saw_definition = false;

      let mut cursor = root.walk();
      for child in root.named_children(&mut cursor) {
         self.visit_node(
            &child,
            content,
            &[file_context.clone()],
            &mut chunks,
            &mut saw_definition,
         );

         let effective = self.unwrap_export(&child);
         let is_definition = self.is_definition_node(&effective, content);

         if is_definition {
            if child.start_byte() > cursor_index {
               let gap_text = &content[cursor_index..child.start_byte()];
               if !gap_text.trim().is_empty() {
                  block_chunks.push(Chunk::new(
                     gap_text.to_string(),
                     cursor_row,
                     child.start_position().row,
                     ChunkType::Block,
                     vec![file_context.clone()],
                  ));
               }
            }

            cursor_index = child.end_byte();
            cursor_row = child.end_position().row;
         }
      }

      if cursor_index < content.len() {
         let tail_text = &content[cursor_index..];
         if !tail_text.trim().is_empty() {
            block_chunks.push(Chunk::new(
               tail_text.to_string(),
               cursor_row,
               root.end_position().row,
               ChunkType::Block,
               vec![file_context.clone()],
            ));
         }
      }

      if !saw_definition {
         return Ok(Vec::new());
      }

      let mut combined = block_chunks;
      combined.extend(chunks);
      combined.sort_by(|a, b| {
         a.start_line
            .cmp(&b.start_line)
            .then(a.end_line.cmp(&b.end_line))
      });

      Ok(combined)
   }

   fn visit_node(
      &self,
      node: &tree_sitter::Node,
      content: &str,
      stack: &[String],
      chunks: &mut Vec<Chunk>,
      saw_definition: &mut bool,
   ) {
      let effective = self.unwrap_export(node);
      let is_definition = self.is_definition_node(&effective, content);
      let mut next_stack = stack.to_vec();

      if is_definition {
         *saw_definition = true;
         let label = self.label_for_node(&effective, content);
         let mut context = stack.to_vec();
         if let Some(label) = label {
            context.push(label);
         }

         let node_text = &content[effective.start_byte()..effective.end_byte()];
         chunks.push(Chunk::new(
            node_text.to_string(),
            effective.start_position().row,
            effective.end_position().row,
            self.classify_node(&effective),
            context.clone(),
         ));

         next_stack = context;
      }

      let mut cursor = effective.walk();
      for child in effective.named_children(&mut cursor) {
         self.visit_node(&child, content, &next_stack, chunks, saw_definition);
      }
   }

   fn unwrap_export<'a>(&self, node: &'a tree_sitter::Node) -> tree_sitter::Node<'a> {
      if node.kind() == "export_statement" && node.named_child_count() > 0 {
         return node.named_child(0).unwrap();
      }
      *node
   }

   fn is_definition_node(&self, node: &tree_sitter::Node, content: &str) -> bool {
      let kind = node.kind();
      matches!(
         kind,
         "function_declaration"
            | "function_definition"
            | "method_definition"
            | "method_declaration"
            | "class_declaration"
            | "class_definition"
            | "interface_declaration"
            | "type_alias_declaration"
            | "type_declaration"
      ) || self.is_top_level_value_def(node, content)
   }

   fn is_top_level_value_def(&self, node: &tree_sitter::Node, content: &str) -> bool {
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

      if self.screaming_const_pattern.is_match(text) {
         return true;
      }

      false
   }

   fn classify_node(&self, node: &tree_sitter::Node) -> ChunkType {
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

   fn get_node_name(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
      if let Some(name_node) = node.child_by_field_name("name") {
         let name = &content[name_node.start_byte()..name_node.end_byte()];
         return Some(name.to_string());
      }

      if let Some(property_node) = node.child_by_field_name("property") {
         let name = &content[property_node.start_byte()..property_node.end_byte()];
         return Some(name.to_string());
      }

      if let Some(identifier_node) = node.child_by_field_name("identifier") {
         let name = &content[identifier_node.start_byte()..identifier_node.end_byte()];
         return Some(name.to_string());
      }

      let mut cursor = node.walk();
      for child in node.named_children(&mut cursor) {
         let child_kind = child.kind();
         if matches!(
            child_kind,
            "identifier" | "property_identifier" | "type_identifier" | "field_identifier"
         ) {
            let name = &content[child.start_byte()..child.end_byte()];
            return Some(name.to_string());
         }

         if child_kind == "variable_declarator"
            && let Some(name) = self.get_node_name(&child, content)
         {
            return Some(name);
         }
      }

      None
   }

   fn label_for_node(&self, node: &tree_sitter::Node, content: &str) -> Option<String> {
      let name = self.get_node_name(node, content);
      let kind = node.kind();

      if kind.contains("class") {
         Some(format!("Class: {}", name.as_deref().unwrap_or("<anonymous class>")))
      } else if kind.contains("method") {
         Some(format!("Method: {}", name.as_deref().unwrap_or("<anonymous method>")))
      } else if kind.contains("interface") {
         Some(format!("Interface: {}", name.as_deref().unwrap_or("<anonymous interface>")))
      } else if kind.contains("type_alias") || kind.contains("type_declaration") {
         Some(format!("Type: {}", name.as_deref().unwrap_or("<anonymous type>")))
      } else if kind.contains("function") {
         Some(format!("Function: {}", name.as_deref().unwrap_or("<anonymous function>")))
      } else if self.is_top_level_value_def(node, content) {
         Some(format!("Function: {}", name.as_deref().unwrap_or("<anonymous function>")))
      } else {
         name.map(|n| format!("Symbol: {}", n))
      }
   }

   fn split_if_too_big(&self, chunk: Chunk) -> Vec<Chunk> {
      let char_count = chunk.content.len();
      let lines: Vec<&str> = chunk.content.lines().collect();
      let line_count = lines.len();

      if line_count <= MAX_LINES && char_count <= MAX_CHARS {
         return vec![chunk];
      }

      if char_count > MAX_CHARS && line_count <= MAX_LINES {
         return self.split_by_chars(chunk);
      }

      let mut sub_chunks = Vec::new();
      let stride = (MAX_LINES - OVERLAP_LINES).max(1);
      let header = self.extract_header_line(&chunk.content);

      let mut i = 0;
      while i < lines.len() {
         let end = (i + MAX_LINES).min(lines.len());
         let sub_lines = &lines[i..end];

         if sub_lines.len() < 3 && i > 0 {
            i += stride;
            continue;
         }

         let mut sub_content = sub_lines.join("\n");
         if let Some(ref h) = header
            && i > 0
            && chunk.chunk_type != Some(ChunkType::Block)
         {
            sub_content = format!("{}\n{}", h, sub_content);
         }

         sub_chunks.push(Chunk::new(
            sub_content,
            chunk.start_line + i,
            chunk.start_line + end,
            chunk.chunk_type.unwrap_or(ChunkType::Other),
            chunk.context.clone(),
         ));

         i += stride;
      }

      sub_chunks
         .into_iter()
         .flat_map(|sc| {
            if sc.content.len() > MAX_CHARS {
               self.split_by_chars(sc)
            } else {
               vec![sc]
            }
         })
         .collect()
   }

   fn split_by_chars(&self, chunk: Chunk) -> Vec<Chunk> {
      let mut chunks = Vec::new();
      let stride = (MAX_CHARS - OVERLAP_CHARS).max(1);
      let content = &chunk.content;

      let mut i = 0;
      while i < content.len() {
         let end = content.ceil_char_boundary(i + MAX_CHARS).min(content.len());
         if end <= i {
            break;
         }

         let sub = &content[i..end].trim();

         let prefix_lines = content[..i].lines().count();
         let sub_line_count = sub.lines().count();

         if !sub.is_empty() {
            chunks.push(Chunk::new(
               sub.to_string(),
               chunk.start_line + prefix_lines,
               chunk.start_line + prefix_lines + sub_line_count,
               chunk.chunk_type.unwrap_or(ChunkType::Other),
               chunk.context.clone(),
            ));
         }

         i = content.ceil_char_boundary(i + stride).min(content.len());
      }

      chunks
   }

   fn extract_header_line(&self, text: &str) -> Option<String> {
      for line in text.lines() {
         let trimmed = line.trim();
         if !trimmed.is_empty() {
            return Some(trimmed.to_string());
         }
      }
      None
   }
}

impl Default for TreeSitterChunker {
   fn default() -> Self {
      Self::new()
   }
}

impl Chunker for TreeSitterChunker {
   fn chunk(&self, content: &str, path: &Path) -> Result<Vec<Chunk>> {
      let raw_chunks = self.chunk_with_tree_sitter(content, path)?;

      let chunks: Vec<Chunk> = raw_chunks
         .into_iter()
         .flat_map(|c| self.split_if_too_big(c))
         .collect();

      Ok(chunks)
   }
}
