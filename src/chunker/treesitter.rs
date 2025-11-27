use std::path::Path;

use tree_sitter::{Language, Parser};

use crate::{
   chunker::{
      Chunker, MAX_CHARS, MAX_LINES, OVERLAP_CHARS, OVERLAP_LINES, fallback::FallbackChunker,
   },
   error::{Result, RsgrepError},
   types::{Chunk, ChunkType},
};

pub struct TreeSitterChunker {
   parser: Parser,
}

impl TreeSitterChunker {
   pub fn new() -> Self {
      Self { parser: Parser::new() }
   }

   fn get_language(&self, path: &Path) -> Option<Language> {
      let ext = path.extension()?.to_str()?.to_lowercase();
      match ext.as_str() {
         // JavaScript/TypeScript
         "js" | "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
         "jsx" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),

         // Python
         "py" => Some(tree_sitter_python::LANGUAGE.into()),

         // Go
         "go" => Some(tree_sitter_go::LANGUAGE.into()),

         // Rust
         "rs" => Some(tree_sitter_rust::LANGUAGE.into()),

         // C/C++
         "c" => Some(tree_sitter_c::LANGUAGE.into()),
         "h" => Some(tree_sitter_c::LANGUAGE.into()),
         "cpp" | "cc" | "cxx" | "c++" => Some(tree_sitter_cpp::LANGUAGE.into()),
         "hpp" | "hxx" | "h++" => Some(tree_sitter_cpp::LANGUAGE.into()),

         // Java
         "java" => Some(tree_sitter_java::LANGUAGE.into()),

         // Ruby
         "rb" => Some(tree_sitter_ruby::LANGUAGE.into()),

         // PHP
         "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),

         // Swift
         "swift" => Some(tree_sitter_swift::LANGUAGE.into()),

         // Web
         "html" | "htm" => Some(tree_sitter_html::LANGUAGE.into()),
         "css" => Some(tree_sitter_css::LANGUAGE.into()),

         // Shell
         "sh" | "bash" => Some(tree_sitter_bash::LANGUAGE.into()),

         // Lua
         "lua" => Some(tree_sitter_lua::LANGUAGE.into()),

         // Elixir
         "ex" | "exs" => Some(tree_sitter_elixir::LANGUAGE.into()),

         // Haskell
         "hs" => Some(tree_sitter_haskell::LANGUAGE.into()),

         // C#
         "cs" => Some(tree_sitter_c_sharp::LANGUAGE.into()),

         // Zig
         "zig" => Some(tree_sitter_zig::LANGUAGE.into()),

         // Nix
         "nix" => Some(tree_sitter_nix::LANGUAGE.into()),

         // R
         "r" => Some(tree_sitter_r::LANGUAGE.into()),

         // Julia
         "jl" => Some(tree_sitter_julia::LANGUAGE.into()),

         // Erlang
         "erl" | "hrl" => Some(tree_sitter_erlang::LANGUAGE.into()),

         // Clojure
         "clj" | "cljs" | "cljc" | "edn" => Some(tree_sitter_clojure::LANGUAGE.into()),

         // Common Lisp
         "lisp" | "lsp" | "cl" => Some(tree_sitter_commonlisp::LANGUAGE_COMMONLISP.into()),

         // Racket
         "rkt" => Some(tree_sitter_racket::LANGUAGE.into()),

         // Assembly
         "s" | "asm" => Some(tree_sitter_asm::LANGUAGE.into()),

         // Config/data formats
         "yaml" | "yml" => Some(tree_sitter_yaml::LANGUAGE.into()),
         "json" => Some(tree_sitter_json::LANGUAGE.into()),

         _ => None,
      }
   }

   fn chunk_with_tree_sitter(&mut self, content: &str, path: &Path) -> Result<Vec<Chunk>> {
      let language = match self.get_language(path) {
         Some(lang) => lang,
         None => {
            return FallbackChunker::new().chunk(content, path);
         },
      };

      self
         .parser
         .set_language(&language)
         .map_err(|e| RsgrepError::Chunker(format!("Failed to set language: {}", e)))?;

      let tree = self
         .parser
         .parse(content, None)
         .ok_or_else(|| RsgrepError::Chunker("Failed to parse file".to_string()))?;

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
         let is_definition = self.is_definition_node(&effective);

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
      let is_definition = self.is_definition_node(&effective);
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

   fn is_definition_node(&self, node: &tree_sitter::Node) -> bool {
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
      ) || self.is_top_level_value_def(node)
   }

   fn is_top_level_value_def(&self, node: &tree_sitter::Node) -> bool {
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

      true
   }

   fn classify_node(&self, node: &tree_sitter::Node) -> ChunkType {
      let kind = node.kind();
      if kind.contains("class") {
         ChunkType::Class
      } else if kind.contains("interface") {
         ChunkType::Interface
      } else if kind.contains("type_alias") || kind.contains("type_declaration") {
         ChunkType::TypeAlias
      } else if self.is_definition_node(node) {
         ChunkType::Function
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
      } else if self.is_top_level_value_def(node) {
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
         let end = content
            .char_indices()
            .map(|(idx, _)| idx)
            .filter(|&idx| idx <= i + MAX_CHARS)
            .last()
            .unwrap_or(content.len())
            .min(content.len());

         if end <= i {
            break;
         }

         let sub = &content[i..end];
         if sub.trim().is_empty() {
            break;
         }

         let prefix_lines = content[..i].lines().count();
         let sub_line_count = sub.lines().count();

         chunks.push(Chunk::new(
            sub.to_string(),
            chunk.start_line + prefix_lines,
            chunk.start_line + prefix_lines + sub_line_count,
            chunk.chunk_type.unwrap_or(ChunkType::Other),
            chunk.context.clone(),
         ));

         let next_i = content
            .char_indices()
            .map(|(idx, _)| idx)
            .filter(|&idx| idx >= i + stride)
            .next()
            .unwrap_or(content.len());
         i = next_i;
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
      let mut chunker = Self::new();
      let raw_chunks = chunker.chunk_with_tree_sitter(content, path)?;

      let chunks: Vec<Chunk> = raw_chunks
         .into_iter()
         .flat_map(|c| chunker.split_if_too_big(c))
         .collect();

      Ok(chunks)
   }
}
