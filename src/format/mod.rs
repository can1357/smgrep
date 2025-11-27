//! Output formatting for search results.
//!
//! Provides different formatters for displaying search results:
//! - [`HumanFormatter`](text::HumanFormatter): Rich TTY output with syntax
//!   highlighting and colors
//! - [`AgentFormatter`](text::AgentFormatter): Compact pipe-friendly output for
//!   LLMs
//! - [`CompactFormatter`](text::CompactFormatter): File paths only (like `grep
//!   -l`)
//! - [`JsonFormatter`](json::JsonFormatter): Machine-readable JSON output

pub mod json;
pub mod text;

use std::path::Path;

use crate::types::SearchResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
   Human,
   Agent,
   Compact,
   Json,
}

pub trait Formatter {
   fn format(&self, results: &[SearchResult], show_scores: bool, show_content: bool) -> String;
}

pub fn create_formatter(mode: OutputMode) -> Box<dyn Formatter> {
   match mode {
      OutputMode::Human => Box::new(text::HumanFormatter::new()),
      OutputMode::Agent => Box::new(text::AgentFormatter::new()),
      OutputMode::Compact => Box::new(text::CompactFormatter),
      OutputMode::Json => Box::new(json::JsonFormatter),
   }
}

pub fn detect_output_mode(json: bool, compact: bool) -> OutputMode {
   if json {
      OutputMode::Json
   } else if compact {
      OutputMode::Compact
   } else if is_terminal::is_terminal(std::io::stdout()) {
      OutputMode::Human
   } else {
      OutputMode::Agent
   }
}

pub fn get_semantic_tags(result: &SearchResult) -> Vec<&'static str> {
   use crate::types::ChunkType;
   let mut tags = Vec::new();

   match result.chunk_type {
      Some(ChunkType::Function)
      | Some(ChunkType::Class)
      | Some(ChunkType::Interface)
      | Some(ChunkType::TypeAlias) => {
         tags.push("Definition");
      },
      _ => {},
   }

   let path_str = result.path.to_lowercase();
   if path_str.contains("test") || path_str.contains("spec") {
      tags.push("Test");
   }

   if result.is_anchor.unwrap_or(false) {
      tags.push("Anchor");
   }

   tags
}

pub fn truncate_line(line: &str, max_len: usize) -> String {
   if line.len() <= max_len {
      line.to_string()
   } else {
      format!("{}...", &line[..max_len])
   }
}

pub fn detect_language(path: &Path) -> Option<&'static str> {
   path
      .extension()
      .and_then(|ext| ext.to_str())
      .and_then(|ext| match ext.to_lowercase().as_str() {
         "ts" | "tsx" => Some("typescript"),
         "js" | "jsx" => Some("javascript"),
         "py" => Some("python"),
         "rs" => Some("rust"),
         "go" => Some("go"),
         "java" => Some("java"),
         "c" => Some("c"),
         "cpp" | "cc" | "cxx" => Some("cpp"),
         "h" | "hpp" => Some("cpp"),
         "cs" => Some("csharp"),
         "rb" => Some("ruby"),
         "php" => Some("php"),
         "swift" => Some("swift"),
         "kt" => Some("kotlin"),
         "scala" => Some("scala"),
         "json" => Some("json"),
         "xml" => Some("xml"),
         "html" | "htm" => Some("html"),
         "css" => Some("css"),
         "scss" | "sass" => Some("scss"),
         "md" => Some("markdown"),
         "yaml" | "yml" => Some("yaml"),
         "toml" => Some("toml"),
         "sh" | "bash" => Some("bash"),
         "sql" => Some("sql"),
         _ => None,
      })
}
