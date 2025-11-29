//! Anchor chunk generation for file context and metadata.
//!
//! Creates special anchor chunks containing file metadata, imports, exports,
//! and preamble to provide context for semantic code search.

use std::{fmt::Write, path::Path, sync::LazyLock};

use regex::Regex;

use crate::{
   Str,
   types::{Chunk, ChunkType},
};

/// Creates an anchor chunk containing file metadata and context.
///
/// Extracts top-level comments, imports, exports, and preamble to provide
/// context for code search. Returns a special chunk marked as an anchor.
pub fn create_anchor_chunk(content: &Str, path: &Path) -> Chunk {
   let lines: Vec<&str> = content.as_str().lines().collect();
   let top_comments = extract_top_comments(&lines);
   let imports = extract_imports(&lines);
   let exports = extract_exports(&lines);

   let mut preamble = Vec::new();
   let mut non_blank = 0;
   let mut total_chars = 0;

   for line in &lines {
      let trimmed = line.trim();
      if trimmed.is_empty() {
         continue;
      }
      preamble.push(*line);
      non_blank += 1;
      total_chars += line.len();
      if non_blank >= 30 || total_chars >= 1200 {
         break;
      }
   }

   let mut anchor_text = String::new();
   write!(anchor_text, "File: {}", path.display()).unwrap();

   if !imports.is_empty() {
      write!(anchor_text, "\n\nImports: {}", imports.join(", ")).unwrap();
   }

   if !exports.is_empty() {
      write!(anchor_text, "\n\nExports: {}", exports.join(", ")).unwrap();
   }

   if !top_comments.is_empty() {
      write!(anchor_text, "\n\nTop comments:\n{}", top_comments.join("\n")).unwrap();
   }

   if !preamble.is_empty() {
      write!(anchor_text, "\n\nPreamble:\n{}", preamble.join("\n")).unwrap();
   }

   anchor_text.push_str("\n\n---\n\n(anchor)");
   let approx_end_line = lines.len().min(non_blank.max(preamble.len()).max(5));

   let mut chunk =
      Chunk::new(Str::from_string(anchor_text), 0, approx_end_line, ChunkType::Block, &[
         format!("File: {}", path.display()).into(),
         "Anchor".into(),
      ]);
   chunk.chunk_index = Some(-1);
   chunk.is_anchor = Some(true);
   chunk
}

fn extract_top_comments(lines: &[&str]) -> Vec<String> {
   let mut comments = Vec::new();
   let mut in_block = false;

   for line in lines {
      let trimmed = line.trim();

      if in_block {
         comments.push(line.to_string());
         if trimmed.contains("*/") {
            in_block = false;
         }
         continue;
      }

      if trimmed.is_empty() {
         comments.push(line.to_string());
         continue;
      }

      if trimmed.starts_with("//") || trimmed.starts_with("#!") || trimmed.starts_with("# ") {
         comments.push(line.to_string());
         continue;
      }

      if trimmed.starts_with("/*") {
         comments.push(line.to_string());
         if !trimmed.contains("*/") {
            in_block = true;
         }
         continue;
      }

      break;
   }

   while let Some(last) = comments.last() {
      if last.trim().is_empty() {
         comments.pop();
      } else {
         break;
      }
   }

   comments
}

macro_rules! static_regex {
   ($($name:ident = $regex:expr),* $(,)?) => {
      $(
         pub(crate) static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($regex).unwrap());
      )*
   };
}

static_regex! {
   IMPORT_FROM_REGEX    = r#"from\s+["']([^"']+)["']"#,
   IMPORT_REGEX         = r#"^import\s+["']([^"']+)["']"#,
   IMPORT_AS_REGEX      = r"import\s+(?:\*\s+as\s+)?([A-Za-z0-9_$]+)",
   REQUIRE_REGEX        = r#"require\(\s*["']([^"']+)["']\s*\)"#,
   EXPORT_REGEX         = r"^export\s+(?:default\s+)?(class|function|const|let|var|interface|type|enum)\s+([A-Za-z0-9_$]+)",
   EXPORT_BRACE_REGEX   = r"^export\s+\{([^}]+)\}",
   CONST_EXPORT_REGEX   = r"(?:^|\n)\s*(?:export\s+)?const\s+[A-Z0-9_]+\s*=",
}

fn extract_imports(lines: &[&str]) -> Vec<String> {
   let mut modules = Vec::new();
   let limit = 200.min(lines.len());

   for line in &lines[..limit] {
      let trimmed = line.trim();
      if trimmed.is_empty() {
         continue;
      }

      if trimmed.starts_with("import ") {
         if let Some(caps) = IMPORT_FROM_REGEX.captures(trimmed)
            && let Some(m) = caps.get(1)
         {
            modules.push(m.as_str().to_string());
            continue;
         }

         if let Some(caps) = IMPORT_REGEX.captures(trimmed)
            && let Some(m) = caps.get(1)
         {
            modules.push(m.as_str().to_string());
            continue;
         }

         if let Some(caps) = IMPORT_AS_REGEX.captures(trimmed)
            && let Some(m) = caps.get(1)
         {
            modules.push(m.as_str().to_string());
         }
         continue;
      }

      if trimmed.starts_with("use ") {
         if let Some(module) = trimmed
            .strip_prefix("use ")
            .and_then(|s| s.split([':', ';']).next())
         {
            modules.push(module.trim().to_string());
         }
         continue;
      }

      if let Some(caps) = REQUIRE_REGEX.captures(trimmed)
         && let Some(m) = caps.get(1)
      {
         modules.push(m.as_str().to_string());
      }
   }

   modules.sort();
   modules.dedup();
   modules
}

fn extract_exports(lines: &[&str]) -> Vec<String> {
   let mut exports = Vec::new();
   let limit = 200.min(lines.len());

   for line in &lines[..limit] {
      let trimmed = line.trim();
      if !trimmed.starts_with("export") && !trimmed.contains("module.exports") {
         continue;
      }

      if let Some(caps) = EXPORT_REGEX.captures(trimmed)
         && let Some(m) = caps.get(2)
      {
         exports.push(m.as_str().to_string());
         continue;
      }

      if let Some(caps) = EXPORT_BRACE_REGEX.captures(trimmed)
         && let Some(m) = caps.get(1)
      {
         let names: Vec<String> = m
            .as_str()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
         exports.extend(names);
         continue;
      }

      if trimmed.starts_with("export default") {
         exports.push("default".to_string());
      }

      if trimmed.contains("module.exports") {
         exports.push("module.exports".to_string());
      }
   }

   exports.sort();
   exports.dedup();
   exports
}
