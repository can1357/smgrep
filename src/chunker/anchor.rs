use std::path::Path;

use crate::types::{Chunk, ChunkType};

pub fn create_anchor_chunk(content: &str, path: &Path) -> Chunk {
   let lines: Vec<&str> = content.lines().collect();
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

   let mut sections = Vec::new();
   sections.push(format!("File: {}", path.display()));

   if !imports.is_empty() {
      sections.push(format!("Imports: {}", imports.join(", ")));
   }

   if !exports.is_empty() {
      sections.push(format!("Exports: {}", exports.join(", ")));
   }

   if !top_comments.is_empty() {
      sections.push(format!("Top comments:\n{}", top_comments.join("\n")));
   }

   if !preamble.is_empty() {
      sections.push(format!("Preamble:\n{}", preamble.join("\n")));
   }

   sections.push("---".to_string());
   sections.push("(anchor)".to_string());

   let anchor_text = sections.join("\n\n");
   let approx_end_line = lines.len().min(non_blank.max(preamble.len()).max(5));

   let mut chunk = Chunk::new(anchor_text, 0, approx_end_line, ChunkType::Block, vec![
      format!("File: {}", path.display()),
      "Anchor".to_string(),
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

fn extract_imports(lines: &[&str]) -> Vec<String> {
   let mut modules = Vec::new();
   let limit = 200.min(lines.len());

   for line in &lines[..limit] {
      let trimmed = line.trim();
      if trimmed.is_empty() {
         continue;
      }

      if trimmed.starts_with("import ") {
         if let Some(caps) = regex::Regex::new(r#"from\s+["']([^"']+)["']"#)
            .ok()
            .and_then(|re| re.captures(trimmed))
            && let Some(m) = caps.get(1)
         {
            modules.push(m.as_str().to_string());
            continue;
         }

         if let Some(caps) = regex::Regex::new(r#"^import\s+["']([^"']+)["']"#)
            .ok()
            .and_then(|re| re.captures(trimmed))
            && let Some(m) = caps.get(1)
         {
            modules.push(m.as_str().to_string());
            continue;
         }

         if let Some(caps) = regex::Regex::new(r"import\s+(?:\*\s+as\s+)?([A-Za-z0-9_$]+)")
            .ok()
            .and_then(|re| re.captures(trimmed))
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

      if let Some(caps) = regex::Regex::new(r#"require\(\s*["']([^"']+)["']\s*\)"#)
         .ok()
         .and_then(|re| re.captures(trimmed))
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

      if let Some(caps) = regex::Regex::new(
         r"^export\s+(?:default\s+)?(class|function|const|let|var|interface|type|enum)\s+([A-Za-z0-9_$]+)",
      )
      .ok()
      .and_then(|re| re.captures(trimmed))
         && let Some(m) = caps.get(2) {
            exports.push(m.as_str().to_string());
            continue;
         }

      if let Some(caps) = regex::Regex::new(r"^export\s+\{([^}]+)\}")
         .ok()
         .and_then(|re| re.captures(trimmed))
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
