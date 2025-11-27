use std::path::Path;

use console::Style;
use syntect::{
   easy::HighlightLines,
   highlighting::ThemeSet,
   parsing::SyntaxSet,
   util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

use super::{Formatter, detect_language, get_semantic_tags, truncate_line};
use crate::types::SearchResult;

pub struct HumanFormatter {
   syntax_set: SyntaxSet,
   theme_set:  ThemeSet,
}

impl Default for HumanFormatter {
   fn default() -> Self {
      Self::new()
   }
}

impl HumanFormatter {
   pub fn new() -> Self {
      Self {
         syntax_set: SyntaxSet::load_defaults_newlines(),
         theme_set:  ThemeSet::load_defaults(),
      }
   }

   fn highlight_code(&self, code: &str, language: Option<&str>) -> String {
      let theme = &self.theme_set.themes["base16-ocean.dark"];

      let syntax = language
         .and_then(|lang| self.syntax_set.find_syntax_by_name(lang))
         .or_else(|| Some(self.syntax_set.find_syntax_plain_text()));

      if let Some(syntax) = syntax {
         let mut highlighter = HighlightLines::new(syntax, theme);
         let mut highlighted = String::new();

         for line in LinesWithEndings::from(code) {
            let ranges = highlighter
               .highlight_line(line, &self.syntax_set)
               .unwrap_or_default();
            let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
            highlighted.push_str(&escaped);
         }

         highlighted
      } else {
         code.to_string()
      }
   }

   fn merge_adjacent_results(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
      if results.is_empty() {
         return results;
      }

      results.sort_by(|a, b| a.path.cmp(&b.path).then(a.start_line.cmp(&b.start_line)));

      let mut merged = Vec::new();
      let mut current: Option<SearchResult> = None;

      for result in results {
         if let Some(ref mut curr) = current {
            if curr.path == result.path
               && result.start_line <= curr.start_line + curr.num_lines + 10
            {
               curr.content.push_str("\n// ...\n");
               curr.content.push_str(&result.content);
               curr.num_lines = result.start_line + result.num_lines - curr.start_line;
               if matches!(
                  result.chunk_type,
                  Some(crate::types::ChunkType::Function) | Some(crate::types::ChunkType::Class)
               ) {
                  curr.chunk_type = result.chunk_type;
               }
            } else {
               merged.push(current.take().unwrap());
               current = Some(result);
            }
         } else {
            current = Some(result);
         }
      }

      if let Some(last) = current {
         merged.push(last);
      }

      merged
   }
}

impl Formatter for HumanFormatter {
   fn format(&self, results: &[SearchResult], show_scores: bool, show_content: bool) -> String {
      if results.is_empty() {
         return "No results found.".to_string();
      }

      let merged = self.merge_adjacent_results(results.to_vec());
      let file_count = merged
         .iter()
         .map(|r| &r.path)
         .collect::<std::collections::HashSet<_>>()
         .len();

      let bold = Style::new().bold();
      let green = Style::new().green();
      let blue = Style::new().blue();
      let dim = Style::new().dim();

      let mut output = String::new();
      output.push('\n');
      output.push_str(
         &bold
            .apply_to(format!(
               "Search Results ({} matches across {} files)",
               merged.len(),
               file_count
            ))
            .to_string(),
      );
      output.push_str("\n\n");

      for (idx, result) in merged.iter().enumerate() {
         let tags = get_semantic_tags(result);
         let tag_str = if !tags.is_empty() {
            format!(" {}", blue.apply_to(format!("[{}]", tags.join(", "))))
         } else {
            String::new()
         };

         let score_str = if show_scores {
            format!(" {}", dim.apply_to(format!("(score: {:.3})", result.score)))
         } else {
            String::new()
         };

         output.push_str(&format!(
            "{}. 📂 {} {}{}{}\n",
            idx + 1,
            green.apply_to(&result.path),
            dim.apply_to(format!(":{}", result.start_line + 1)),
            tag_str,
            score_str
         ));

         let lines: Vec<String> = result.content.lines().map(|s| s.to_string()).collect();
         let max_lines = if show_content { usize::MAX } else { 12 };
         let display_lines = if lines.len() > max_lines {
            let mut truncated = lines[..max_lines].to_vec();
            truncated.push(format!("... (+{} more lines)", lines.len() - max_lines));
            truncated
         } else {
            lines
         };

         let code = display_lines.join("\n");
         let highlighted = self.highlight_code(&code, detect_language(Path::new(&result.path)));

         for (line_idx, line) in highlighted.lines().enumerate() {
            let line_num = result.start_line + line_idx as u32 + 1;
            output.push_str(&format!(
               "   {} │ {}\n",
               dim.apply_to(format!("{:4}", line_num)),
               line
            ));
         }

         output.push('\n');
      }

      output.push_str(
         &dim
            .apply_to(format!("{} matches across {} files", merged.len(), file_count))
            .to_string(),
      );

      output.trim_end().to_string()
   }
}

pub struct AgentFormatter;

impl Default for AgentFormatter {
   fn default() -> Self {
      Self::new()
   }
}

impl AgentFormatter {
   pub fn new() -> Self {
      Self
   }

   fn clean_snippet_lines(snippet: &str) -> Vec<String> {
      const NOISE_PREFIXES: &[&str] =
         &["File:", "Top comments:", "Preamble:", "(anchor)", "Imports:", "Exports:", "---"];

      snippet
         .split('\n')
         .map(|line| {
            let mut next = line.trim_end().to_string();
            if let Some(idx) = next.find("File:") {
               next = next[..idx].trim_end().to_string();
            }
            next
         })
         .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !NOISE_PREFIXES.iter().any(|p| trimmed.starts_with(p))
         })
         .map(|line| truncate_line(&line, 140))
         .collect()
   }
}

impl Formatter for AgentFormatter {
   fn format(&self, results: &[SearchResult], _show_scores: bool, show_content: bool) -> String {
      if results.is_empty() {
         return "No results found.".to_string();
      }

      let file_count = results
         .iter()
         .map(|r| &r.path)
         .collect::<std::collections::HashSet<_>>()
         .len();

      let max_lines = if show_content { usize::MAX } else { 16 };
      let mut output = String::new();

      for result in results {
         let line = result.start_line + 1;
         let tags = get_semantic_tags(result);
         let tag_str = if !tags.is_empty() {
            format!(" [{}]", tags.join(","))
         } else {
            String::new()
         };

         output.push_str(&format!("{}:{}{}\n", result.path, line, tag_str));

         let lines = Self::clean_snippet_lines(&result.content);
         let display_lines = if lines.len() > max_lines {
            let mut truncated = lines[..max_lines].to_vec();
            truncated.push(format!("... (+{} more lines)", lines.len() - max_lines));
            truncated
         } else {
            lines
         };

         for line in display_lines {
            output.push_str(&format!("  {}\n", line));
         }

         output.push('\n');
      }

      output.push_str(&format!(
         "osgrep results ({} matches across {} files)",
         results.len(),
         file_count
      ));

      output.trim().to_string()
   }
}

pub struct CompactFormatter;

impl Formatter for CompactFormatter {
   fn format(&self, results: &[SearchResult], _show_scores: bool, _show_content: bool) -> String {
      let mut unique_paths: Vec<&str> = results
         .iter()
         .map(|r| r.path.as_str())
         .collect::<std::collections::HashSet<_>>()
         .into_iter()
         .collect();

      unique_paths.sort();
      unique_paths.join("\n")
   }
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::types::ChunkType;

   fn create_test_result(path: &str, start_line: u32, content: &str) -> SearchResult {
      SearchResult {
         path: path.to_string(),
         content: content.to_string(),
         score: 0.95,
         start_line,
         num_lines: content.lines().count() as u32,
         chunk_type: Some(ChunkType::Function),
         is_anchor: Some(false),
      }
   }

   #[test]
   fn test_compact_formatter() {
      let results = vec![
         create_test_result("src/main.rs", 10, "fn main() {}"),
         create_test_result("src/lib.rs", 5, "pub fn test() {}"),
         create_test_result("src/main.rs", 20, "fn other() {}"),
      ];

      let formatter = CompactFormatter;
      let output = formatter.format(&results, false, false);

      assert!(output.contains("src/lib.rs"));
      assert!(output.contains("src/main.rs"));
      assert_eq!(output.lines().count(), 2);
   }

   #[test]
   fn test_agent_formatter() {
      let results = vec![create_test_result(
         "src/auth.rs",
         41,
         "pub fn authenticate() {\n  let token = jwt::sign();\n}",
      )];

      let formatter = AgentFormatter::new();
      let output = formatter.format(&results, false, false);

      assert!(output.contains("src/auth.rs:42"));
      assert!(output.contains("[Definition]"));
      assert!(output.contains("pub fn authenticate()"));
   }
}
