use std::collections::HashMap;

use crate::types::{ChunkType, SearchResult};

pub fn apply_structural_boost(results: &mut [SearchResult]) {
   for result in results.iter_mut() {
      match result.chunk_type {
         Some(ChunkType::Function)
         | Some(ChunkType::Class)
         | Some(ChunkType::Interface)
         | Some(ChunkType::Method)
         | Some(ChunkType::TypeAlias) => {
            result.score *= 1.25;
         },
         _ => {},
      }

      if is_test_file(&result.path) {
         result.score *= 0.85;
      }

      if is_doc_or_config(&result.path) {
         result.score *= 0.5;
      }
   }
}

pub fn deduplicate(results: Vec<SearchResult>) -> Vec<SearchResult> {
   let mut seen: HashMap<(String, u32), usize> = HashMap::new();
   let mut deduplicated: Vec<SearchResult> = Vec::with_capacity(results.len());

   for result in results {
      let key = (result.path.clone(), result.start_line);

      match seen.get(&key) {
         Some(&idx) => {
            if result.score > deduplicated[idx].score {
               deduplicated[idx] = result;
            }
         },
         None => {
            seen.insert(key, deduplicated.len());
            deduplicated.push(result);
         },
      }
   }

   deduplicated
}

pub fn apply_per_file_limit(results: Vec<SearchResult>, limit: usize) -> Vec<SearchResult> {
   let mut by_path: HashMap<String, Vec<SearchResult>> = HashMap::new();

   for result in results {
      by_path.entry(result.path.clone()).or_default().push(result);
   }

   for results in by_path.values_mut() {
      results.sort_by(|a, b| {
         b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
      });
      results.truncate(limit);
   }

   let mut final_results: Vec<SearchResult> = by_path.into_values().flatten().collect();

   final_results.sort_by(|a, b| {
      b.score
         .partial_cmp(&a.score)
         .unwrap_or(std::cmp::Ordering::Equal)
   });

   final_results
}

fn is_test_file(path: &str) -> bool {
   let lower = path.to_lowercase();
   lower.contains(".test.") || lower.contains(".spec.") || lower.contains("__tests__")
}

fn is_doc_or_config(path: &str) -> bool {
   let lower = path.to_lowercase();
   lower.ends_with(".md")
      || lower.ends_with(".mdx")
      || lower.ends_with(".txt")
      || lower.ends_with(".json")
      || lower.ends_with(".yaml")
      || lower.ends_with(".yml")
      || lower.ends_with(".lock")
      || lower.contains("/docs/")
}

#[cfg(test)]
mod tests {
   use super::*;

   fn make_result(path: &str, start_line: u32, score: f32, chunk_type: ChunkType) -> SearchResult {
      SearchResult {
         path: path.to_string(),
         content: String::new(),
         score,
         start_line,
         num_lines: 10,
         chunk_type: Some(chunk_type),
         is_anchor: Some(false),
      }
   }

   #[test]
   fn test_apply_structural_boost() {
      let mut results = vec![
         make_result("src/main.rs", 1, 1.0, ChunkType::Function),
         make_result("src/lib.rs", 1, 1.0, ChunkType::Block),
         make_result("src/test.rs", 1, 1.0, ChunkType::Function),
         make_result("README.md", 1, 1.0, ChunkType::Other),
      ];

      apply_structural_boost(&mut results);

      assert!((results[0].score - 1.0625).abs() < 1e-6);
      assert!((results[1].score - 1.0).abs() < 1e-6);
      assert!((results[2].score - 1.0625).abs() < 1e-6);
      assert!((results[3].score - 0.5).abs() < 1e-6);
   }

   #[test]
   fn test_deduplicate() {
      let results = vec![
         make_result("src/main.rs", 10, 1.0, ChunkType::Function),
         make_result("src/main.rs", 10, 2.0, ChunkType::Function),
         make_result("src/lib.rs", 20, 1.5, ChunkType::Class),
      ];

      let deduped = deduplicate(results);
      assert_eq!(deduped.len(), 2);
      assert!((deduped[0].score - 2.0).abs() < 1e-6);
      assert_eq!(deduped[0].path, "src/main.rs");
   }

   #[test]
   fn test_apply_per_file_limit() {
      let results = vec![
         make_result("file1.rs", 1, 5.0, ChunkType::Function),
         make_result("file1.rs", 2, 4.0, ChunkType::Function),
         make_result("file1.rs", 3, 3.0, ChunkType::Function),
         make_result("file2.rs", 1, 2.0, ChunkType::Function),
      ];

      let limited = apply_per_file_limit(results, 2);
      assert_eq!(limited.len(), 3);

      let file1_count = limited.iter().filter(|r| r.path == "file1.rs").count();
      assert_eq!(file1_count, 2);
   }

   #[test]
   fn test_is_test_file() {
      assert!(is_test_file("src/main.test.ts"));
      assert!(is_test_file("src/component.spec.js"));
      assert!(is_test_file("src/__tests__/utils.js"));
      assert!(!is_test_file("src/main.rs"));
   }

   #[test]
   fn test_is_doc_or_config() {
      assert!(is_doc_or_config("README.md"));
      assert!(is_doc_or_config("package.json"));
      assert!(is_doc_or_config("config.yaml"));
      assert!(is_doc_or_config("docs/guide.md"));
      assert!(!is_doc_or_config("src/main.rs"));
   }
}
