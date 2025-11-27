use std::{path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::store::Store as RsgrepStore;

#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
   path:       String,
   score:      f32,
   content:    String,
   #[serde(skip_serializing_if = "Option::is_none")]
   chunk_type: Option<String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   start_line: Option<usize>,
   #[serde(skip_serializing_if = "Option::is_none")]
   end_line:   Option<usize>,
}

#[derive(Debug, Serialize)]
struct JsonOutput {
   results: Vec<SearchResult>,
}

#[allow(clippy::too_many_arguments)]
pub async fn execute(
   query: String,
   path: Option<PathBuf>,
   max: usize,
   per_file: usize,
   content: bool,
   compact: bool,
   scores: bool,
   sync: bool,
   dry_run: bool,
   json: bool,
   no_rerank: bool,
   plain: bool,
   store_id: Option<String>,
) -> Result<()> {
   let root = std::env::current_dir().context("failed to get current directory")?;
   let search_path = path.unwrap_or_else(|| root.clone());

   if json && let Some(results) = try_server_fastpath(&query, max, !no_rerank, &search_path).await?
   {
      println!("{}", serde_json::to_string(&JsonOutput { results })?);
      return Ok(());
   }

   let resolved_store_id = store_id.unwrap_or_else(|| {
      root
         .file_name()
         .and_then(|s| s.to_str())
         .unwrap_or("default")
         .to_string()
   });

   if dry_run {
      if json {
         println!("{}", serde_json::to_string(&JsonOutput { results: vec![] })?);
      } else {
         println!("Dry run: would search for '{}' in {:?}", query, search_path);
         println!("Store ID: {}", resolved_store_id);
         println!("Max results: {}", max);
      }
      return Ok(());
   }

   if sync && !json {
      let spinner = ProgressBar::new_spinner();
      spinner.set_style(
         ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
      );
      spinner.enable_steady_tick(Duration::from_millis(100));
      spinner.set_message("Syncing files to index...");

      tokio::time::sleep(Duration::from_millis(100)).await;

      spinner.finish_with_message("Sync complete");
   }

   let results = perform_search(&query, &search_path, max, per_file, !no_rerank).await?;

   if results.is_empty() {
      if !json {
         println!("No results found for '{}'", query);
         if !sync {
            println!("\nTip: Use --sync to re-index before searching");
         }
      } else {
         println!("{}", serde_json::to_string(&JsonOutput { results: vec![] })?);
      }
      return Ok(());
   }

   if json {
      println!("{}", serde_json::to_string(&JsonOutput { results })?);
   } else {
      format_results(&results, &query, &root, content, compact, scores, plain)?;
   }

   Ok(())
}

async fn try_server_fastpath(
   query: &str,
   max: usize,
   rerank: bool,
   path: &PathBuf,
) -> Result<Option<Vec<SearchResult>>> {
   let port = std::env::var("RSGREP_PORT").unwrap_or_else(|_| "4444".to_string());
   let base_url = format!("http://localhost:{}", port);

   let client = reqwest::Client::builder()
      .timeout(Duration::from_millis(100))
      .build()?;

   if let Ok(response) = client.get(format!("{}/health", base_url)).send().await
      && response.status().is_success()
   {
      #[derive(Serialize)]
      struct SearchRequest {
         query:  String,
         limit:  usize,
         rerank: bool,
         path:   String,
      }

      let search_response = client
         .post(format!("{}/search", base_url))
         .json(&SearchRequest {
            query: query.to_string(),
            limit: max,
            rerank,
            path: path.to_string_lossy().to_string(),
         })
         .send()
         .await?;

      if search_response.status().is_success() {
         #[derive(Deserialize)]
         struct ServerResponse {
            results: Vec<SearchResult>,
         }

         let payload: ServerResponse = search_response.json().await?;
         return Ok(Some(payload.results));
      }
   }

   Ok(None)
}

async fn perform_search(
   query: &str,
   path: &PathBuf,
   max: usize,
   per_file: usize,
   rerank: bool,
) -> Result<Vec<SearchResult>> {
   let store_id = crate::git::resolve_store_id(path)?;

   let store = std::sync::Arc::new(crate::store::LanceStore::new()?);
   let embedder = std::sync::Arc::new(crate::embed::worker::EmbedWorker::new()?);

   if RsgrepStore::is_empty(&*store, &store_id).await? {
      let fs = crate::file::LocalFileSystem::new();
      let chunker = crate::chunker::treesitter::TreeSitterChunker::new();

      let sync_engine = crate::sync::SyncEngine::new(fs, chunker, embedder.clone(), store.clone());

      sync_engine
         .initial_sync(&store_id, path, false, None)
         .await?;
   }

   let engine = crate::search::SearchEngine::new(store, embedder);
   let response = engine
      .search(&store_id, query, max, per_file, None, rerank)
      .await?;

   let root_str = path.to_string_lossy().to_string();

   let results = response
      .results
      .into_iter()
      .map(|r| {
         let rel_path = r
            .path
            .strip_prefix(&root_str)
            .unwrap_or(&r.path)
            .trim_start_matches('/')
            .to_string();

         SearchResult {
            path:       rel_path,
            score:      r.score,
            content:    r.content,
            chunk_type: Some(format!("{:?}", r.chunk_type).to_lowercase()),
            start_line: Some(r.start_line as usize),
            end_line:   Some((r.start_line + r.num_lines) as usize),
         }
      })
      .collect();

   Ok(results)
}

fn format_results(
   results: &[SearchResult],
   query: &str,
   root: &PathBuf,
   content: bool,
   compact: bool,
   scores: bool,
   plain: bool,
) -> Result<()> {
   if compact {
      for result in results {
         println!("{}", result.path);
      }
      return Ok(());
   }

   if !plain {
      println!("\n{}", style(format!("Search results for: {}", query)).bold());
      println!("{}", style(format!("Root: {}\n", root.display())).dim());
   } else {
      println!("\nSearch results for: {}", query);
      println!("Root: {}\n", root.display());
   }

   for (i, result) in results.iter().enumerate() {
      if !plain {
         print!("{} ", style(format!("{}.", i + 1)).bold().cyan());
         print!("{}", style(&result.path).green());

         if scores {
            print!(" {}", style(format!("(score: {:.3})", result.score)).dim());
         }

         if let Some(ref chunk_type) = result.chunk_type {
            print!(" {}", style(format!("[{}]", chunk_type)).yellow());
         }

         println!();

         if content || result.content.lines().count() <= 5 {
            for line in result.content.lines() {
               println!("  {}", line);
            }
         } else {
            let snippet: Vec<&str> = result.content.lines().take(3).collect();
            for line in snippet {
               println!("  {}", line);
            }
            println!("  {}", style("...").dim());
         }
      } else {
         print!("{}. {}", i + 1, result.path);

         if scores {
            print!(" (score: {:.3})", result.score);
         }

         if let Some(ref chunk_type) = result.chunk_type {
            print!(" [{}]", chunk_type);
         }

         println!();

         if content {
            for line in result.content.lines() {
               println!("  {}", line);
            }
         }
      }

      println!();
   }

   Ok(())
}
