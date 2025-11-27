use std::{path::PathBuf, process::Command, time::Duration};

use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;

use crate::ipc::{self, Request, Response};

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
   #[serde(skip_serializing_if = "Option::is_none")]
   is_anchor:  Option<bool>,
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

   let resolved_store_id = store_id
      .map(Ok)
      .unwrap_or_else(|| crate::git::resolve_store_id(&search_path))?;

   if let Some(results) =
      try_daemon_search(&query, max, !no_rerank, &search_path, &resolved_store_id).await?
   {
      if json {
         println!("{}", serde_json::to_string(&JsonOutput { results })?);
      } else {
         format_results(&results, &query, &root, content, compact, scores, plain)?;
      }
      return Ok(());
   }

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

   let results =
      perform_search(&query, &search_path, &resolved_store_id, max, per_file, !no_rerank).await?;

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

async fn try_daemon_search(
   query: &str,
   max: usize,
   rerank: bool,
   path: &PathBuf,
   store_id: &str,
) -> Result<Option<Vec<SearchResult>>> {
   let socket_path = ipc::socket_path(store_id);

   let stream = match UnixStream::connect(&socket_path).await {
      Ok(s) => s,
      Err(_) => {
         spawn_daemon(path)?;

         for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if let Ok(s) = UnixStream::connect(&socket_path).await {
               return send_search_request(s, query, max, rerank, path)
                  .await
                  .map(Some);
            }
         }

         return Ok(None);
      },
   };

   send_search_request(stream, query, max, rerank, path)
      .await
      .map(Some)
}

async fn send_search_request(
   mut stream: UnixStream,
   query: &str,
   max: usize,
   rerank: bool,
   path: &PathBuf,
) -> Result<Vec<SearchResult>> {
   let request = Request::Search {
      query: query.to_string(),
      limit: max,
      path: Some(path.to_string_lossy().to_string()),
      rerank,
   };

   let mut buffer = ipc::SocketBuffer::new();
   buffer.send(&mut stream, &request).await?;
   let response: Response = buffer.recv(&mut stream).await?;

   match response {
      Response::Search(search_response) => {
         let results = search_response
            .results
            .into_iter()
            .map(|r| SearchResult {
               path:       r.path,
               score:      r.score,
               content:    r.content,
               chunk_type: r.chunk_type.map(|ct| format!("{:?}", ct).to_lowercase()),
               start_line: Some(r.start_line as usize),
               end_line:   Some((r.start_line + r.num_lines) as usize),
               is_anchor:  r.is_anchor,
            })
            .collect();
         Ok(results)
      },
      Response::Error { message } => anyhow::bail!("server error: {}", message),
      _ => anyhow::bail!("unexpected response from server"),
   }
}

fn spawn_daemon(path: &PathBuf) -> Result<()> {
   let exe = std::env::current_exe().context("failed to get current executable")?;

   Command::new(&exe)
      .arg("serve")
      .arg("--path")
      .arg(path)
      .stdin(std::process::Stdio::null())
      .stdout(std::process::Stdio::null())
      .stderr(std::process::Stdio::null())
      .spawn()
      .context("failed to spawn daemon")?;

   Ok(())
}

async fn perform_search(
   query: &str,
   path: &PathBuf,
   store_id: &str,
   max: usize,
   per_file: usize,
   rerank: bool,
) -> Result<Vec<SearchResult>> {
   let store = std::sync::Arc::new(crate::store::LanceStore::new()?);
   let embedder = std::sync::Arc::new(crate::embed::worker::EmbedWorker::new()?);

   let fs = crate::file::LocalFileSystem::new();
   let chunker = crate::chunker::treesitter::TreeSitterChunker::new();
   let sync_engine = crate::sync::SyncEngine::new(fs, chunker, embedder.clone(), store.clone());

   sync_engine
      .initial_sync(store_id, path, false, None)
      .await?;

   let engine = crate::search::SearchEngine::new(store, embedder);
   let response = engine
      .search(store_id, query, max, per_file, None, rerank)
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
            is_anchor:  r.is_anchor,
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
   const MAX_PREVIEW_LINES: usize = 12;

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

   let display_results: Vec<_> = results
      .iter()
      .filter(|r| !r.is_anchor.unwrap_or(false))
      .collect();

   for (i, result) in display_results.iter().enumerate() {
      let start_line = result.start_line.unwrap_or(1);
      let lines: Vec<&str> = result.content.lines().collect();
      let total_lines = lines.len();
      let show_all = content || total_lines <= MAX_PREVIEW_LINES;
      let display_lines = if show_all {
         total_lines
      } else {
         MAX_PREVIEW_LINES
      };
      let line_num_width = format!("{}", start_line + display_lines).len();

      if !plain {
         print!("{}", style(format!("{}) ", i + 1)).bold().cyan());
         print!("{}:{}", style(&result.path).green(), start_line);

         if scores {
            print!(" {}", style(format!("(score: {:.3})", result.score)).dim());
         }

         println!();

         for (j, line) in lines.iter().take(display_lines).enumerate() {
            let line_num = start_line + j;
            println!(
               "{:>width$} {} {}",
               style(line_num).dim(),
               style("|").dim(),
               line,
               width = line_num_width
            );
         }

         if !show_all {
            let remaining = total_lines - display_lines;
            println!(
               "{:>width$} {} {}",
               "",
               style("|").dim(),
               style(format!("... (+{} more lines)", remaining)).dim(),
               width = line_num_width
            );
         }
      } else {
         print!("{}) {}:{}", i + 1, result.path, start_line);

         if scores {
            print!(" (score: {:.3})", result.score);
         }

         println!();

         for (j, line) in lines.iter().take(display_lines).enumerate() {
            let line_num = start_line + j;
            println!("{:>width$} | {}", line_num, line, width = line_num_width);
         }

         if !show_all {
            let remaining = total_lines - display_lines;
            println!("{:>width$} | ... (+{} more lines)", "", remaining, width = line_num_width);
         }
      }

      println!();
   }

   Ok(())
}
