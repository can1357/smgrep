use std::{fs, path::PathBuf, time::SystemTime};

use anyhow::{Context, Result};
use console::style;

pub async fn execute() -> Result<()> {
   let home = directories::UserDirs::new()
      .context("failed to get user directories")?
      .home_dir()
      .to_path_buf();

   let data_dir = home.join(".rsgrep").join("data");

   if !data_dir.exists() {
      println!("No stores found.");
      println!(
         "\nRun {} in a repository to create your first store.",
         style("rsgrep index").green()
      );
      return Ok(());
   }

   let mut stores = Vec::new();

   for entry in fs::read_dir(&data_dir)? {
      let entry = entry?;
      let path = entry.path();

      if path.is_dir()
         && let Some(name) = path.file_name().and_then(|n| n.to_str())
      {
         let metadata = fs::metadata(&path)?;
         let modified = metadata.modified()?;
         let size = get_dir_size(&path)?;

         stores.push(StoreInfo { name: name.to_string(), path, size, modified });
      }
   }

   if stores.is_empty() {
      println!("No stores found.");
      println!(
         "\nRun {} in a repository to create your first store.",
         style("rsgrep index").green()
      );
      return Ok(());
   }

   stores.sort_by(|a, b| b.modified.cmp(&a.modified));

   println!(
      "\n{} {}",
      style(format!("Found {} store(s):", stores.len())).bold(),
      style("(in ~/.rsgrep/data)").dim()
   );
   println!();

   for store in stores {
      println!("  {}", style(&store.name).green().bold());
      println!(
         "    Size: {} • Modified: {}",
         style(format_size(store.size)).dim(),
         style(format_time_ago(store.modified)).dim()
      );
      println!();
   }

   println!("{}", style("To clean up a store: rm -rf ~/.rsgrep/data/<store-name>").dim());
   println!("{}", style("To use a specific store: rsgrep --store <store-name> <query>").dim());

   Ok(())
}

struct StoreInfo {
   name:     String,
   path:     PathBuf,
   size:     u64,
   modified: SystemTime,
}

fn get_dir_size(path: &PathBuf) -> Result<u64> {
   let mut total = 0u64;

   if path.is_dir() {
      for entry in fs::read_dir(path)? {
         let entry = entry?;
         let metadata = entry.metadata()?;

         if metadata.is_dir() {
            total += get_dir_size(&entry.path())?;
         } else {
            total += metadata.len();
         }
      }
   }

   Ok(total)
}

fn format_size(bytes: u64) -> String {
   const KB: u64 = 1024;
   const MB: u64 = KB * 1024;
   const GB: u64 = MB * 1024;

   if bytes < KB {
      format!("{} B", bytes)
   } else if bytes < MB {
      format!("{:.1} KB", bytes as f64 / KB as f64)
   } else if bytes < GB {
      format!("{:.1} MB", bytes as f64 / MB as f64)
   } else {
      format!("{:.1} GB", bytes as f64 / GB as f64)
   }
}

fn format_time_ago(time: SystemTime) -> String {
   let now = SystemTime::now();
   let duration = now.duration_since(time).unwrap_or_default();

   let seconds = duration.as_secs();
   let minutes = seconds / 60;
   let hours = minutes / 60;
   let days = hours / 24;

   if days > 0 {
      format!("{}d ago", days)
   } else if hours > 0 {
      format!("{}h ago", hours)
   } else if minutes > 0 {
      format!("{}m ago", minutes)
   } else {
      "just now".to_string()
   }
}
