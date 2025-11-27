use std::{fs, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{
   config::{COLBERT_MODEL, DENSE_MODEL},
   grammar::{GRAMMAR_URLS, GrammarManager},
};

pub async fn execute() -> Result<()> {
   println!("{}\n", style("rsgrep Setup").bold());

   let home = directories::UserDirs::new()
      .context("failed to get user directories")?
      .home_dir()
      .to_path_buf();

   let root = home.join(".rsgrep");
   let models = root.join("models");
   let data = root.join("data");
   let grammars = root.join("grammars");

   fs::create_dir_all(&root).context("failed to create .rsgrep directory")?;
   fs::create_dir_all(&models).context("failed to create models directory")?;
   fs::create_dir_all(&data).context("failed to create data directory")?;
   fs::create_dir_all(&grammars).context("failed to create grammars directory")?;

   println!("{}", style("Checking directories...").dim());
   check_dir("Root", &root);
   check_dir("Models", &models);
   check_dir("Data (Vector DB)", &data);
   check_dir("Grammars", &grammars);
   println!();

   println!("{}", style("Downloading models...").bold());
   download_models(&models).await?;
   println!();

   println!("{}", style("Downloading grammars...").bold());
   download_grammars(&grammars).await?;

   println!("\n{}", style("Setup Complete!").green().bold());
   println!("\n{}", style("You can now run:").dim());
   println!("   {} {}", style("rsgrep index").green(), style("# Index your repository").dim());
   println!(
      "   {} {}",
      style("rsgrep \"search query\"").green(),
      style("# Search your code").dim()
   );
   println!("   {} {}", style("rsgrep doctor").green(), style("# Check health status").dim());
   println!(
      "\n{}",
      style("Note: Grammars are also downloaded automatically on first use.").dim()
   );

   Ok(())
}

fn check_dir(name: &str, path: &PathBuf) {
   let exists = path.exists();
   let symbol = if exists {
      style("✓").green()
   } else {
      style("✗").red()
   };
   println!("{} {}: {}", symbol, name, style(path.display()).dim());
}

async fn download_models(models_dir: &PathBuf) -> Result<()> {
   for model_id in [DENSE_MODEL, COLBERT_MODEL] {
      let model_path = models_dir.join(model_id.replace('/', "--"));

      if model_path.exists() {
         println!("{} Model: {}", style("✓").green(), style(model_id).dim());
         continue;
      }

      let spinner = ProgressBar::new_spinner();
      spinner.set_style(
         ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
      );
      spinner.enable_steady_tick(Duration::from_millis(100));
      spinner.set_message(format!("Downloading {}...", model_id));

      match download_model_from_hf(model_id, &model_path).await {
         Ok(_) => {
            spinner.finish_with_message(format!(
               "{} Downloaded: {}",
               style("✓").green(),
               style(model_id).dim()
            ));
         },
         Err(e) => {
            spinner.finish_with_message(format!(
               "{} Failed: {} - {}",
               style("✗").red(),
               model_id,
               e
            ));
         },
      }
   }

   Ok(())
}

async fn download_grammars(grammars_dir: &PathBuf) -> Result<()> {
   let grammar_manager = GrammarManager::with_auto_download(false)?;

   for (lang, _url) in GRAMMAR_URLS {
      let grammar_path = grammars_dir.join(format!("tree-sitter-{}.wasm", lang));

      if grammar_path.exists() {
         println!("{} Grammar: {}", style("✓").green(), style(lang).dim());
         continue;
      }

      let spinner = ProgressBar::new_spinner();
      spinner.set_style(
         ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
      );
      spinner.enable_steady_tick(Duration::from_millis(100));
      spinner.set_message(format!("Downloading {} grammar...", lang));

      match grammar_manager.download_grammar(lang).await {
         Ok(_) => {
            spinner.finish_with_message(format!(
               "{} Downloaded: {}",
               style("✓").green(),
               style(lang).dim()
            ));
         },
         Err(e) => {
            spinner.finish_with_message(format!(
               "{} Failed: {} - {}",
               style("✗").red(),
               lang,
               e
            ));
         },
      }
   }

   Ok(())
}

async fn download_model_from_hf(model_id: &str, dest: &PathBuf) -> Result<()> {
   fs::create_dir_all(dest)?;

   let api = hf_hub::api::tokio::Api::new()?;
   let repo = api.model(model_id.to_string());

   let files_to_download =
      vec!["config.json", "tokenizer.json", "tokenizer_config.json", "model.onnx"];

   for file in files_to_download {
      match repo.get(file).await {
         Ok(path) => {
            let dest_file = dest.join(file);
            if let Some(parent) = dest_file.parent() {
               fs::create_dir_all(parent)?;
            }
            fs::copy(path, dest_file)?;
         },
         Err(_e) => {},
      }
   }

   Ok(())
}
