use std::{
   fs,
   path::{Path, PathBuf},
   time::Duration,
};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{
   Result, config,
   error::ConfigError,
   grammar::{GRAMMAR_URLS, GrammarManager},
};

pub async fn execute() -> Result<()> {
   println!("{}\n", style("smgrep Setup").bold());

   let home = directories::UserDirs::new()
      .ok_or(ConfigError::GetUserDirectories)?
      .home_dir()
      .to_path_buf();

   let root = home.join(".smgrep");
   let models = root.join("models");
   let data = root.join("data");
   let grammars = root.join("grammars");

   fs::create_dir_all(&root)?;
   fs::create_dir_all(&models)?;
   fs::create_dir_all(&data)?;
   fs::create_dir_all(&grammars)?;

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
   println!("   {} {}", style("smgrep index").green(), style("# Index your repository").dim());
   println!(
      "   {} {}",
      style("smgrep \"search query\"").green(),
      style("# Search your code").dim()
   );
   println!("   {} {}", style("smgrep doctor").green(), style("# Check health status").dim());
   println!("\n{}", style("Note: Grammars are also downloaded automatically on first use.").dim());

   Ok(())
}

fn check_dir(name: &str, path: &Path) {
   let exists = path.exists();
   let symbol = if exists {
      style("✓").green()
   } else {
      style("✗").red()
   };
   println!("{} {}: {}", symbol, name, style(path.display()).dim());
}

async fn download_models(models_dir: &Path) -> Result<()> {
   let cfg = config::get();
   let models = [&cfg.dense_model, &cfg.colbert_model];
   for model_id in models {
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
      spinner.set_message(format!("Downloading {model_id}..."));

      match download_model_from_hf(model_id, &model_path).await {
         Ok(()) => {
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

async fn download_grammars(grammars_dir: &Path) -> Result<()> {
   let grammar_manager = GrammarManager::with_auto_download(false)?;

   for (lang, _url) in GRAMMAR_URLS {
      let grammar_path = grammars_dir.join(format!("tree-sitter-{lang}.wasm"));

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
      spinner.set_message(format!("Downloading {lang} grammar..."));

      match grammar_manager.download_grammar(lang).await {
         Ok(()) => {
            spinner.finish_with_message(format!(
               "{} Downloaded: {}",
               style("✓").green(),
               style(lang).dim()
            ));
         },
         Err(e) => {
            spinner.finish_with_message(format!("{} Failed: {} - {}", style("✗").red(), lang, e));
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
