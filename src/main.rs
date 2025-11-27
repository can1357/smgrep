use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use smgrep::commands;

fn version_string() -> &'static str {
   const VERSION: &str = env!("CARGO_PKG_VERSION");
   const GIT_HASH: &str = env!("GIT_HASH");
   const GIT_TAG: &str = env!("GIT_TAG");
   const GIT_DIRTY: &str = env!("GIT_DIRTY");

   static VERSION_STRING: std::sync::OnceLock<String> = std::sync::OnceLock::new();
   VERSION_STRING.get_or_init(|| {
      let dirty = if GIT_DIRTY == "true" { "-dirty" } else { "" };
      if GIT_TAG.is_empty() {
         format!("{VERSION} ({GIT_HASH}{dirty})")
      } else {
         format!("{VERSION} ({GIT_TAG}, {GIT_HASH}{dirty})")
      }
   })
}

#[derive(Parser)]
#[command(name = "smgrep")]
#[command(about = "Semantic code search tool - Rust port of osgrep")]
#[command(version = version_string())]
struct Cli {
   #[arg(long, env = "SMGREP_STORE")]
   store: Option<String>,

   #[command(subcommand)]
   command: Option<Commands>,

   #[arg(trailing_var_arg = true)]
   query: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
   Search {
      #[arg(help = "Search query")]
      query: String,

      #[arg(help = "Directory to search (default: cwd)")]
      path: Option<PathBuf>,

      #[arg(
         short = 'm',
         long,
         alias = "max-count",
         default_value = "10",
         help = "Maximum total results"
      )]
      max: usize,

      #[arg(long, default_value = "1", help = "Maximum results per file")]
      per_file: usize,

      #[arg(short = 'c', long, help = "Show full content")]
      content: bool,

      #[arg(long, help = "Show file paths only (like grep -l)")]
      compact: bool,

      #[arg(long, help = "Show relevance scores")]
      scores: bool,

      #[arg(short = 's', long, help = "Force re-index before search")]
      sync: bool,

      #[arg(long, help = "Show what would be indexed")]
      dry_run: bool,

      #[arg(long, help = "JSON output")]
      json: bool,

      #[arg(long, help = "Skip ColBERT reranking")]
      no_rerank: bool,

      #[arg(long, help = "Disable ANSI colors and use simpler formatting")]
      plain: bool,
   },

   Index {
      #[arg(short = 'p', long, help = "Directory to index (default: cwd)")]
      path: Option<PathBuf>,

      #[arg(short = 'd', long, help = "Show what would be indexed")]
      dry_run: bool,

      #[arg(short = 'r', long, help = "Delete and re-index")]
      reset: bool,
   },

   Serve {
      #[arg(long, help = "Directory to serve (default: cwd)")]
      path: Option<PathBuf>,
   },

   Stop {
      #[arg(long, help = "Directory of server to stop (default: cwd)")]
      path: Option<PathBuf>,
   },

   #[command(name = "stop-all")]
   StopAll,

   Status,

   Setup,

   Doctor,

   List,

   #[command(name = "claude-install")]
   ClaudeInstall,

   #[command(name = "mcp")]
   Mcp,
}

#[tokio::main]
async fn main() -> Result<()> {
   tracing_subscriber::fmt()
      .with_env_filter(
         tracing_subscriber::EnvFilter::from_default_env()
            .add_directive(tracing::Level::WARN.into()),
      )
      .init();

   let cli = Cli::parse();

   if cli.command.is_none() && !cli.query.is_empty() {
      let query = cli.query.join(" ");
      return commands::search::execute(
         query, None, 10, 1, false, false, false, false, false, false, false, false, cli.store,
      )
      .await;
   }

   match cli.command {
      Some(Commands::Search {
         query,
         path,
         max,
         per_file,
         content,
         compact,
         scores,
         sync,
         dry_run,
         json,
         no_rerank,
         plain,
      }) => {
         commands::search::execute(
            query, path, max, per_file, content, compact, scores, sync, dry_run, json, no_rerank,
            plain, cli.store,
         )
         .await
      },
      Some(Commands::Index { path, dry_run, reset }) => {
         commands::index::execute(path, dry_run, reset, cli.store).await
      },
      Some(Commands::Serve { path }) => commands::serve::execute(path, cli.store).await,
      Some(Commands::Stop { path }) => commands::stop::execute(path).await,
      Some(Commands::StopAll) => commands::stop_all::execute().await,
      Some(Commands::Status) => commands::status::execute().await,
      Some(Commands::Setup) => commands::setup::execute().await,
      Some(Commands::Doctor) => commands::doctor::execute().await,
      Some(Commands::List) => commands::list::execute().await,
      Some(Commands::ClaudeInstall) => commands::claude_install::execute().await,
      Some(Commands::Mcp) => commands::mcp::execute().await,
      None => {
         eprintln!("No command or query provided. Use --help for usage information.");
         std::process::exit(1);
      },
   }
}
