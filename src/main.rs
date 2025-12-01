use std::{path::PathBuf, sync::LazyLock};

use clap::{Parser, Subcommand};
use smgrep::{
   Result,
   cmd::{self, search::SearchOptions},
   version,
};
use tracing::Level;
use tracing_subscriber::EnvFilter;

static VERSION_STRING: LazyLock<String> = LazyLock::new(version::version_string);

fn version_string() -> &'static str {
   &VERSION_STRING
}

/// Command-line arguments for the smgrep application
#[derive(Parser)]
#[command(name = "smgrep")]
#[command(about = "Semantic code search tool")]
#[command(version = version_string())]
struct Cli {
   #[arg(long, env = "SMGREP_STORE")]
   store: Option<String>,

   #[command(subcommand)]
   command: Option<Cmd>,

   #[arg(trailing_var_arg = true)]
   query: Vec<String>,
}

/// Available subcommands for smgrep
#[derive(Subcommand)]
enum Cmd {
   #[command(about = "Search indexed code semantically")]
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

   #[command(about = "Index a directory for semantic search")]
   Index {
      #[arg(short = 'p', long, help = "Directory to index (default: cwd)")]
      path: Option<PathBuf>,

      #[arg(short = 'd', long, help = "Show what would be indexed")]
      dry_run: bool,

      #[arg(short = 'r', long, help = "Delete and re-index")]
      reset: bool,
   },

   #[command(about = "Start a background daemon for faster searches")]
   Serve {
      #[arg(long, help = "Directory to serve (default: cwd)")]
      path: Option<PathBuf>,
   },

   #[command(about = "Stop the daemon for a directory")]
   Stop {
      #[arg(long, help = "Directory of server to stop (default: cwd)")]
      path: Option<PathBuf>,
   },

   #[command(name = "stop-all", about = "Stop all running daemons")]
   StopAll,

   #[command(about = "Show status of running daemons")]
   Status,

   #[command(about = "Remove index data and metadata for a store")]
   Clean {
      #[arg(help = "Store ID to clean (default: current directory's store)")]
      store_id: Option<String>,

      #[arg(long, help = "Clean all stores")]
      all: bool,
   },

   #[command(about = "Download and configure embedding models")]
   Setup,

   #[command(about = "Check system configuration and dependencies")]
   Doctor,

   #[command(about = "List indexed files in a directory")]
   List,

   #[command(name = "claude-install", about = "Install smgrep as a Claude Code MCP server")]
   ClaudeInstall,

   #[command(name = "mcp", about = "Run as an MCP server (stdio transport)")]
   Mcp,
}

fn main() -> Result<()> {
   tracing_subscriber::fmt()
      .with_env_filter(EnvFilter::from_default_env().add_directive(Level::WARN.into()))
      .init();

   let cli = Cli::parse();

   // On macOS Apple Silicon with Metal, use single-threaded runtime for the serve
   // command. The candle Metal backend creates a command buffer at initialization
   // and enqueues it. In multi-threaded mode, a different worker thread does the
   // actual GPU work with its own buffer, leaving the initial buffer enqueued but
   // uncommitted. Metal command queues block waiting for enqueued buffers to be
   // committed in order, causing the GPU wait to hang indefinitely.
   #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
   let is_serve = matches!(&cli.command, Some(Cmd::Serve { .. }));
   #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
   let is_serve = false;

   if is_serve {
      tokio::runtime::Builder::new_current_thread()
         .enable_all()
         .build()
         .expect("failed to build tokio runtime")
         .block_on(run_command(cli))
   } else {
      tokio::runtime::Builder::new_multi_thread()
         .enable_all()
         .build()
         .expect("failed to build tokio runtime")
         .block_on(run_command(cli))
   }
}

async fn run_command(cli: Cli) -> Result<()> {
   if cli.command.is_none() && !cli.query.is_empty() {
      let query = cli.query.join(" ");
      return cmd::search::execute(query, None, 10, 1, SearchOptions::default(), cli.store).await;
   }

   match cli.command {
      Some(Cmd::Search {
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
         cmd::search::execute(
            query,
            path,
            max,
            per_file,
            SearchOptions { content, compact, scores, sync, dry_run, json, no_rerank, plain },
            cli.store,
         )
         .await
      },
      Some(Cmd::Index { path, dry_run, reset }) => {
         cmd::index::execute(path, dry_run, reset, cli.store).await
      },
      Some(Cmd::Serve { path }) => cmd::serve::execute(path, cli.store).await,
      Some(Cmd::Stop { path }) => cmd::stop::execute(path).await,
      Some(Cmd::StopAll) => cmd::stop_all::execute().await,
      Some(Cmd::Status) => cmd::status::execute().await,
      Some(Cmd::Clean { store_id, all }) => cmd::clean::execute(store_id, all),
      Some(Cmd::Setup) => cmd::setup::execute().await,
      Some(Cmd::Doctor) => cmd::doctor::execute(),
      Some(Cmd::List) => cmd::list::execute(),
      Some(Cmd::ClaudeInstall) => cmd::claude_install::execute(),
      Some(Cmd::Mcp) => cmd::mcp::execute().await,
      None => {
         eprintln!("No command or query provided. Use --help for usage information.");
         std::process::exit(1);
      },
   }
}
