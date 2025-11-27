use std::{path::PathBuf, sync::LazyLock};

use clap::{Parser, Subcommand};
use smgrep::{
   Result,
   cmd::{self, search::SearchOptions},
};
use tracing::Level;
use tracing_subscriber::EnvFilter;

static VERSION_STRING: LazyLock<String> = LazyLock::new(|| {
   const VERSION: &str = env!("CARGO_PKG_VERSION");
   const GIT_HASH: &str = env!("GIT_HASH");
   const GIT_TAG: &str = env!("GIT_TAG");
   const GIT_DIRTY: &str = env!("GIT_DIRTY");

   let dirty = if GIT_DIRTY == "true" { "-dirty" } else { "" };
   if GIT_TAG.is_empty() {
      format!("{VERSION} ({GIT_HASH}{dirty})")
   } else {
      format!("{VERSION} ({GIT_TAG}, {GIT_HASH}{dirty})")
   }
});

fn version_string() -> &'static str {
   &VERSION_STRING
}

#[derive(Parser)]
#[command(name = "smgrep")]
#[command(about = "Semantic code search tool - Rust port of osgrep")]
#[command(version = version_string())]
struct Cli {
   #[arg(long, env = "SMGREP_STORE")]
   store: Option<String>,

   #[command(subcommand)]
   command: Option<Cmd>,

   #[arg(trailing_var_arg = true)]
   query: Vec<String>,
}

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

#[tokio::main]
async fn main() -> Result<()> {
   tracing_subscriber::fmt()
      .with_env_filter(EnvFilter::from_default_env().add_directive(Level::WARN.into()))
      .init();

   let cli = Cli::parse();

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
