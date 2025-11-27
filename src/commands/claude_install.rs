use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use console::style;

fn find_rsgrep_root() -> Result<PathBuf> {
   let exe = std::env::current_exe().context("failed to get current executable path")?;
   let exe_dir = exe.parent().context("executable has no parent directory")?;

   let candidates = [
      exe_dir.to_path_buf(),
      exe_dir.join(".."),
      exe_dir.join("../.."),
      exe_dir.join("../share/rsgrep"),
   ];

   for candidate in &candidates {
      let marketplace = candidate.join(".claude-plugin/marketplace.json");
      if marketplace.exists() {
         return Ok(candidate.canonicalize().context("failed to canonicalize path")?);
      }
   }

   bail!(
      "rsgrep root directory not found (looking for .claude-plugin/marketplace.json). searched:\n{}",
      candidates
         .iter()
         .map(|p| format!("  • {}", p.display()))
         .collect::<Vec<_>>()
         .join("\n")
   )
}

fn run_claude_command(args: &[&str]) -> Result<()> {
   let status = Command::new("claude")
      .args(args)
      .stdin(Stdio::inherit())
      .stdout(Stdio::inherit())
      .stderr(Stdio::inherit())
      .status()
      .context("failed to execute claude command")?;

   if !status.success() {
      bail!("claude exited with code {}", status.code().unwrap_or(-1));
   }

   Ok(())
}

pub async fn execute() -> Result<()> {
   println!(
      "{}",
      style("Installing rsgrep plugin for Claude Code...").cyan().bold()
   );

   let rsgrep_root = find_rsgrep_root()?;
   let root_path = rsgrep_root.to_string_lossy();

   println!("rsgrep root: {}", style(&root_path).dim());

   println!("{}", style("Adding local marketplace...").dim());
   match run_claude_command(&["plugin", "marketplace", "add", &root_path]) {
      Ok(()) => println!("{}", style("✓ Added rsgrep marketplace").green()),
      Err(e) => {
         eprintln!("{}", style(format!("✗ Failed to add marketplace: {}", e)).red());
         print_troubleshooting();
         return Err(e);
      },
   }

   println!("{}", style("Installing plugin...").dim());
   match run_claude_command(&["plugin", "install", "rsgrep"]) {
      Ok(()) => println!("{}", style("✓ Installed rsgrep plugin").green()),
      Err(e) => {
         eprintln!("{}", style(format!("✗ Failed to install plugin: {}", e)).red());
         print_troubleshooting();
         return Err(e);
      },
   }

   println!();
   println!("{}", style("Next steps:").bold());
   println!("  1. Restart Claude Code if it's running");
   println!("  2. The plugin will automatically index your project when you open it");
   println!("  3. Claude will use rsgrep for semantic code search automatically");
   println!("  4. You can also use `rsgrep` commands directly in your terminal");

   Ok(())
}

fn print_troubleshooting() {
   eprintln!();
   eprintln!("{}", style("Troubleshooting:").yellow().bold());
   eprintln!("  • Ensure you have Claude Code installed");
   eprintln!("  • Try running: claude --version");
   eprintln!("  • Check that rsgrep was installed correctly with plugin files");
}
