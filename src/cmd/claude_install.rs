use std::{
   fs,
   io::Cursor,
   path::PathBuf,
   process::{Command, Stdio},
};

use console::style;

use crate::{Result, config, error::Error};

const PLUGIN_BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/plugin-bundle.tar"));

fn extract_plugin_bundle(dest: &PathBuf) -> Result<()> {
   fs::create_dir_all(dest)?;
   let mut archive = tar::Archive::new(Cursor::new(PLUGIN_BUNDLE));
   archive.unpack(dest)?;
   Ok(())
}

fn run_claude_command(args: &[&str]) -> Result<()> {
   let status = Command::new("claude")
      .args(args)
      .stdin(Stdio::inherit())
      .stdout(Stdio::inherit())
      .stderr(Stdio::inherit())
      .status()
      .map_err(Error::ClaudeSpawn)?;

   if !status.success() {
      return Err(Error::ClaudeCommand(status.code().unwrap_or(-1)));
   }

   Ok(())
}

pub fn execute() -> Result<()> {
   println!(
      "{}",
      style("Installing smgrep plugin for Claude Code...")
         .cyan()
         .bold()
   );

   let marketplace_dir = config::marketplace_dir();
   println!("Marketplace: {}", style(marketplace_dir.display()).dim());

   println!("{}", style("Extracting plugin files...").dim());
   extract_plugin_bundle(&marketplace_dir)?;
   println!("{}", style("✓ Plugin files extracted").green());

   let marketplace_path = marketplace_dir.to_string_lossy();

   println!("{}", style("Adding marketplace...").dim());
   match run_claude_command(&["plugin", "marketplace", "add", &marketplace_path]) {
      Ok(()) => println!("{}", style("✓ Added smgrep marketplace").green()),
      Err(e) => {
         eprintln!("{}", style(format!("✗ Failed to add marketplace: {e}")).red());
         print_troubleshooting();
         return Err(e);
      },
   }

   println!("{}", style("Installing plugin...").dim());
   match run_claude_command(&["plugin", "install", "smgrep@smgrep"]) {
      Ok(()) => println!("{}", style("✓ Installed smgrep plugin").green()),
      Err(e) => {
         eprintln!("{}", style(format!("✗ Failed to install plugin: {e}")).red());
         print_troubleshooting();
         return Err(e);
      },
   }

   println!();
   println!("{}", style("Next steps:").bold());
   println!("  1. Restart Claude Code if it's running");
   println!("  2. The plugin will automatically index your project when you open it");
   println!("  3. Claude will use smgrep for semantic code search automatically");

   Ok(())
}

fn print_troubleshooting() {
   eprintln!();
   eprintln!("{}", style("Troubleshooting:").yellow().bold());
   eprintln!("  • Ensure you have Claude Code installed");
   eprintln!("  • Try running: claude --version");
}
