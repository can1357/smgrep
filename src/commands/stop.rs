use std::path::PathBuf;

use anyhow::{Context, Result};
use console::style;
use tokio::net::UnixStream;

use crate::ipc::{self, Request, Response};

pub async fn execute(path: Option<PathBuf>) -> Result<()> {
   let root = std::env::current_dir()?;
   let target_path = path.unwrap_or(root);

   let store_id = crate::git::resolve_store_id(&target_path)?;
   let socket_path = ipc::socket_path(&store_id);

   if !socket_path.exists() {
      println!("{}", style("No server running for this project").yellow());
      return Ok(());
   }

   let mut buffer = ipc::SocketBuffer::new();

   match UnixStream::connect(&socket_path).await {
      Ok(mut stream) => {
         buffer.send(&mut stream, &Request::Shutdown).await?;

         match buffer.recv(&mut stream).await {
            Ok(Response::Shutdown { success: true }) => {
               println!("{}", style("Server stopped").green());
            },
            Ok(_) => {
               println!("{}", style("Unexpected response from server").yellow());
            },
            Err(_) => {
               println!("{}", style("Server stopped").green());
            },
         }
      },
      Err(_) => {
         std::fs::remove_file(&socket_path).context("failed to remove stale socket")?;
         println!("{}", style("Removed stale socket").yellow());
      },
   }

   Ok(())
}
