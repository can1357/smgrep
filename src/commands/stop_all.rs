use anyhow::Result;
use console::style;
use tokio::net::UnixStream;

use crate::ipc::{self, Request, Response};

pub async fn execute() -> Result<()> {
   let socks_dir = ipc::socket_dir();

   if !socks_dir.exists() {
      println!("{}", style("No servers running").yellow());
      return Ok(());
   }

   let entries: Vec<_> = std::fs::read_dir(&socks_dir)?
      .filter_map(|e| e.ok())
      .filter(|e| e.path().extension().is_some_and(|ext| ext == "sock"))
      .collect();

   if entries.is_empty() {
      println!("{}", style("No servers running").yellow());
      return Ok(());
   }

   let mut stopped = 0;
   let mut failed = 0;

   for entry in entries {
      let socket_path = entry.path();

      match UnixStream::connect(&socket_path).await {
         Ok(mut stream) => {
            let mut buffer = ipc::SocketBuffer::new();
            if let Err(e) = buffer.send(&mut stream, &Request::Shutdown).await {
               tracing::debug!("Failed to send shutdown to {:?}: {}", socket_path, e);
               failed += 1;
               continue;
            }

            match buffer.recv(&mut stream).await {
               Ok(Response::Shutdown { success: true }) | Err(_) => stopped += 1,
               Ok(_) => failed += 1,
            }
         },
         Err(_) => {
            let _ = std::fs::remove_file(&socket_path);
            stopped += 1;
         },
      }
   }

   println!("{}", style(format!("Stopped {} servers, {} failed", stopped, failed)).green());

   Ok(())
}
