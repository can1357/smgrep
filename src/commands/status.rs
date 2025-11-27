use anyhow::Result;
use console::style;
use tokio::net::UnixStream;

use crate::ipc::{self, Request, Response};

pub async fn execute() -> Result<()> {
   let socks_dir = ipc::socket_dir();

   if !socks_dir.exists() {
      println!("{}", style("No servers running").dim());
      return Ok(());
   }

   let entries: Vec<_> = std::fs::read_dir(&socks_dir)?
      .filter_map(|e| e.ok())
      .filter(|e| e.path().extension().is_some_and(|ext| ext == "sock"))
      .collect();

   if entries.is_empty() {
      println!("{}", style("No servers running").dim());
      return Ok(());
   }

   println!("{}", style("Running servers:").bold());
   println!();

   let mut buffer = ipc::SocketBuffer::new();
   for entry in entries {
      let socket_path = entry.path();
      let store_id = socket_path
         .file_stem()
         .and_then(|s| s.to_str())
         .unwrap_or("unknown");

      match UnixStream::connect(&socket_path).await {
         Ok(mut stream) => {
            if let Err(_) = buffer.send(&mut stream, &Request::Health).await {
               println!("  {} {} {}", style("●").yellow(), store_id, style("(unresponsive)").dim());
               continue;
            }

            match buffer.recv(&mut stream).await {
               Ok(Response::Health { status }) => {
                  let state = if status.indexing {
                     format!("indexing {}%", status.progress)
                  } else {
                     "ready".to_string()
                  };
                  println!(
                     "  {} {} {}",
                     style("●").green(),
                     store_id,
                     style(format!("({})", state)).dim()
                  );
               },
               _ => {
                  println!("  {} {} {}", style("●").yellow(), store_id, style("(unknown)").dim());
               },
            }
         },
         Err(_) => {
            println!("  {} {} {}", style("●").red(), store_id, style("(stale)").dim());
         },
      }
   }

   Ok(())
}
