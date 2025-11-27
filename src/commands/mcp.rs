use std::{
   io::{BufRead, Write},
   path::PathBuf,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::UnixStream;

use crate::ipc::{self, Request, Response, SocketBuffer};

#[derive(Deserialize)]
struct JsonRpcRequest {
   #[allow(dead_code)]
   jsonrpc: String,
   id:      Option<Value>,
   method:  String,
   #[serde(default)]
   params:  Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
   jsonrpc: &'static str,
   id:      Value,
   #[serde(skip_serializing_if = "Option::is_none")]
   result:  Option<Value>,
   #[serde(skip_serializing_if = "Option::is_none")]
   error:   Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
   code:    i32,
   message: String,
}

impl JsonRpcResponse {
   fn success(id: Value, result: Value) -> Self {
      Self { jsonrpc: "2.0", id, result: Some(result), error: None }
   }

   fn error(id: Value, code: i32, message: String) -> Self {
      Self { jsonrpc: "2.0", id, result: None, error: Some(JsonRpcError { code, message }) }
   }
}

struct DaemonConn {
   stream: UnixStream,
   buffer: SocketBuffer,
   cwd:    PathBuf,
}

impl DaemonConn {
   async fn connect(cwd: PathBuf) -> Result<Self> {
      let store_id = crate::git::resolve_store_id(&cwd)?;
      let socket_path = ipc::socket_path(&store_id);

      let stream = match UnixStream::connect(&socket_path).await {
         Ok(s) => s,
         Err(_) => {
            spawn_daemon(&cwd)?;
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            UnixStream::connect(&socket_path).await?
         }
      };

      Ok(Self { stream, buffer: SocketBuffer::new(), cwd })
   }

   async fn search(&mut self, query: &str, limit: usize) -> Result<String> {
      let request = Request::Search {
         query:  query.to_string(),
         limit,
         path:   Some(self.cwd.to_string_lossy().to_string()),
         rerank: true,
      };

      self.buffer.send(&mut self.stream, &request).await?;
      let response: Response = self.buffer.recv(&mut self.stream).await?;

      match response {
         Response::Search(search_response) => {
            let mut output = String::new();
            for r in search_response.results {
               output.push_str(&format!("{}:{}\n", r.path, r.start_line));
               for line in r.content.lines().take(10) {
                  output.push_str(&format!("  {}\n", line));
               }
               output.push('\n');
            }
            if output.is_empty() {
               output = format!("No results found for '{}'", query);
            }
            Ok(output)
         }
         Response::Error { message } => anyhow::bail!("Search failed: {}", message),
         _ => anyhow::bail!("Unexpected response"),
      }
   }
}

pub async fn execute() -> Result<()> {
   let stdin = std::io::stdin();
   let stdout = std::io::stdout();
   let mut stdout = stdout.lock();

   let cwd = std::env::current_dir()?;
   let mut conn: Option<DaemonConn> = None;

   for line in stdin.lock().lines() {
      let line = line.context("failed to read line")?;
      if line.is_empty() {
         continue;
      }

      let request: JsonRpcRequest = match serde_json::from_str(&line) {
         Ok(r) => r,
         Err(e) => {
            let response =
               JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {}", e));
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
            continue;
         }
      };

      let id = request.id.clone().unwrap_or(Value::Null);
      let response = handle_request(request, &cwd, &mut conn).await;
      let response = match response {
         Ok(result) => JsonRpcResponse::success(id, result),
         Err(e) => JsonRpcResponse::error(id, -32603, e.to_string()),
      };

      writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
      stdout.flush()?;
   }

   Ok(())
}

async fn handle_request(
   request: JsonRpcRequest,
   cwd: &PathBuf,
   conn: &mut Option<DaemonConn>,
) -> Result<Value> {
   match request.method.as_str() {
      "initialize" => Ok(json!({
         "protocolVersion": "2024-11-05",
         "capabilities": {
            "tools": {}
         },
         "serverInfo": {
            "name": "rsgrep",
            "version": env!("CARGO_PKG_VERSION")
         }
      })),

      "notifications/initialized" => Ok(Value::Null),

      "tools/list" => Ok(json!({
         "tools": [{
            "name": "sem_search",
            "description": "Semantic code search. Finds code by meaning, not just text matching. Use for questions like 'where is X implemented' or 'how does Y work'.",
            "inputSchema": {
               "type": "object",
               "properties": {
                  "query": {
                     "type": "string",
                     "description": "Natural language query describing what you're looking for"
                  },
                  "limit": {
                     "type": "integer",
                     "description": "Maximum number of results (default: 10)",
                     "default": 10
                  }
               },
               "required": ["query"]
            }
         }]
      })),

      "tools/call" => {
         let name = request.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
         let args = request.params.get("arguments").cloned().unwrap_or(json!({}));

         match name {
            "sem_search" => {
               let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
               let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

               let result = do_search_with_retry(cwd, conn, query, limit).await?;
               Ok(json!({
                  "content": [{
                     "type": "text",
                     "text": result
                  }]
               }))
            }
            _ => anyhow::bail!("Unknown tool: {}", name),
         }
      }

      _ => anyhow::bail!("Unknown method: {}", request.method),
   }
}

async fn do_search_with_retry(
   cwd: &PathBuf,
   conn: &mut Option<DaemonConn>,
   query: &str,
   limit: usize,
) -> Result<String> {
   // Ensure connection exists
   if conn.is_none() {
      *conn = Some(DaemonConn::connect(cwd.clone()).await?);
   }

   // Try search, reconnect on failure
   match conn.as_mut().unwrap().search(query, limit).await {
      Ok(result) => Ok(result),
      Err(_) => {
         // Connection failed, reconnect and retry once
         *conn = Some(DaemonConn::connect(cwd.clone()).await?);
         conn.as_mut().unwrap().search(query, limit).await
      }
   }
}

fn spawn_daemon(path: &PathBuf) -> Result<()> {
   let exe = std::env::current_exe()?;
   std::process::Command::new(&exe)
      .arg("serve")
      .arg("--path")
      .arg(path)
      .stdin(std::process::Stdio::null())
      .stdout(std::process::Stdio::null())
      .stderr(std::process::Stdio::null())
      .spawn()?;
   Ok(())
}
