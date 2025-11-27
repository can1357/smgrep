use rsgrep::{
   format::{OutputMode, create_formatter, detect_output_mode},
   types::{ChunkType, SearchResult},
};

fn main() {
   let results = vec![
      SearchResult {
         path:       "src/auth.rs".to_string(),
         content:    "pub fn authenticate_user(credentials: &Credentials) -> Result<Token> {\n    \
                      let token = jwt::sign(credentials, &SECRET)?;\n    Ok(Token { value: token, \
                      expires: now() + TTL })\n}"
            .to_string(),
         score:      0.95,
         start_line: 41,
         num_lines:  4,
         chunk_type: ChunkType::Function,
         is_anchor:  false,
      },
      SearchResult {
         path:       "src/handlers/login.rs".to_string(),
         content:    "async fn handle_login(req: Request) -> Result<Response> {\n    let body = \
                      req.json::<LoginRequest>().await?;\n    let token = \
                      authenticate_user(&body.credentials)?;\n    Ok(Response::json(token))\n}"
            .to_string(),
         score:      0.87,
         start_line: 14,
         num_lines:  5,
         chunk_type: ChunkType::Function,
         is_anchor:  false,
      },
      SearchResult {
         path:       "tests/auth_test.rs".to_string(),
         content:    "#[test]\nfn test_authenticate_valid_credentials() {\n    let creds = \
                      Credentials::new(\"user\", \"pass\");\n    let result = \
                      authenticate_user(&creds);\n    assert!(result.is_ok());\n}"
            .to_string(),
         score:      0.72,
         start_line: 10,
         num_lines:  6,
         chunk_type: ChunkType::Function,
         is_anchor:  false,
      },
   ];

   println!("=== Agent Mode (pipe output) ===");
   let agent_formatter = create_formatter(OutputMode::Agent);
   println!("{}\n", agent_formatter.format(&results, false, false));

   println!("\n=== Compact Mode (file paths only) ===");
   let compact_formatter = create_formatter(OutputMode::Compact);
   println!("{}\n", compact_formatter.format(&results, false, false));

   println!("\n=== JSON Mode ===");
   let json_formatter = create_formatter(OutputMode::Json);
   println!("{}\n", json_formatter.format(&results, false, false));

   println!("\n=== Human Mode (with syntax highlighting) ===");
   let human_formatter = create_formatter(OutputMode::Human);
   println!("{}\n", human_formatter.format(&results, true, false));

   println!("\n=== Auto-detect mode ===");
   let mode = detect_output_mode(false, false);
   println!("Detected mode: {:?}", mode);
}
