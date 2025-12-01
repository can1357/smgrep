<div align="center">
  <a href="https://github.com/can1357/smgrep">
    <img src="assets/logo.png" alt="smgrep" width="128" height="128" />
  </a>
  <h1>smgrep</h1>
  <p><em>Semantic code search, GPU-accelerated.</em></p>
  <a href="https://crates.io/crates/smgrep"><img src="https://img.shields.io/crates/v/smgrep.svg" alt="Crates.io" /></a>
  <a href="https://crates.io/crates/smgrep"><img src="https://img.shields.io/crates/d/smgrep.svg" alt="Downloads" /></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
</div>

Natural-language search that works like `grep`. Fast, local, GPU-accelerated, and built for coding agents.

- **Semantic:** Finds concepts ("where do transactions get created?"), not just strings.
- **GPU-Accelerated:** CUDA on NVIDIA GPUs, Metal on Apple Silicon.
- **Local & Private:** 100% local embeddings. No API keys required.
- **Auto-Isolated:** Each repository gets its own index automatically.
- **On-Demand Grammars:** Tree-sitter WASM grammars download automatically as needed.
- **Agent-Ready:** Native MCP server and Claude Code integration.

## Quick Start

1. **Install**

   ```bash
   cargo install smgrep
   ```

   Or build from source:

   ```bash
   git clone https://github.com/can1357/smgrep
   cd smgrep
   cargo build --release
   ```

   For CPU-only builds (no CUDA):

   ```bash
   cargo build --release --no-default-features
   ```

2. **Setup (Recommended)**

   ```bash
   smgrep setup
   ```

   Downloads embedding models (~500MB) and tree-sitter grammars upfront. If you skip this, models download automatically on first use.

3. **Search**

   ```bash
   cd my-repo
   smgrep "where do we handle authentication?"
   ```

   **Your first search will automatically index the repository.** Each repository is automatically isolated with its own index. Switching between repos "just works".

## Coding Agent Integration

### Claude Code

1. Run `smgrep claude-install`
2. Open Claude Code (`claude`) and ask questions about your codebase.
3. The plugin auto-starts the `smgrep serve` daemon and provides semantic search.

### MCP Server

smgrep includes a built-in MCP (Model Context Protocol) server:

```bash
smgrep mcp
```

This exposes a `sem_search` tool that agents can use for semantic code search. The server auto-starts the background daemon if needed.

## Commands

### `smgrep [query]`

The default command. Searches the current directory using semantic meaning.

```bash
smgrep "how is the database connection pooled?"
```

**Options:**
| Flag | Description | Default |
| --- | --- | --- |
| `-m <n>` | Max total results to return | `10` |
| `--per-file <n>` | Max matches per file | `1` |
| `-c`, `--content` | Show full chunk content | `false` |
| `--compact` | Show file paths only | `false` |
| `--scores` | Show relevance scores | `false` |
| `-s`, `--sync` | Force re-index before search | `false` |
| `--dry-run` | Show what would be indexed | `false` |
| `--json` | JSON output format | `false` |
| `--no-rerank` | Skip ColBERT reranking | `false` |
| `--plain` | Disable ANSI colors | `false` |

**Examples:**

```bash
# General concept search
smgrep "API rate limiting logic"

# Deep dive (more matches per file)
smgrep "error handling" --per-file 5

# Just the file paths
smgrep "user validation" --compact

# JSON for scripting
smgrep "config parsing" --json
```

### `smgrep index`

Manually indexes the repository.

```bash
smgrep index              # Index current dir
smgrep index --dry-run    # See what would be indexed
smgrep index --reset      # Delete and re-index from scratch
```

### `smgrep serve`

Runs a background daemon with file watching for instant searches.

- Keeps LanceDB and embedding models resident for fast responses
- Watches the repo and incrementally re-indexes on change
- Communicates via Unix socket (or TCP on Windows)

```bash
smgrep serve              # Start daemon for current repo
smgrep serve --path /repo # Start for specific path
```

### `smgrep stop` / `smgrep stop-all`

Stop running daemons.

```bash
smgrep stop               # Stop daemon for current repo
smgrep stop-all           # Stop all smgrep daemons
```

### `smgrep clean`

Remove index data and metadata for a store.

```bash
smgrep clean              # Clean current directory's store
smgrep clean my-store     # Clean specific store by ID
smgrep clean --all        # Clean all stores
```

### `smgrep status`

Show status of running daemons.

### `smgrep list`

Lists all indexed repositories and their metadata.

### `smgrep doctor`

Checks installation health, model availability, and grammar status.

```bash
smgrep doctor
```

## GPU Acceleration

smgrep uses [candle](https://github.com/huggingface/candle) for ML inference with optional CUDA support.

**With CUDA (default):**

Requires CUDA toolkit installed with environment configured:

```bash
export CUDA_ROOT=/usr/local/cuda  # or your CUDA installation path
export PATH="$CUDA_ROOT/bin:$PATH"

cargo build --release
```

Embedding speed is significantly faster on NVIDIA GPUs.

**CPU-only:**

```bash
cargo build --release --no-default-features
```

**Environment variables:**

- `SMGREP_DISABLE_GPU=1` - Force CPU even when CUDA is available
- `SMGREP_BATCH_SIZE=N` - Override batch size (auto-adapts on OOM)

## Architecture

smgrep combines several techniques for high-quality semantic search:

1. **Smart Chunking:** Tree-sitter parses code by function/class boundaries, ensuring embeddings capture complete logical blocks. Grammars download on-demand as WASM modules.

2. **Hybrid Search:** Dense embeddings (sentence-transformers) for broad recall, ColBERT reranking for precision.

3. **Quantized Storage:** ColBERT embeddings are quantized to int8 for efficient storage in LanceDB.

4. **Automatic Repository Isolation:** Stores are named by git remote URL or directory hash.

5. **Incremental Indexing:** File watcher detects changes and updates only affected chunks.

**Supported languages (37):** TypeScript, TSX, JavaScript, Python, Go, Rust, C, C++, C#, Java, Kotlin, Scala, Ruby, PHP, Elixir, Haskell, OCaml, Julia, Zig, Lua, Odin, Objective-C, Verilog, HTML, CSS, XML, Markdown, JSON, YAML, TOML, Bash, Make, Starlark, HCL, Terraform, Diff, Regex

## Configuration

smgrep uses a TOML config file at `~/.smgrep/config.toml`. All options can also be set via environment variables with the `SMGREP_` prefix.

### Config File

```toml
# ~/.smgrep/config.toml

# ============================================================================
# Models
# ============================================================================

# Dense embedding model (HuggingFace model ID)
# Used for initial semantic similarity search
dense_model = "ibm-granite/granite-embedding-small-english-r2"

# ColBERT reranking model (HuggingFace model ID)
# Used for precise reranking of search results
colbert_model = "answerdotai/answerai-colbert-small-v1"

# Model dimensions (must match the models above)
dense_dim = 384
colbert_dim = 96

# Query prefix (some models require a prefix like "query: ")
query_prefix = ""

# Maximum sequence lengths for tokenization
dense_max_length = 256
colbert_max_length = 256

# ============================================================================
# Performance
# ============================================================================

# Batch size for embedding computation
# Higher = faster but more memory. Auto-reduces on OOM.
default_batch_size = 48
max_batch_size = 96

# Maximum threads for parallel processing
max_threads = 32

# Force CPU inference even when CUDA is available
disable_gpu = false

# Low-impact mode: reduces resource usage for background indexing
low_impact = false

# Fast mode: skip ColBERT reranking for quicker (but less precise) results
fast_mode = false

# ============================================================================
# Server
# ============================================================================

# TCP port for daemon communication
port = 4444

# Idle timeout: shutdown daemon after this many seconds of inactivity
idle_timeout_secs = 1800  # 30 minutes

# How often to check for idle timeout
idle_check_interval_secs = 60

# Timeout for embedding worker operations (milliseconds)
worker_timeout_ms = 60000

# ============================================================================
# Debug
# ============================================================================

# Enable model loading debug output
debug_models = false

# Enable embedding debug output
debug_embed = false

# Enable profiling
profile_enabled = false

# Skip saving metadata (for testing)
skip_meta_save = false
```

### Environment Variables

Any config option can be set via environment variable with the `SMGREP_` prefix:

```bash
# Examples
export SMGREP_DISABLE_GPU=true
export SMGREP_DEFAULT_BATCH_SIZE=24
export SMGREP_IDLE_TIMEOUT_SECS=3600
```

| Variable                    | Description           | Default       |
| --------------------------- | --------------------- | ------------- |
| `SMGREP_STORE`              | Override store name   | auto-detected |
| `SMGREP_DISABLE_GPU`        | Force CPU inference   | `false`       |
| `SMGREP_DEFAULT_BATCH_SIZE` | Embedding batch size  | `48`          |
| `SMGREP_LOW_IMPACT`         | Reduce resource usage | `false`       |
| `SMGREP_FAST_MODE`          | Skip reranking        | `false`       |

### Ignoring Files

smgrep respects `.gitignore` and `.smignore` files.

Create `.smignore` in your repository root:

```
# Ignore generated files
dist/
*.min.js

# Ignore test fixtures
test/fixtures/
```

### Manual Store Management

- **View all stores:** `smgrep list`
- **Override auto-detection:** `smgrep --store custom-name "query"`
- **Data location:** `~/.smgrep/`

## Troubleshooting

- **Index feels stale?** Run `smgrep index` to refresh.
- **Weird results?** Run `smgrep doctor` to verify models and grammars.
- **Need a fresh start?** `smgrep index --reset` or delete `~/.smgrep/`.
- **GPU OOM?** Batch size auto-reduces, or set `SMGREP_DISABLE_GPU=1`.

## Building from Source

```bash
git clone https://github.com/can1357/smgrep
cd smgrep
cargo build --release

# Run tests
cargo test
```

## Acknowledgments

smgrep is inspired by [osgrep](https://github.com/Ryandonofrio3/osgrep) and [mgrep](https://github.com/mixedbread-ai/mgrep) by MixedBread.

## License

Licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.
