use std::path::PathBuf;

use directories::BaseDirs;

pub const DENSE_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const COLBERT_MODEL: &str = "colbert-ir/colbertv2.0";

pub const DENSE_DIM: usize = 384;
pub const COLBERT_DIM: usize = 128;

pub const QUERY_PREFIX: &str = "";

pub const DEFAULT_BATCH_SIZE: usize = 48;
pub const MAX_BATCH_SIZE: usize = 96;

pub const MAX_THREADS: usize = 4;

pub fn default_threads() -> usize {
   (num_cpus::get() - 1).max(1).min(MAX_THREADS)
}

pub fn data_dir() -> PathBuf {
   BaseDirs::new()
      .expect("failed to locate base directories")
      .home_dir()
      .join(".rsgrep")
}

pub fn model_dir() -> PathBuf {
   data_dir().join("models")
}

pub fn port() -> u16 {
   std::env::var("RSGREP_PORT")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(4444)
}

pub fn threads() -> usize {
   std::env::var("RSGREP_THREADS")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or_else(default_threads)
      .min(MAX_THREADS)
}

pub fn batch_size() -> usize {
   std::env::var("RSGREP_BATCH_SIZE")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(DEFAULT_BATCH_SIZE)
      .min(MAX_BATCH_SIZE)
}

pub fn low_impact() -> bool {
   std::env::var("RSGREP_LOW_IMPACT")
      .ok()
      .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn worker_timeout_ms() -> u64 {
   std::env::var("RSGREP_WORKER_TIMEOUT_MS")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(60000)
}

pub fn fast_mode() -> bool {
   std::env::var("RSGREP_FAST")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn profile_enabled() -> bool {
   std::env::var("RSGREP_PROFILE")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn skip_meta_save() -> bool {
   std::env::var("RSGREP_SKIP_META_SAVE")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn debug_models() -> bool {
   std::env::var("RSGREP_DEBUG_MODELS")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}

pub fn debug_embed() -> bool {
   std::env::var("RSGREP_DEBUG_EMBED")
      .ok()
      .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
      .unwrap_or(false)
}
