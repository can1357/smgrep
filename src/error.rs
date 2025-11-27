use thiserror::Error;

#[derive(Debug, Error)]
pub enum RsgrepError {
   #[error("io error")]
   Io(#[from] std::io::Error),

   #[error("store error: {0}")]
   Store(String),

   #[error("embedding error: {0}")]
   Embedding(String),

   #[error("chunker error: {0}")]
   Chunker(String),

   #[error("git error: {0}")]
   Git(String),

   #[error("config error: {0}")]
   Config(String),

   #[error("http error: {0}")]
   Http(String),

   #[error("serialization error")]
   Serialization(#[from] serde_json::Error),

   #[error(transparent)]
   Other(#[from] anyhow::Error),

   #[error("lancedb error: {0}")]
   LanceDb(String),
}

pub type Result<T, E = RsgrepError> = std::result::Result<T, E>;
