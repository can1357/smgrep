pub mod chunker;
pub mod commands;
pub mod config;
pub mod embed;
pub mod error;
pub mod file;
pub mod format;
pub mod git;
pub mod meta;
pub mod search;
pub mod store;
pub mod sync;
pub mod types;

pub use error::{Result, RsgrepError};
pub use types::*;
