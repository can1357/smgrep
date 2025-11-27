use std::{
   fs,
   path::{Path, PathBuf},
   sync::OnceLock,
};

use directories::BaseDirs;
use figment::{
   Figment,
   providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
   pub dense_model:   String,
   pub colbert_model: String,
   pub dense_dim:     usize,
   pub colbert_dim:   usize,

   pub query_prefix:       String,
   pub dense_max_length:   usize,
   pub colbert_max_length: usize,
   pub default_batch_size: usize,
   pub max_batch_size:     usize,
   pub max_threads:        usize,

   pub port:                     u16,
   pub idle_timeout_secs:        u64,
   pub idle_check_interval_secs: u64,
   pub worker_timeout_ms:        u64,

   pub low_impact:      bool,
   pub disable_gpu:     bool,
   pub fast_mode:       bool,
   pub profile_enabled: bool,
   pub skip_meta_save:  bool,
   pub debug_models:    bool,
   pub debug_embed:     bool,
}

impl Default for Config {
   fn default() -> Self {
      Self {
         dense_model:              "ibm-granite/granite-embedding-30m-english".to_string(),
         colbert_model:            "answerdotai/answerai-colbert-small-v1".to_string(),
         dense_dim:                384,
         colbert_dim:              96,
         query_prefix:             String::new(),
         dense_max_length:         256,
         colbert_max_length:       256,
         default_batch_size:       48,
         max_batch_size:           96,
         max_threads:              32,
         port:                     4444,
         idle_timeout_secs:        30 * 60,
         idle_check_interval_secs: 60,
         worker_timeout_ms:        60000,
         low_impact:               false,
         disable_gpu:              false,
         fast_mode:                false,
         profile_enabled:          false,
         skip_meta_save:           false,
         debug_models:             false,
         debug_embed:              false,
      }
   }
}

impl Config {
   pub fn load() -> Self {
      let config_path = config_file_path();
      if !config_path.exists() {
         Self::create_default_config(&config_path);
      }

      Figment::from(Serialized::defaults(Self::default()))
         .merge(Toml::file(config_path))
         .merge(Env::prefixed("SMGREP_").lowercase(false))
         .extract()
         .unwrap_or_default()
   }

   fn create_default_config(path: &Path) {
      if let Some(parent) = path.parent() {
         let _ = fs::create_dir_all(parent);
      }
      let default_config = Self::default();
      if let Ok(toml) = toml::to_string_pretty(&default_config) {
         let _ = fs::write(path, toml);
      }
   }

   pub fn batch_size(&self) -> usize {
      self.default_batch_size.min(self.max_batch_size)
   }

   pub fn default_threads(&self) -> usize {
      (num_cpus::get().saturating_sub(4)).clamp(1, self.max_threads)
   }
}

pub fn config_file_path() -> PathBuf {
   data_dir().join("config.toml")
}

pub fn get() -> &'static Config {
   CONFIG.get_or_init(Config::load)
}

pub fn data_dir() -> PathBuf {
   BaseDirs::new()
      .expect("failed to locate base directories")
      .home_dir()
      .join(".smgrep")
}

pub fn model_dir() -> PathBuf {
   data_dir().join("models")
}

pub fn marketplace_dir() -> PathBuf {
   data_dir().join("marketplace")
}
