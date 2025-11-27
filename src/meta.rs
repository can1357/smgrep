use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, config};

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FileMeta {
   pub hash:  String,
   pub mtime: u64,
}

#[derive(Serialize, Deserialize, Default)]
pub struct MetaStore {
   #[serde(default)]
   files:  HashMap<String, FileMeta>,
   #[serde(default)]
   hashes: HashMap<String, String>,
   #[serde(skip)]
   path:   PathBuf,
   #[serde(skip)]
   dirty:  bool,
}

impl MetaStore {
   pub fn load(store_id: &str) -> Result<Self> {
      let meta_dir = config::data_dir().join("meta");
      let path = meta_dir.join(format!("{}.json", store_id));

      let mut store = if path.exists() {
         let content = std::fs::read_to_string(&path)?;
         let mut store: MetaStore = serde_json::from_str(&content)?;
         store.path = path;
         store
      } else {
         MetaStore { files: HashMap::new(), hashes: HashMap::new(), path, dirty: false }
      };

      store.dirty = false;
      Ok(store)
   }

   pub fn get_hash(&self, path: &str) -> Option<&String> {
      self.files.get(path).map(|m| &m.hash).or_else(|| self.hashes.get(path))
   }

   pub fn get_mtime(&self, path: &str) -> Option<u64> {
      self.files.get(path).map(|m| m.mtime)
   }

   pub fn get_meta(&self, path: &str) -> Option<&FileMeta> {
      self.files.get(path)
   }

   pub fn set_hash(&mut self, path: String, hash: String) {
      if let Some(meta) = self.files.get_mut(&path) {
         meta.hash = hash;
      } else {
         self.files.insert(path.clone(), FileMeta { hash: hash.clone(), mtime: 0 });
         self.hashes.remove(&path);
      }
      self.dirty = true;
   }

   pub fn set_meta(&mut self, path: String, hash: String, mtime: u64) {
      self.files.insert(path.clone(), FileMeta { hash, mtime });
      self.hashes.remove(&path);
      self.dirty = true;
   }

   pub fn remove(&mut self, path: &str) {
      self.files.remove(path);
      self.hashes.remove(path);
      self.dirty = true;
   }

   pub fn save(&self) -> Result<()> {
      if let Some(parent) = self.path.parent() {
         std::fs::create_dir_all(parent)?;
      }

      let content = serde_json::to_string_pretty(&self)?;
      std::fs::write(&self.path, content)?;

      Ok(())
   }

   pub fn all_paths(&self) -> impl Iterator<Item = &String> {
      self.files.keys().chain(self.hashes.keys())
   }

   pub fn delete_by_prefix(&mut self, prefix: &str) {
      self.files.retain(|path, _| !path.starts_with(prefix));
      self.hashes.retain(|path, _| !path.starts_with(prefix));
      self.dirty = true;
   }
}

#[cfg(test)]
mod tests {
   use tempfile::TempDir;

   use super::*;

   #[test]
   fn load_nonexistent_creates_empty() {
      let temp_dir = TempDir::new().unwrap();
      unsafe {
         std::env::set_var("HOME", temp_dir.path());
      }

      let store = MetaStore::load("test_store").unwrap();
      assert_eq!(store.hashes.len(), 0);
   }

   #[test]
   fn set_and_get_hash() {
      let temp_dir = TempDir::new().unwrap();
      unsafe {
         std::env::set_var("HOME", temp_dir.path());
      }

      let mut store = MetaStore::load("test_store").unwrap();
      store.set_hash("/path/to/file".to_string(), "abc123".to_string());

      assert_eq!(store.get_hash("/path/to/file"), Some(&"abc123".to_string()));
      assert!(store.dirty);
   }

   #[test]
   fn save_and_load_roundtrip() {
      let temp_dir = TempDir::new().unwrap();
      unsafe {
         std::env::set_var("HOME", temp_dir.path());
      }

      let mut store = MetaStore::load("test_store").unwrap();
      store.set_hash("/file1".to_string(), "hash1".to_string());
      store.set_hash("/file2".to_string(), "hash2".to_string());
      store.save().unwrap();

      let loaded = MetaStore::load("test_store").unwrap();
      assert_eq!(loaded.get_hash("/file1"), Some(&"hash1".to_string()));
      assert_eq!(loaded.get_hash("/file2"), Some(&"hash2".to_string()));
   }

   #[test]
   fn remove_hash() {
      let temp_dir = TempDir::new().unwrap();
      unsafe {
         std::env::set_var("HOME", temp_dir.path());
      }

      let mut store = MetaStore::load("test_store").unwrap();
      store.set_hash("/file1".to_string(), "hash1".to_string());
      store.remove("/file1");

      assert_eq!(store.get_hash("/file1"), None);
   }

   #[test]
   fn all_paths_returns_keys() {
      let temp_dir = TempDir::new().unwrap();
      unsafe {
         std::env::set_var("HOME", temp_dir.path());
      }

      let mut store = MetaStore::load("test_store").unwrap();
      store.set_hash("/file1".to_string(), "hash1".to_string());
      store.set_hash("/file2".to_string(), "hash2".to_string());

      let paths: Vec<_> = store.all_paths().collect();
      assert_eq!(paths.len(), 2);
      assert!(paths.contains(&&"/file1".to_string()));
      assert!(paths.contains(&&"/file2".to_string()));
   }
}
