use std::{
   collections::HashMap,
   fmt, fs,
   path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, config};

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FileMeta {
   pub hash:  FileHash,
   pub mtime: u64,
}

#[derive(Serialize, Deserialize, Copy, Clone, Default, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct FileHash([u8; 32]);

impl FileHash {
   pub fn from_slice(slice: &[u8]) -> Option<Self> {
      let (this, rem) = slice.split_first_chunk()?;
      rem.is_empty().then_some(Self(*this))
   }

   pub const fn new(hash: [u8; 32]) -> Self {
      Self(hash)
   }

   pub fn sum(dat: impl AsRef<[u8]>) -> Self {
      Self(Sha256::digest(dat.as_ref()).into())
   }
}

impl AsRef<[u8]> for FileHash {
   fn as_ref(&self) -> &[u8] {
      &self.0
   }
}

impl AsMut<[u8]> for FileHash {
   fn as_mut(&mut self) -> &mut [u8] {
      &mut self.0
   }
}

impl fmt::Display for FileHash {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "{}", hex::encode(self.0))
   }
}

impl fmt::Debug for FileHash {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "Hash({})", hex::encode(self.0))
   }
}

impl std::ops::Deref for FileHash {
   type Target = [u8];

   fn deref(&self) -> &Self::Target {
      &self.0
   }
}

impl std::ops::DerefMut for FileHash {
   fn deref_mut(&mut self) -> &mut Self::Target {
      &mut self.0
   }
}

#[derive(Serialize, Deserialize, Default)]
pub struct MetaStore {
   #[serde(default)]
   files:  HashMap<PathBuf, FileMeta>,
   #[serde(default)]
   hashes: HashMap<PathBuf, FileHash>,
   #[serde(skip)]
   path:   PathBuf,
   #[serde(skip)]
   dirty:  bool,
}

impl MetaStore {
   pub fn load(store_id: &str) -> Result<Self> {
      let meta_dir = config::data_dir().join("meta");
      let path = meta_dir.join(format!("{store_id}.json"));

      let mut store = if path.exists() {
         let content = fs::read_to_string(&path)?;
         let mut store: Self = serde_json::from_str(&content)?;
         store.path = path;
         store
      } else {
         Self { files: HashMap::new(), hashes: HashMap::new(), path, dirty: false }
      };

      store.dirty = false;
      Ok(store)
   }

   pub fn get_hash(&self, path: &Path) -> Option<FileHash> {
      self
         .files
         .get(path)
         .map(|m| m.hash)
         .or_else(|| self.hashes.get(path).copied())
   }

   pub fn get_mtime(&self, path: &Path) -> Option<u64> {
      self.files.get(path).map(|m| m.mtime)
   }

   pub fn get_meta(&self, path: &Path) -> Option<&FileMeta> {
      self.files.get(path)
   }

   pub fn set_hash(&mut self, path: &Path, hash: FileHash) {
      if let Some(meta) = self.files.get_mut(path) {
         meta.hash = hash;
      } else {
         self
            .files
            .insert(path.to_path_buf(), FileMeta { hash, mtime: 0 });
         self.hashes.remove(path);
      }
      self.dirty = true;
   }

   pub fn set_meta(&mut self, path: PathBuf, hash: FileHash, mtime: u64) {
      self.files.insert(path.clone(), FileMeta { hash, mtime });
      self.hashes.remove(&path);
      self.dirty = true;
   }

   pub fn remove(&mut self, path: &Path) {
      self.files.remove(path);
      self.hashes.remove(path);
      self.dirty = true;
   }

   pub fn save(&self) -> Result<()> {
      if let Some(parent) = self.path.parent() {
         fs::create_dir_all(parent)?;
      }

      let content = serde_json::to_string_pretty(&self)?;
      fs::write(&self.path, content)?;

      Ok(())
   }

   pub fn all_paths(&self) -> impl Iterator<Item = &PathBuf> {
      self.files.keys().chain(self.hashes.keys())
   }

   pub fn delete_by_prefix(&mut self, prefix: &Path) {
      self.files.retain(|path, _| !path.starts_with(prefix));
      self.hashes.retain(|path, _| !path.starts_with(prefix));
      self.dirty = true;
   }
}

#[cfg(test)]
mod tests {
   use tempfile::TempDir;

   use super::*;

   fn with_temp_home(f: impl FnOnce(&TempDir)) {
      let temp_dir = TempDir::new().unwrap();
      // SAFETY: we are setting the HOME environment variable to a temporary directory
      unsafe {
         std::env::set_var("HOME", temp_dir.path());
      }
      f(&temp_dir);
   }

   #[test]
   fn load_nonexistent_creates_empty() {
      with_temp_home(|_| {
         let store = MetaStore::load("test_store").unwrap();
         assert_eq!(store.hashes.len(), 0);
      });
   }

   #[test]
   fn set_and_get_hash() {
      with_temp_home(|_| {
         let hash = FileHash::sum(b"abc123");
         let mut store = MetaStore::load("test_store").unwrap();
         store.set_hash(Path::new("/path/to/file"), hash);

         assert_eq!(store.get_hash("/path/to/file".as_ref()), Some(hash));
         assert!(store.dirty);
      });
   }

   #[test]
   fn save_and_load_roundtrip() {
      with_temp_home(|_| {
         let mut store = MetaStore::load("test_store").unwrap();
         let hash1 = FileHash::sum(b"hash1");
         let hash2 = FileHash::sum(b"hash2");
         store.set_hash(Path::new("/file1"), hash1);
         store.set_hash(Path::new("/file2"), hash2);
         store.save().unwrap();

         let loaded = MetaStore::load("test_store").unwrap();
         assert_eq!(loaded.get_hash("/file1".as_ref()), Some(hash1));
         assert_eq!(loaded.get_hash("/file2".as_ref()), Some(hash2));
      });
   }

   #[test]
   fn remove_hash() {
      with_temp_home(|_| {
         let mut store = MetaStore::load("test_store").unwrap();
         let hash = FileHash::sum(b"hash1");
         store.set_hash(Path::new("/file1"), hash);
         store.remove("/file1".as_ref());

         assert_eq!(store.get_hash("/file1".as_ref()), None);
      });
   }

   #[test]
   fn all_paths_returns_keys() {
      with_temp_home(|_| {
         let mut store = MetaStore::load("test_store").unwrap();
         let hash1 = FileHash::sum(b"hash1");
         let hash2 = FileHash::sum(b"hash2");
         store.set_hash(Path::new("/file1"), hash1);
         store.set_hash(Path::new("/file2"), hash2);

         let paths: Vec<_> = store.all_paths().collect();
         assert_eq!(paths.len(), 2);
         assert!(paths.contains(&&PathBuf::from("/file1")));
         assert!(paths.contains(&&PathBuf::from("/file2")));
      });
   }
}
