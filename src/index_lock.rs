use std::{
   fs::{self, File, OpenOptions},
   path::PathBuf,
};

use crate::{Result, config};

pub struct IndexLock {
   file: File,
}

impl IndexLock {
   pub fn acquire(store_id: &str) -> Result<Self> {
      let lock_path: PathBuf = config::data_dir().join(format!("{store_id}.lock"));

      if let Some(parent) = lock_path.parent() {
         fs::create_dir_all(parent)?;
      }

      let file = OpenOptions::new()
         .create(true)
         .truncate(true)
         .read(true)
         .write(true)
         .open(&lock_path)?;

      file.lock()?;

      Ok(Self { file })
   }
}

impl Drop for IndexLock {
   fn drop(&mut self) {
      let _ = self.file.unlock();
   }
}
