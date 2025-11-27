use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{Debouncer, new_debouncer};
use parking_lot::Mutex;

use super::IgnorePatterns;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchAction {
   Upsert,
   Delete,
}

pub struct FileWatcher {
   debouncer:       Option<Debouncer<RecommendedWatcher>>,
   pending:         Arc<Mutex<HashMap<PathBuf, WatchAction>>>,
   root:            PathBuf,
   ignore_patterns: Arc<IgnorePatterns>,
}

impl FileWatcher {
   pub fn new<F>(
      root: PathBuf,
      ignore_patterns: IgnorePatterns,
      on_changes: F,
   ) -> anyhow::Result<Self>
   where
      F: Fn(Vec<(PathBuf, WatchAction)>) + Send + 'static,
   {
      let pending = Arc::new(Mutex::new(HashMap::new()));
      let pending_clone = Arc::clone(&pending);
      let ignore_patterns = Arc::new(ignore_patterns);
      let ignore_patterns_clone = Arc::clone(&ignore_patterns);

      let mut debouncer = new_debouncer(
         Duration::from_millis(300),
         move |res: notify_debouncer_mini::DebounceEventResult| match res {
            Ok(events) => {
               let mut pending_map = pending_clone.lock();
               for event in events {
                  let path = event.path;
                  if ignore_patterns_clone.is_ignored(&path) {
                     continue;
                  }

                  pending_map.insert(path, WatchAction::Upsert);
               }

               let changes: Vec<(PathBuf, WatchAction)> = pending_map.drain().collect();
               if !changes.is_empty() {
                  on_changes(changes);
               }
            },
            Err(error) => {
               tracing::error!("File watcher error: {:?}", error);
            },
         },
      )?;

      debouncer.watcher().watch(&root, RecursiveMode::Recursive)?;

      Ok(Self { debouncer: Some(debouncer), pending, root, ignore_patterns })
   }

   pub fn stop(&mut self) {
      if let Some(mut debouncer) = self.debouncer.take()
         && let Err(e) = debouncer.watcher().unwatch(&self.root)
      {
         tracing::warn!("Failed to unwatch root directory: {}", e);
      }
   }
}

impl Drop for FileWatcher {
   fn drop(&mut self) {
      self.stop();
   }
}
