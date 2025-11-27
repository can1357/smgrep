use std::path::{Path, PathBuf};

use git2::Repository;
use sha2::{Digest, Sha256};

use crate::error::{Result, RsgrepError};

pub fn is_git_repo(path: &Path) -> bool {
   Repository::open(path).is_ok()
}

pub fn get_repo_root(path: &Path) -> Option<PathBuf> {
   Repository::discover(path)
      .ok()
      .and_then(|repo| repo.workdir().map(|p| p.to_path_buf()))
}

pub fn get_remote_url(repo: &Repository) -> Option<String> {
   repo
      .find_remote("origin")
      .ok()
      .and_then(|remote| remote.url().map(|s| s.to_string()))
}

pub fn get_tracked_files(repo: &Repository) -> Result<Vec<PathBuf>> {
   let mut files = Vec::new();
   let index = repo
      .index()
      .map_err(|e| RsgrepError::Git(format!("failed to read index: {}", e)))?;

   let workdir = repo
      .workdir()
      .ok_or_else(|| RsgrepError::Git("repository has no working directory".to_string()))?;

   for entry in index.iter() {
      let path_bytes = entry.path.as_slice();
      if let Ok(path_str) = std::str::from_utf8(path_bytes) {
         let file_path = workdir.join(path_str);
         if file_path.exists() {
            files.push(file_path);
         }
      }
   }

   Ok(files)
}

pub fn resolve_store_id(path: &Path) -> Result<String> {
   let abs_path = path.canonicalize().map_err(RsgrepError::Io)?;

   if let Ok(repo) = Repository::open(&abs_path)
      && let Some(remote_url) = get_remote_url(&repo)
      && let Some(store_id) = extract_owner_repo(&remote_url)
   {
      return Ok(store_id);
   }

   let dir_name = abs_path
      .file_name()
      .and_then(|n| n.to_str())
      .unwrap_or("unknown");

   let path_hash = compute_path_hash(&abs_path);

   Ok(format!("{}-{}", dir_name, &path_hash[..8]))
}

fn extract_owner_repo(url: &str) -> Option<String> {
   let url = url.trim_end_matches(".git");

   if let Some(path_part) = url.strip_prefix("https://github.com/") {
      let parts: Vec<&str> = path_part.split('/').collect();
      if parts.len() >= 2 {
         return Some(format!("{}-{}", parts[0], parts[1]));
      }
   }

   if let Some(path_part) = url.strip_prefix("git@github.com:") {
      let parts: Vec<&str> = path_part.split('/').collect();
      if parts.len() >= 2 {
         return Some(format!("{}-{}", parts[0], parts[1]));
      }
   }

   if url.contains("://")
      && let Some(path_start) = url.rfind('/')
      && let Some(path_before) = url[..path_start].rfind('/')
   {
      let owner = &url[path_before + 1..path_start];
      let repo = &url[path_start + 1..];
      if !owner.is_empty() && !repo.is_empty() {
         return Some(format!("{}-{}", owner, repo));
      }
   }

   None
}

fn compute_path_hash(path: &Path) -> String {
   let mut hasher = Sha256::new();
   hasher.update(path.to_string_lossy().as_bytes());
   hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn extract_owner_repo_https() {
      let url = "https://github.com/kris-hansen/osgrep";
      assert_eq!(extract_owner_repo(url), Some("kris-hansen-osgrep".to_string()));
   }

   #[test]
   fn extract_owner_repo_https_with_git() {
      let url = "https://github.com/kris-hansen/osgrep.git";
      assert_eq!(extract_owner_repo(url), Some("kris-hansen-osgrep".to_string()));
   }

   #[test]
   fn extract_owner_repo_ssh() {
      let url = "git@github.com:kris-hansen/osgrep.git";
      assert_eq!(extract_owner_repo(url), Some("kris-hansen-osgrep".to_string()));
   }

   #[test]
   fn path_hash_computed() {
      let path = Path::new("/tmp/test");
      let hash = compute_path_hash(path);
      assert_eq!(hash.len(), 64);
   }
}
