use std::{
   fs,
   path::{Path, PathBuf},
   process::Command,
};

use git2::Repository;

use crate::error::{Error, Result};

const SUPPORTED_EXTENSIONS: &[&str] = &[
   "ts",
   "tsx",
   "js",
   "jsx",
   "py",
   "go",
   "rs",
   "java",
   "c",
   "cpp",
   "h",
   "hpp",
   "cs",
   "rb",
   "php",
   "swift",
   "kt",
   "scala",
   "vue",
   "svelte",
   "json",
   "yaml",
   "yml",
   "toml",
   "md",
   "txt",
   "sql",
   "sh",
   "bash",
   "zsh",
   "dockerfile",
   "makefile",
   "el",
   "clj",
   "cljs",
   "cljc",
   "edn",
   "ex",
   "exs",
   "dart",
   "m",
   "mm",
   "f90",
   "f95",
   "f03",
   "f08",
   "env",
   "gitignore",
   "gradle",
   "cmake",
   "proto",
   "graphql",
   "gql",
   "tf",
   "tfvars",
   "lua",
   "r",
   "R",
   "jl",
   "hs",
   "ml",
   "mli",
   "nim",
   "zig",
   "v",
   "cr",
];

const MAX_FILE_SIZE: u64 = 1024 * 1024;

pub trait FileSystem {
   fn get_files(&self, root: &Path) -> Result<Box<dyn Iterator<Item = PathBuf>>>;
}

pub struct LocalFileSystem;

impl LocalFileSystem {
   pub const fn new() -> Self {
      Self
   }

   fn is_supported_extension(path: &Path) -> bool {
      let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
      let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");

      SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
         || filename.eq_ignore_ascii_case("dockerfile")
         || filename.eq_ignore_ascii_case("makefile")
   }

   fn should_include_file(path: &Path) -> bool {
      if !Self::is_supported_extension(path) {
         return false;
      }

      if let Some(filename) = path.file_name().and_then(|f| f.to_str())
         && filename.starts_with('.')
      {
         return false;
      }

      if let Ok(metadata) = fs::metadata(path)
         && metadata.len() > MAX_FILE_SIZE
      {
         return false;
      }

      true
   }

   fn get_git_files(root: &Path) -> Result<Vec<PathBuf>> {
      let repo = Repository::open(root).map_err(Error::OpenRepository)?;

      let mut files = Vec::new();

      let index = repo.index().map_err(Error::ReadIndex)?;

      for entry in index.iter() {
         let path_bytes = entry.path.as_slice();
         if let Ok(path_str) = std::str::from_utf8(path_bytes) {
            let file_path = root.join(path_str);
            if file_path.exists() && Self::should_include_file(&file_path) {
               files.push(file_path);
            }
         }
      }

      let repo_path = repo.workdir().unwrap_or_else(|| repo.path());
      if let Ok(output) = Command::new("git")
         .args(["ls-files", "--others", "--exclude-standard"])
         .current_dir(repo_path)
         .output()
         && output.status.success()
      {
         for line in String::from_utf8_lossy(&output.stdout).lines() {
            let file_path = root.join(line);
            if file_path.exists() && Self::should_include_file(&file_path) {
               files.push(file_path);
            }
         }
      }

      Ok(files)
   }

   fn is_git_repository(path: &Path) -> bool {
      path.join(".git").exists()
   }

   fn get_walkdir_files(root: &Path) -> Vec<PathBuf> {
      Self::get_walkdir_files_recursive(root, root)
   }

   fn get_walkdir_files_recursive(dir: &Path, root: &Path) -> Vec<PathBuf> {
      let mut files = Vec::new();

      let Ok(entries) = fs::read_dir(dir) else {
         return files;
      };

      for entry in entries.filter_map(|e| e.ok()) {
         let path = entry.path();

         if let Some(filename) = path.file_name().and_then(|f| f.to_str())
            && filename.starts_with('.')
         {
            continue;
         }

         if path.is_dir() {
            if path != root && Self::is_git_repository(&path) {
               if let Ok(git_files) = Self::get_git_files(&path) {
                  files.extend(git_files);
               } else {
                  files.extend(Self::get_walkdir_files_recursive(&path, &path));
               }
            } else {
               files.extend(Self::get_walkdir_files_recursive(&path, root));
            }
         } else if path.is_file() && Self::should_include_file(&path) {
            files.push(path);
         }
      }

      files
   }
}

impl FileSystem for LocalFileSystem {
   fn get_files(&self, root: &Path) -> Result<Box<dyn Iterator<Item = PathBuf>>> {
      let files = if Repository::open(root).is_ok() {
         Self::get_git_files(root)?
      } else {
         Self::get_walkdir_files(root)
      };

      Ok(Box::new(files.into_iter()))
   }
}

impl Default for LocalFileSystem {
   fn default() -> Self {
      Self::new()
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn supported_extension_recognized() {
      assert!(LocalFileSystem::is_supported_extension(Path::new("test.rs")));
      assert!(LocalFileSystem::is_supported_extension(Path::new("test.ts")));
      assert!(LocalFileSystem::is_supported_extension(Path::new("test.py")));
      assert!(!LocalFileSystem::is_supported_extension(Path::new("test.bin")));
   }

   #[test]
   fn hidden_files_filtered() {
      assert!(!LocalFileSystem::should_include_file(Path::new(".hidden.rs")));
      assert!(LocalFileSystem::should_include_file(Path::new("visible.rs")));
   }
}
