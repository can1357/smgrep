use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
   "**/node_modules/**",
   "**/dist/**",
   "**/build/**",
   "**/out/**",
   "**/target/**",
   "**/__pycache__/**",
   "**/.git/**",
   "**/.venv/**",
   "**/venv/**",
   "*.lock",
   "*.bin",
   "*.ipynb",
   "*.pyc",
   "*.txt",
   "*.onnx",
   "package-lock.json",
   "yarn.lock",
   "pnpm-lock.yaml",
   "bun.lockb",
   "composer.lock",
   "Cargo.lock",
   "Gemfile.lock",
   "*.min.js",
   "*.min.css",
   "*.map",
   "**/coverage/**",
   "**/.nyc_output/**",
   "**/.pytest_cache/**",
];

pub struct IgnorePatterns {
   gitignore: Option<Gitignore>,
}

impl IgnorePatterns {
   pub fn new(root: &Path) -> Self {
      let mut builder = GitignoreBuilder::new(root);

      for pattern in DEFAULT_IGNORE_PATTERNS {
         let _ = builder.add_line(None, pattern);
      }

      let gitignore = root.join(".gitignore");
      if gitignore.exists() {
         let _ = builder.add(&gitignore);
      }

      let smgrep_ignore = root.join(".smignore");
      if smgrep_ignore.exists() {
         let _ = builder.add(&smgrep_ignore);
      }

      Self { gitignore: builder.build().ok() }
   }

   pub fn is_ignored(&self, path: &Path) -> bool {
      if let Some(ref gi) = self.gitignore {
         gi.matched(path, path.is_dir()).is_ignore()
      } else {
         false
      }
   }
}

#[cfg(test)]
mod tests {
   use std::fs;

   use tempfile::TempDir;

   use super::*;

   #[test]
   fn default_patterns_loaded() {
      let tmp = TempDir::new().unwrap();
      let ignore = IgnorePatterns::new(tmp.path());

      fs::create_dir_all(tmp.path().join("node_modules")).unwrap();
      fs::create_dir_all(tmp.path().join("dist")).unwrap();
      fs::create_dir_all(tmp.path().join("src")).unwrap();

      let node_modules = tmp.path().join("node_modules").join("package");
      let dist = tmp.path().join("dist").join("main.js");
      let src = tmp.path().join("src").join("main.rs");

      fs::write(&node_modules, "").unwrap();
      fs::write(&dist, "").unwrap();
      fs::write(&src, "").unwrap();

      assert!(ignore.is_ignored(&node_modules));
      assert!(ignore.is_ignored(&dist));
      assert!(!ignore.is_ignored(&src));
   }

   #[test]
   fn glob_patterns_work() {
      let tmp = TempDir::new().unwrap();
      let ignore = IgnorePatterns::new(tmp.path());
      let min_js = tmp.path().join("test.min.js");
      let min_css = tmp.path().join("bundle.min.css");
      let map = tmp.path().join("app.js.map");
      let normal_js = tmp.path().join("app.js");
      assert!(ignore.is_ignored(&min_js));
      assert!(ignore.is_ignored(&min_css));
      assert!(ignore.is_ignored(&map));
      assert!(!ignore.is_ignored(&normal_js));
   }

   #[test]
   fn negation_patterns_work() {
      let tmp = TempDir::new().unwrap();

      let ignore_file = tmp.path().join(".smignore");
      fs::write(&ignore_file, "*.log\n!important.log\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let test_log = tmp.path().join("test.log");
      let important_log = tmp.path().join("important.log");
      fs::write(&test_log, "").unwrap();
      fs::write(&important_log, "").unwrap();

      assert!(ignore.is_ignored(&test_log));
      assert!(!ignore.is_ignored(&important_log));
   }

   #[test]
   fn comment_patterns_ignored() {
      let tmp = TempDir::new().unwrap();

      let ignore_file = tmp.path().join(".smignore");
      fs::write(&ignore_file, "# This is a comment\n*.tmp\n# Another comment\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let tmp_file = tmp.path().join("test.tmp");
      fs::write(&tmp_file, "").unwrap();

      assert!(ignore.is_ignored(&tmp_file));
   }

   #[test]
   fn anchored_patterns_work() {
      let tmp = TempDir::new().unwrap();

      let ignore_file = tmp.path().join(".smignore");
      fs::write(&ignore_file, "/root.config\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let root_file = tmp.path().join("root.config");
      let nested_file = tmp.path().join("nested").join("root.config");
      fs::write(&root_file, "").unwrap();
      fs::create_dir(tmp.path().join("nested")).unwrap();
      fs::write(&nested_file, "").unwrap();

      assert!(ignore.is_ignored(&root_file));
      assert!(!ignore.is_ignored(&nested_file));
   }

   #[test]
   fn double_star_patterns_work() {
      let tmp = TempDir::new().unwrap();

      let ignore_file = tmp.path().join(".smignore");
      fs::write(&ignore_file, "**/generated/**\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let generated_dir = tmp.path().join("src").join("generated");
      fs::create_dir_all(&generated_dir).unwrap();
      let generated_file = generated_dir.join("code.ts");
      fs::write(&generated_file, "").unwrap();

      assert!(ignore.is_ignored(&generated_file));
   }

   #[test]
   fn respects_gitignore() {
      let tmp = TempDir::new().unwrap();

      let gitignore_file = tmp.path().join(".gitignore");
      fs::write(&gitignore_file, "*.secret\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let secret_file = tmp.path().join("passwords.secret");
      fs::write(&secret_file, "").unwrap();

      assert!(ignore.is_ignored(&secret_file));
   }
}
