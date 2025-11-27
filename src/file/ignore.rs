use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
   "node_modules",
   "dist",
   "build",
   "target",
   "__pycache__",
   ".git",
   ".venv",
   "*.lock",
   "package-lock.json",
   "yarn.lock",
   "pnpm-lock.yaml",
   "Cargo.lock",
   "*.min.js",
   "*.min.css",
   "*.map",
   "coverage",
   ".nyc_output",
   ".pytest_cache",
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

      let rsgrep_ignore = root.join(".rsgrep").join("ignore");
      if rsgrep_ignore.exists() {
         let _ = builder.add(&rsgrep_ignore);
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
      let node_modules = tmp.path().join("node_modules").join("package");
      let dist = tmp.path().join("dist").join("main.js");
      let src = tmp.path().join("src").join("main.rs");
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
      let rsgrep_dir = tmp.path().join(".rsgrep");
      fs::create_dir(&rsgrep_dir).unwrap();

      let ignore_file = rsgrep_dir.join("ignore");
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
      let rsgrep_dir = tmp.path().join(".rsgrep");
      fs::create_dir(&rsgrep_dir).unwrap();

      let ignore_file = rsgrep_dir.join("ignore");
      fs::write(&ignore_file, "# This is a comment\n*.tmp\n# Another comment\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let tmp_file = tmp.path().join("test.tmp");
      fs::write(&tmp_file, "").unwrap();

      assert!(ignore.is_ignored(&tmp_file));
   }

   #[test]
   fn anchored_patterns_work() {
      let tmp = TempDir::new().unwrap();
      let rsgrep_dir = tmp.path().join(".rsgrep");
      fs::create_dir(&rsgrep_dir).unwrap();

      let ignore_file = rsgrep_dir.join("ignore");
      fs::write(&ignore_file, "/root.txt\n").unwrap();

      let ignore = IgnorePatterns::new(tmp.path());

      let root_file = tmp.path().join("root.txt");
      let nested_file = tmp.path().join("nested").join("root.txt");
      fs::write(&root_file, "").unwrap();
      fs::create_dir(tmp.path().join("nested")).unwrap();
      fs::write(&nested_file, "").unwrap();

      assert!(ignore.is_ignored(&root_file));
      assert!(!ignore.is_ignored(&nested_file));
   }

   #[test]
   fn double_star_patterns_work() {
      let tmp = TempDir::new().unwrap();
      let rsgrep_dir = tmp.path().join(".rsgrep");
      fs::create_dir(&rsgrep_dir).unwrap();

      let ignore_file = rsgrep_dir.join("ignore");
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
