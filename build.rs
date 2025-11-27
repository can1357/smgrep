use std::process::Command;

fn main() {
   println!("cargo:rerun-if-changed=.git/HEAD");
   println!("cargo:rerun-if-changed=.git/refs/");

   let git_hash = Command::new("git")
      .args(["rev-parse", "--short", "HEAD"])
      .output()
      .ok()
      .filter(|o| o.status.success())
      .and_then(|o| String::from_utf8(o.stdout).ok())
      .map(|s| s.trim().to_string())
      .unwrap_or_else(|| "unknown".to_string());

   let git_tag = Command::new("git")
      .args(["describe", "--tags", "--abbrev=0"])
      .output()
      .ok()
      .filter(|o| o.status.success())
      .and_then(|o| String::from_utf8(o.stdout).ok())
      .map(|s| s.trim().to_string())
      .unwrap_or_default();

   let git_dirty = Command::new("git")
      .args(["status", "--porcelain"])
      .output()
      .ok()
      .filter(|o| o.status.success())
      .map(|o| !o.stdout.is_empty())
      .unwrap_or(false);

   println!("cargo:rustc-env=GIT_HASH={}", git_hash);
   println!("cargo:rustc-env=GIT_TAG={}", git_tag);
   println!("cargo:rustc-env=GIT_DIRTY={}", if git_dirty { "true" } else { "false" });
}
