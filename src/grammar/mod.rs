use std::{
   collections::HashMap,
   fs,
   path::{Path, PathBuf},
};

use parking_lot::RwLock;
use tree_sitter::{Language, Parser, WasmStore, wasmtime};

use crate::error::{Result, RsgrepError};

pub const GRAMMAR_URLS: &[(&str, &str)] = &[
   ("typescript", "https://github.com/nicholasjchua/tree-sitter-typescript/releases/download/v0.25.5/tree-sitter-typescript.wasm"),
   ("tsx", "https://github.com/nicholasjchua/tree-sitter-typescript/releases/download/v0.25.5/tree-sitter-tsx.wasm"),
   ("python", "https://github.com/nicholasjchua/tree-sitter-python/releases/download/v0.25.0/tree-sitter-python.wasm"),
   ("go", "https://github.com/nicholasjchua/tree-sitter-go/releases/download/v0.25.0/tree-sitter-go.wasm"),
   ("rust", "https://github.com/nicholasjchua/tree-sitter-rust/releases/download/v0.25.0/tree-sitter-rust.wasm"),
   ("javascript", "https://github.com/nicholasjchua/tree-sitter-javascript/releases/download/v0.25.0/tree-sitter-javascript.wasm"),
   ("c", "https://github.com/nicholasjchua/tree-sitter-c/releases/download/v0.25.1/tree-sitter-c.wasm"),
   ("cpp", "https://github.com/nicholasjchua/tree-sitter-cpp/releases/download/v0.25.0/tree-sitter-cpp.wasm"),
   ("java", "https://github.com/nicholasjchua/tree-sitter-java/releases/download/v0.25.0/tree-sitter-java.wasm"),
   ("ruby", "https://github.com/nicholasjchua/tree-sitter-ruby/releases/download/v0.25.0/tree-sitter-ruby.wasm"),
   ("php", "https://github.com/nicholasjchua/tree-sitter-php/releases/download/v0.25.0/tree-sitter-php.wasm"),
   ("swift", "https://github.com/nicholasjchua/tree-sitter-swift/releases/download/v0.25.0/tree-sitter-swift.wasm"),
   ("html", "https://github.com/nicholasjchua/tree-sitter-html/releases/download/v0.25.0/tree-sitter-html.wasm"),
   ("css", "https://github.com/nicholasjchua/tree-sitter-css/releases/download/v0.25.0/tree-sitter-css.wasm"),
   ("bash", "https://github.com/nicholasjchua/tree-sitter-bash/releases/download/v0.25.0/tree-sitter-bash.wasm"),
   ("json", "https://github.com/nicholasjchua/tree-sitter-json/releases/download/v0.25.0/tree-sitter-json.wasm"),
   ("yaml", "https://github.com/nicholasjchua/tree-sitter-yaml/releases/download/v0.25.0/tree-sitter-yaml.wasm"),
];

pub static EXTENSION_MAP: &[(&str, &str)] = &[
   ("js", "javascript"),
   ("mjs", "javascript"),
   ("cjs", "javascript"),
   ("ts", "typescript"),
   ("mts", "typescript"),
   ("cts", "typescript"),
   ("jsx", "tsx"),
   ("tsx", "tsx"),
   ("py", "python"),
   ("pyi", "python"),
   ("go", "go"),
   ("rs", "rust"),
   ("c", "c"),
   ("h", "c"),
   ("cpp", "cpp"),
   ("cc", "cpp"),
   ("cxx", "cpp"),
   ("c++", "cpp"),
   ("hpp", "cpp"),
   ("hxx", "cpp"),
   ("h++", "cpp"),
   ("java", "java"),
   ("rb", "ruby"),
   ("php", "php"),
   ("swift", "swift"),
   ("html", "html"),
   ("htm", "html"),
   ("css", "css"),
   ("sh", "bash"),
   ("bash", "bash"),
   ("json", "json"),
   ("yaml", "yaml"),
   ("yml", "yaml"),
];

pub struct GrammarManager {
   grammars_dir:  PathBuf,
   engine:        wasmtime::Engine,
   languages:     RwLock<HashMap<String, Language>>,
   downloading:   RwLock<HashMap<String, ()>>,
   auto_download: bool,
}

impl GrammarManager {
   pub fn new() -> Result<Self> {
      Self::with_auto_download(true)
   }

   pub fn with_auto_download(auto_download: bool) -> Result<Self> {
      let home = directories::UserDirs::new()
         .ok_or_else(|| RsgrepError::Config("failed to get user directories".into()))?
         .home_dir()
         .to_path_buf();

      let grammars_dir = home.join(".rsgrep").join("grammars");
      fs::create_dir_all(&grammars_dir)
         .map_err(|e| RsgrepError::Config(format!("failed to create grammars directory: {}", e)))?;

      let engine = wasmtime::Engine::default();

      Ok(Self {
         grammars_dir,
         engine,
         languages: RwLock::new(HashMap::new()),
         downloading: RwLock::new(HashMap::new()),
         auto_download,
      })
   }

   pub fn grammars_dir(&self) -> &Path {
      &self.grammars_dir
   }

   pub fn extension_to_language(ext: &str) -> Option<&'static str> {
      EXTENSION_MAP
         .iter()
         .find(|(e, _)| *e == ext)
         .map(|(_, lang)| *lang)
   }

   pub fn grammar_url(lang: &str) -> Option<&'static str> {
      GRAMMAR_URLS
         .iter()
         .find(|(l, _)| *l == lang)
         .map(|(_, url)| *url)
   }

   pub fn grammar_path(&self, lang: &str) -> PathBuf {
      self.grammars_dir.join(format!("tree-sitter-{}.wasm", lang))
   }

   pub fn is_available(&self, lang: &str) -> bool {
      self.grammar_path(lang).exists()
   }

   pub fn available_languages(&self) -> Vec<&'static str> {
      GRAMMAR_URLS
         .iter()
         .filter(|(lang, _)| self.is_available(lang))
         .map(|(lang, _)| *lang)
         .collect()
   }

   pub fn missing_languages(&self) -> Vec<&'static str> {
      GRAMMAR_URLS
         .iter()
         .filter(|(lang, _)| !self.is_available(lang))
         .map(|(lang, _)| *lang)
         .collect()
   }

   fn load_language_from_file(&self, lang: &str) -> Result<Language> {
      let wasm_path = self.grammar_path(lang);
      let wasm_bytes =
         fs::read(&wasm_path).map_err(|e| RsgrepError::Chunker(format!("failed to read WASM file: {}", e)))?;

      let mut store = WasmStore::new(&self.engine)
         .map_err(|e| RsgrepError::Chunker(format!("failed to create WASM store: {}", e)))?;

      let language = store
         .load_language(lang, &wasm_bytes)
         .map_err(|e| RsgrepError::Chunker(format!("failed to load language {}: {}", lang, e)))?;

      Ok(language)
   }

   fn download_grammar_sync(&self, lang: &str) -> Result<()> {
      let url = Self::grammar_url(lang).ok_or_else(|| RsgrepError::Config(format!("unknown language: {}", lang)))?;
      let dest = self.grammar_path(lang);
      let temp_dest = self.grammars_dir.join(format!(".tree-sitter-{}.wasm.tmp", lang));

      tracing::info!("downloading grammar for {} from {}", lang, url);

      let download = async {
         let response = reqwest::get(url)
            .await
            .map_err(|e| RsgrepError::Config(format!("failed to download {}: {}", lang, e)))?;

         if !response.status().is_success() {
            return Err(RsgrepError::Config(format!(
               "failed to download {}: HTTP {}",
               lang,
               response.status()
            )));
         }

         let bytes = response
            .bytes()
            .await
            .map_err(|e| RsgrepError::Config(format!("failed to read response: {}", e)))?;

         fs::write(&temp_dest, &bytes)
            .map_err(|e| RsgrepError::Config(format!("failed to write WASM file: {}", e)))?;

         fs::rename(&temp_dest, &dest)
            .map_err(|e| RsgrepError::Config(format!("failed to rename WASM file: {}", e)))?;

         Ok::<_, RsgrepError>(())
      };

      if let Ok(handle) = tokio::runtime::Handle::try_current() {
         tokio::task::block_in_place(|| handle.block_on(download))?;
      } else {
         tokio::runtime::Runtime::new()
            .map_err(|e| RsgrepError::Config(format!("failed to create runtime: {}", e)))?
            .block_on(download)?;
      }

      tracing::info!("downloaded grammar for {}", lang);

      Ok(())
   }

   pub fn get_language(&self, lang: &str) -> Result<Option<Language>> {
      {
         let cache = self.languages.read();
         if let Some(language) = cache.get(lang) {
            return Ok(Some(language.clone()));
         }
      }

      if !self.is_available(lang) {
         if !self.auto_download {
            return Ok(None);
         }

         if Self::grammar_url(lang).is_none() {
            return Ok(None);
         }

         {
            let mut downloading = self.downloading.write();
            if downloading.contains_key(lang) {
               return Ok(None);
            }
            downloading.insert(lang.to_string(), ());
         }

         let result = self.download_grammar_sync(lang);

         {
            let mut downloading = self.downloading.write();
            downloading.remove(lang);
         }

         if let Err(e) = result {
            tracing::warn!("failed to download grammar for {}: {}", lang, e);
            return Ok(None);
         }
      }

      let language = self.load_language_from_file(lang)?;

      {
         let mut cache = self.languages.write();
         cache.insert(lang.to_string(), language.clone());
      }

      Ok(Some(language))
   }

   pub fn get_language_for_path(&self, path: &Path) -> Result<Option<Language>> {
      let ext = match path.extension().and_then(|e| e.to_str()) {
         Some(e) => e.to_lowercase(),
         None => return Ok(None),
      };

      let lang = match Self::extension_to_language(&ext) {
         Some(l) => l,
         None => return Ok(None),
      };

      self.get_language(lang)
   }

   pub fn create_parser_with_store(&self) -> Result<(Parser, WasmStore)> {
      let parser = Parser::new();
      let store = WasmStore::new(&self.engine)
         .map_err(|e| RsgrepError::Chunker(format!("failed to create WASM store: {}", e)))?;

      Ok((parser, store))
   }

   pub async fn download_grammar(&self, lang: &str) -> Result<()> {
      let url = Self::grammar_url(lang).ok_or_else(|| RsgrepError::Config(format!("unknown language: {}", lang)))?;

      let dest = self.grammar_path(lang);

      let response = reqwest::get(url)
         .await
         .map_err(|e| RsgrepError::Config(format!("failed to download {}: {}", lang, e)))?;

      if !response.status().is_success() {
         return Err(RsgrepError::Config(format!(
            "failed to download {}: HTTP {}",
            lang,
            response.status()
         )));
      }

      let bytes = response
         .bytes()
         .await
         .map_err(|e| RsgrepError::Config(format!("failed to read response: {}", e)))?;

      fs::write(&dest, &bytes).map_err(|e| RsgrepError::Config(format!("failed to write WASM file: {}", e)))?;

      Ok(())
   }

   pub async fn download_all_grammars(&self) -> Vec<(&'static str, Result<()>)> {
      let mut results = Vec::new();

      for (lang, _) in GRAMMAR_URLS {
         let result = self.download_grammar(lang).await;
         results.push((*lang, result));
      }

      results
   }
}

impl Default for GrammarManager {
   fn default() -> Self {
      Self::new().expect("failed to create grammar manager")
   }
}
