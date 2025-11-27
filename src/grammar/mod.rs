use std::{
   collections::{HashMap, HashSet},
   fs, io,
   path::{Path, PathBuf},
};

use parking_lot::RwLock;
use tree_sitter::{Language, Parser, WasmStore, wasmtime};

use crate::error::{ChunkerError, ConfigError, Error, Result};

pub const GRAMMAR_URLS: &[(&str, &str)] = &[
    ("typescript", "https://github.com/tree-sitter/tree-sitter-typescript/releases/latest/download/tree-sitter-typescript.wasm"),
    ("tsx",        "https://github.com/tree-sitter/tree-sitter-typescript/releases/latest/download/tree-sitter-tsx.wasm"),
    ("python",     "https://github.com/tree-sitter/tree-sitter-python/releases/latest/download/tree-sitter-python.wasm"),
    ("go",         "https://github.com/tree-sitter/tree-sitter-go/releases/latest/download/tree-sitter-go.wasm"),
    ("rust",       "https://github.com/tree-sitter/tree-sitter-rust/releases/latest/download/tree-sitter-rust.wasm"),
    ("javascript", "https://github.com/tree-sitter/tree-sitter-javascript/releases/latest/download/tree-sitter-javascript.wasm"),
    ("c",          "https://github.com/tree-sitter/tree-sitter-c/releases/latest/download/tree-sitter-c.wasm"),
    ("cpp",        "https://github.com/tree-sitter/tree-sitter-cpp/releases/latest/download/tree-sitter-cpp.wasm"),
    ("java",       "https://github.com/tree-sitter/tree-sitter-java/releases/latest/download/tree-sitter-java.wasm"),
    ("ruby",       "https://github.com/tree-sitter/tree-sitter-ruby/releases/latest/download/tree-sitter-ruby.wasm"),
    ("php",        "https://github.com/tree-sitter/tree-sitter-php/releases/latest/download/tree-sitter-php.wasm"),
    ("html",       "https://github.com/tree-sitter/tree-sitter-html/releases/latest/download/tree-sitter-html.wasm"),
    ("css",        "https://github.com/tree-sitter/tree-sitter-css/releases/latest/download/tree-sitter-css.wasm"),
    ("bash",       "https://github.com/tree-sitter/tree-sitter-bash/releases/latest/download/tree-sitter-bash.wasm"),
    ("json",       "https://github.com/tree-sitter/tree-sitter-json/releases/latest/download/tree-sitter-json.wasm"),
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
   ("html", "html"),
   ("htm", "html"),
   ("css", "css"),
   ("sh", "bash"),
   ("bash", "bash"),
   ("json", "json"),
];

pub struct GrammarManager {
   grammars_dir:  PathBuf,
   engine:        wasmtime::Engine,
   languages:     RwLock<HashMap<String, Language>>,
   downloading:   RwLock<HashSet<String>>,
   auto_download: bool,
}

impl std::fmt::Debug for GrammarManager {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      f.debug_struct("GrammarManager")
         .field("languages", &self.languages)
         .field("grammars_dir", &self.grammars_dir)
         .field("downloading", &self.downloading)
         .field("auto_download", &self.auto_download)
         .finish()
   }
}

impl GrammarManager {
   pub fn new() -> Result<Self> {
      Self::with_auto_download(true)
   }

   pub fn with_auto_download(auto_download: bool) -> Result<Self> {
      let home = directories::UserDirs::new()
         .ok_or_else(|| Error::Config(ConfigError::GetUserDirectories))?
         .home_dir()
         .to_path_buf();

      let grammars_dir = home.join(".smgrep").join("grammars");
      fs::create_dir_all(&grammars_dir)
         .map_err(|e| Error::Config(ConfigError::CreateGrammarsDir(e)))?;

      let engine = wasmtime::Engine::default();

      Ok(Self {
         grammars_dir,
         engine,
         languages: RwLock::new(HashMap::new()),
         downloading: RwLock::new(HashSet::new()),
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
      self.grammars_dir.join(format!("tree-sitter-{lang}.wasm"))
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
      let wasm_bytes = fs::read(&wasm_path)?;

      let mut store = WasmStore::new(&self.engine).map_err(ChunkerError::CreateWasmStore)?;

      let language = store.load_language(lang, &wasm_bytes).map_err(|e| {
         Error::Chunker(ChunkerError::LoadLanguage { lang: lang.to_string(), reason: e })
      })?;

      Ok(language)
   }

   fn download_grammar_sync(&self, lang: &str) -> Result<()> {
      let url = Self::grammar_url(lang)
         .ok_or_else(|| Error::Config(ConfigError::UnknownLanguage(lang.to_string())))?;
      let dest = self.grammar_path(lang);
      let temp_dest = self
         .grammars_dir
         .join(format!(".tree-sitter-{lang}.wasm.tmp"));

      tracing::info!("downloading grammar for {} from {}", lang, url);

      let download = async {
         let response = reqwest::get(url).await.map_err(|e| {
            Error::Config(ConfigError::DownloadFailed { lang: lang.to_string(), reason: e })
         })?;

         if !response.status().is_success() {
            return Err(Error::Config(ConfigError::DownloadHttpStatus {
               lang:   lang.to_string(),
               status: response.status().as_u16(),
            }));
         }

         let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::Config(ConfigError::ReadResponse(e)))?;

         fs::write(&temp_dest, &bytes).map_err(|e| Error::Config(ConfigError::WriteWasmFile(e)))?;

         fs::rename(&temp_dest, &dest)
            .map_err(|e| Error::Config(ConfigError::RenameWasmFile(e)))?;

         Ok::<_, Error>(())
      };

      if let Ok(handle) = tokio::runtime::Handle::try_current() {
         tokio::task::block_in_place(|| handle.block_on(download))?;
      } else {
         tokio::runtime::Runtime::new()
            .map_err(|e| Error::Config(ConfigError::CreateRuntime(io::Error::other(e))))?
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
            if downloading.contains(lang) {
               return Ok(None);
            }
            downloading.insert(lang.to_string());
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

      let Some(lang) = Self::extension_to_language(&ext) else {
         return Ok(None);
      };

      self.get_language(lang)
   }

   pub fn create_parser_with_store(&self) -> Result<(Parser, WasmStore)> {
      let parser = Parser::new();
      let store = WasmStore::new(&self.engine).map_err(ChunkerError::CreateWasmStore)?;

      Ok((parser, store))
   }

   pub async fn download_grammar(&self, lang: &str) -> Result<()> {
      let url = Self::grammar_url(lang)
         .ok_or_else(|| Error::Config(ConfigError::UnknownLanguage(lang.to_string())))?;

      let dest = self.grammar_path(lang);
      let response = reqwest::get(url).await.map_err(|e| {
         Error::Config(ConfigError::DownloadFailed { lang: lang.to_string(), reason: e })
      })?;

      if !response.status().is_success() {
         return Err(Error::Config(ConfigError::DownloadHttpStatus {
            lang:   lang.to_string(),
            status: response.status().as_u16(),
         }));
      }

      let bytes = response
         .bytes()
         .await
         .map_err(|e| Error::Config(ConfigError::ReadResponse(e)))?;

      fs::write(&dest, &bytes).map_err(|e| Error::Config(ConfigError::WriteWasmFile(e)))?;

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
