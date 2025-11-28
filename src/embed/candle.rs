use std::{
   fmt, fs, io,
   path::PathBuf,
   sync::{
      OnceLock,
      atomic::{AtomicUsize, Ordering},
   },
};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{Repo, RepoType, api::tokio::Api};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::{
   Str, config,
   embed::{Embedder, HybridEmbedding, QueryEmbedding},
   error::Result,
};

const MAX_SEQ_LEN_DENSE: usize = 256;
const MAX_SEQ_LEN_COLBERT: usize = 512;
const MIN_BATCH_SIZE: usize = 1;

#[derive(Debug)]
pub struct Models(DenseModelState, ColbertModelState);

#[derive(Debug)]
pub struct CandleEmbedder {
   models:              OnceLock<Models>,
   init_lock:           Mutex<()>,
   device:              Device,
   adaptive_batch_size: AtomicUsize,
}

struct DenseModelState {
   name:      &'static str,
   bert:      BertModel,
   tokenizer: Tokenizer,
}

impl fmt::Debug for DenseModelState {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("DenseModelState")
         .field("bert", &self.name)
         .field("device", &self.bert.device)
         .field("tokenizer", &self.tokenizer)
         .finish()
   }
}

struct ColbertModelState {
   name:       &'static str,
   bert:       BertModel,
   projection: Linear,
   tokenizer:  Tokenizer,
}

impl fmt::Debug for ColbertModelState {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("ColbertModelState")
         .field("bert", &self.name)
         .field("device", &self.bert.device)
         .field("projection", &self.projection)
         .field("tokenizer", &self.tokenizer)
         .finish()
   }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
   #[error("failed to load tokenizer: {0}")]
   LoadTokenizer(#[source] tokenizers::Error),

   #[error("failed to read config: {0}")]
   ReadConfig(#[source] io::Error),

   #[error("failed to parse config: {0}")]
   ParseConfig(#[from] serde_json::Error),

   #[error("failed to load weights: {0}")]
   LoadWeights(#[source] candle_core::Error),

   #[error("failed to load model: {0}")]
   LoadModel(#[source] candle_core::Error),

   #[error("failed to load colbert model: {0}")]
   LoadColbertModel(#[source] candle_core::Error),

   #[error("failed to load projection: {0}")]
   LoadProjection(#[source] candle_core::Error),

   #[error("failed to create model cache: {0}")]
   CreateModelCache(#[from] io::Error),

   #[error("failed to initialize hf_hub API: {0}")]
   InitHfHub(#[from] hf_hub::api::tokio::ApiError),

   #[error(
      "failed to download model file {file} from {model}: {reason}. Run 'smgrep setup' to \
       download models."
   )]
   DownloadModel { file: String, model: String, reason: String },

   #[error("invalid model path")]
   InvalidModelPath,

   #[error("dense model not loaded")]
   DenseModelNotLoaded,

   #[error("colbert model not loaded")]
   ColbertModelNotLoaded,

   #[error("tokenization failed: {0}")]
   TokenizationFailed(#[from] tokenizers::Error),

   #[error("failed to create tensor: {0}")]
   CreateTensor(#[source] candle_core::Error),

   #[error("failed to unsqueeze: {0}")]
   Unsqueeze(#[source] candle_core::Error),

   #[error("failed to reshape: {0}")]
   Reshape(#[source] candle_core::Error),

   #[error("failed to create mask: {0}")]
   CreateMask(#[source] candle_core::Error),

   #[error("forward pass failed: {0}")]
   ForwardPass(#[source] candle_core::Error),

   #[error("failed to get batch: {0}")]
   GetBatch(#[source] candle_core::Error),

   #[error("failed to extract CLS: {0}")]
   ExtractCls(#[source] candle_core::Error),

   #[error("failed to convert to vec: {0}")]
   ConvertToVec(#[source] candle_core::Error),

   #[error("failed to extract token {i}: {e}")]
   ExtractToken {
      i: usize,
      #[source]
      e: candle_core::Error,
   },

   #[error("failed to convert token {i}: {e}")]
   ConvertToken {
      i: usize,
      #[source]
      e: candle_core::Error,
   },

   #[error("projection failed: {0}")]
   Projection(#[source] candle_core::Error),

   #[error("dtype conversion failed: {0}")]
   DtypeConversion(#[source] candle_core::Error),

   #[error("failed to convert: {0}")]
   Convert(#[source] candle_core::Error),

   #[error("worker closed")]
   WorkerClosed,

   #[error("work cancelled")]
   WorkCancelled,
}

fn is_oom_error(err: &str) -> bool {
   err.contains("out of memory")
      || err.contains("OUT_OF_MEMORY")
      || err.contains("CUDA_ERROR_OUT_OF_MEMORY")
      || err.contains("alloc")
}

impl CandleEmbedder {
   pub fn new() -> Result<Self> {
      let cfg = config::get();
      let device = if cfg.disable_gpu {
         Device::Cpu
      } else {
         Device::cuda_if_available(0).unwrap_or(Device::Cpu)
      };
      if device.is_cuda() {
         tracing::info!("using CUDA device");
      } else {
         tracing::info!("using CPU device");
      }

      let initial_batch = cfg.batch_size();

      Ok(Self {
         models: OnceLock::new(),
         init_lock: Mutex::new(()),
         device,
         adaptive_batch_size: AtomicUsize::new(initial_batch),
      })
   }

   pub fn current_batch_size(&self) -> usize {
      self.adaptive_batch_size.load(Ordering::Relaxed)
   }

   fn reduce_batch_size(&self) -> usize {
      let current = self.adaptive_batch_size.load(Ordering::Relaxed);
      let new_size = (current / 2).max(MIN_BATCH_SIZE);
      self.adaptive_batch_size.store(new_size, Ordering::Relaxed);
      tracing::warn!("OOM detected, reducing batch size: {} -> {}", current, new_size);
      new_size
   }

   #[inline(always)]
   async fn models(&self) -> Result<&Models> {
      if self.models.get().is_some() {
         return Ok(self.models.get().unwrap());
      }
      self.init_models_cold().await
   }

   #[cold]
   async fn init_models_cold(&self) -> Result<&Models> {
      let _guard = self.init_lock.lock().await;
      if self.models.get().is_some() {
         return Ok(self.models.get().unwrap());
      }

      let dense = Self::load_dense(&self.device).await?;
      let colbert = Self::load_colbert(&self.device).await?;

      self
         .models
         .set(Models(dense, colbert))
         .expect("should be exclusive under self.init_lock");
      Ok(self.models.get().unwrap())
   }

   async fn load_dense(device: &Device) -> Result<DenseModelState> {
      let cfg = config::get();
      let model_path = Self::download_model(&cfg.dense_model).await?;

      if cfg.debug_models {
         tracing::info!("loading dense model from {:?}", model_path);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(EmbeddingError::LoadTokenizer)?;

      let config: BertConfig = serde_json::from_str(
         &fs::read_to_string(model_path.join("config.json")).map_err(EmbeddingError::ReadConfig)?,
      )?;

      // SAFETY: VarBuilder::from_mmaped_safetensors is safe to use as it properly
      // handles memory mapping
      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], DTYPE, device)
            .map_err(EmbeddingError::LoadWeights)?
      };

      let bert = BertModel::load(vb, &config).map_err(EmbeddingError::LoadModel)?;

      if cfg.debug_models {
         tracing::info!("dense model loaded");
      }

      Ok(DenseModelState { name: cfg.dense_model.as_str(), bert, tokenizer })
   }

   async fn load_colbert(device: &Device) -> Result<ColbertModelState> {
      let cfg = config::get();
      let model_path = Self::download_model(&cfg.colbert_model).await?;

      if cfg.debug_models {
         tracing::info!("loading colbert model from {:?}", model_path);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(EmbeddingError::LoadTokenizer)?;

      let config: BertConfig = serde_json::from_str(
         &fs::read_to_string(model_path.join("config.json")).map_err(EmbeddingError::ReadConfig)?,
      )?;

      // SAFETY: VarBuilder::from_mmaped_safetensors is safe to use as it properly
      // handles memory mapping
      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], DTYPE, device)
            .map_err(EmbeddingError::LoadWeights)?
      };

      let bert = BertModel::load(vb.clone(), &config).map_err(EmbeddingError::LoadColbertModel)?;

      let projection =
         candle_nn::linear_no_bias(config.hidden_size, cfg.colbert_dim, vb.pp("linear"))
            .map_err(EmbeddingError::LoadProjection)?;

      if cfg.debug_models {
         tracing::info!(
            "colbert model loaded (hidden={}, proj={})",
            config.hidden_size,
            cfg.colbert_dim
         );
      }

      Ok(ColbertModelState { name: cfg.colbert_model.as_str(), bert, projection, tokenizer })
   }

   async fn download_model(model_id: &str) -> Result<PathBuf> {
      let cache_dir = config::model_dir();
      fs::create_dir_all(cache_dir).map_err(EmbeddingError::CreateModelCache)?;

      let api = Api::new().map_err(EmbeddingError::InitHfHub)?;

      let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

      let model_files = ["config.json", "tokenizer.json", "model.safetensors"];
      let mut paths = Vec::new();

      for filename in &model_files {
         let path = repo
            .get(filename)
            .await
            .map_err(|e| EmbeddingError::DownloadModel {
               file:   filename.to_string(),
               model:  model_id.to_string(),
               reason: e.to_string(),
            })?;
         paths.push(path);
      }

      Ok(paths[0]
         .parent()
         .ok_or(EmbeddingError::InvalidModelPath)?
         .to_path_buf())
   }

   async fn tokenize_dense(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
      let Models(dense, _) = self.models().await?;

      let encoding = dense
         .tokenizer
         .encode(text, true)
         .map_err(EmbeddingError::from)?;

      let mut token_ids = encoding.get_ids().to_vec();
      let mut attention_mask = vec![1u32; token_ids.len()];

      if token_ids.len() > MAX_SEQ_LEN_DENSE {
         token_ids.truncate(MAX_SEQ_LEN_DENSE);
         attention_mask.truncate(MAX_SEQ_LEN_DENSE);
      }

      Ok((token_ids, attention_mask))
   }

   async fn tokenize_dense_batch(&self, texts: &[Str]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let Models(dense, _) = self.models().await?;

      texts
         .iter()
         .map(|text| {
            let encoding = dense
               .tokenizer
               .encode(text.as_str(), true)
               .map_err(EmbeddingError::from)?;

            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = vec![1u32; token_ids.len()];

            if token_ids.len() > MAX_SEQ_LEN_DENSE {
               token_ids.truncate(MAX_SEQ_LEN_DENSE);
               attention_mask.truncate(MAX_SEQ_LEN_DENSE);
            }

            Ok((token_ids, attention_mask))
         })
         .collect()
   }

   async fn tokenize_colbert(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
      let Models(_, colbert) = self.models().await?;

      let encoding = colbert
         .tokenizer
         .encode(text, true)
         .map_err(EmbeddingError::from)?;

      let mut token_ids = encoding.get_ids().to_vec();
      let mut attention_mask = vec![1u32; token_ids.len()];

      if token_ids.len() > MAX_SEQ_LEN_COLBERT {
         token_ids.truncate(MAX_SEQ_LEN_COLBERT);
         attention_mask.truncate(MAX_SEQ_LEN_COLBERT);
      }

      Ok((token_ids, attention_mask))
   }

   async fn tokenize_colbert_batch(&self, texts: &[Str]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let Models(_, colbert) = self.models().await?;

      texts
         .iter()
         .map(|text| {
            let encoding = colbert
               .tokenizer
               .encode(text.as_str(), true)
               .map_err(EmbeddingError::from)?;

            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = vec![1u32; token_ids.len()];

            if token_ids.len() > MAX_SEQ_LEN_COLBERT {
               token_ids.truncate(MAX_SEQ_LEN_COLBERT);
               attention_mask.truncate(MAX_SEQ_LEN_COLBERT);
            }

            Ok((token_ids, attention_mask))
         })
         .collect()
   }

   fn normalize_l2(embeddings: &mut [f32]) {
      let norm: f32 = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
      if norm > 0.0 {
         for x in embeddings.iter_mut() {
            *x /= norm;
         }
      }
   }

   fn quantize_embeddings(embeddings: &[Vec<f32>]) -> (Vec<u8>, f64) {
      if embeddings.is_empty() || embeddings[0].is_empty() {
         return (Vec::new(), 1.0);
      }

      let max_val = embeddings
         .iter()
         .flatten()
         .map(|x| x.abs())
         .fold(0.0f32, f32::max);

      if max_val == 0.0 {
         return (vec![0; embeddings.len() * embeddings[0].len()], 1.0);
      }

      let scale = max_val as f64 / 127.0;
      let quantized: Vec<u8> = embeddings
         .iter()
         .flatten()
         .map(|x| ((x / max_val) * 127.0) as i8 as u8)
         .collect();

      (quantized, scale)
   }

   async fn compute_dense_embedding(&self, text: &str) -> Result<Vec<f32>> {
      let (token_ids, attention_mask) = self.tokenize_dense(text).await?;

      let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)
         .map_err(EmbeddingError::CreateTensor)?
         .unsqueeze(0)
         .map_err(EmbeddingError::Unsqueeze)?;

      let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
         .map_err(EmbeddingError::CreateMask)?
         .unsqueeze(0)
         .map_err(EmbeddingError::Unsqueeze)?;

      let Models(dense, _) = self.models().await?;

      let embeddings = dense
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(EmbeddingError::ForwardPass)?;

      let cls_embedding = embeddings
         .get(0)
         .map_err(EmbeddingError::GetBatch)?
         .get(0)
         .map_err(EmbeddingError::ExtractCls)?;

      let mut dense_vec: Vec<f32> = cls_embedding
         .to_vec1()
         .map_err(EmbeddingError::ConvertToVec)?;

      Self::normalize_l2(&mut dense_vec);
      Ok(dense_vec)
   }

   async fn compute_dense_embeddings_batch_inner(
      &self,
      tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<Vec<Vec<f32>>> {
      if tokenized.is_empty() {
         return Ok(Vec::new());
      }

      let max_len = tokenized
         .iter()
         .map(|(ids, _)| ids.len())
         .max()
         .unwrap_or(0);
      let batch_size = tokenized.len();

      let mut all_token_ids = Vec::with_capacity(batch_size * max_len);
      let mut all_attention_masks = Vec::with_capacity(batch_size * max_len);

      for (token_ids, attention_mask) in tokenized {
         all_token_ids.extend(token_ids);
         all_token_ids.extend(vec![0u32; max_len - token_ids.len()]);
         all_attention_masks.extend(attention_mask);
         all_attention_masks.extend(vec![0u32; max_len - attention_mask.len()]);
      }

      let token_ids_tensor = Tensor::new(&all_token_ids[..], &self.device)
         .map_err(EmbeddingError::CreateTensor)?
         .reshape(&[batch_size, max_len])
         .map_err(EmbeddingError::Reshape)?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(EmbeddingError::CreateMask)?
         .reshape(&[batch_size, max_len])
         .map_err(EmbeddingError::Reshape)?;

      let Models(dense, _) = self.models().await?;

      let embeddings = dense
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(EmbeddingError::ForwardPass)?;

      let all_embeddings: Vec<Vec<Vec<f32>>> =
         embeddings.to_vec3().map_err(EmbeddingError::Convert)?;

      let mut results = Vec::with_capacity(batch_size);
      for batch_emb in &all_embeddings {
         let mut dense_vec = batch_emb[0].clone();
         Self::normalize_l2(&mut dense_vec);
         results.push(dense_vec);
      }

      Ok(results)
   }

   async fn compute_colbert_embedding(&self, text: &str) -> Result<Vec<Vec<f32>>> {
      let (token_ids, attention_mask) = self.tokenize_colbert(text).await?;
      let seq_len = token_ids.len();

      let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)
         .map_err(EmbeddingError::CreateTensor)?
         .unsqueeze(0)
         .map_err(EmbeddingError::Unsqueeze)?;

      let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
         .map_err(EmbeddingError::CreateMask)?
         .unsqueeze(0)
         .map_err(EmbeddingError::Unsqueeze)?;

      let Models(_, colbert) = self.models().await?;

      let embeddings = colbert
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(EmbeddingError::ForwardPass)?;
      let projected = colbert
         .projection
         .forward(&embeddings)
         .map_err(EmbeddingError::Projection)?;

      let batch_embeddings = projected.get(0).map_err(EmbeddingError::GetBatch)?;

      let mut token_embeddings = Vec::with_capacity(seq_len);
      for i in 0..seq_len {
         let token_emb = batch_embeddings
            .get(i)
            .map_err(|e| EmbeddingError::ExtractToken { i, e })?;

         let mut vec: Vec<f32> = token_emb
            .to_vec1()
            .map_err(|e| EmbeddingError::ConvertToken { i, e })?;

         Self::normalize_l2(&mut vec);
         token_embeddings.push(vec);
      }

      Ok(token_embeddings)
   }

   async fn compute_colbert_embeddings_batch_inner(
      &self,
      tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<Vec<Vec<Vec<f32>>>> {
      if tokenized.is_empty() {
         return Ok(Vec::new());
      }

      let max_len = tokenized
         .iter()
         .map(|(ids, _)| ids.len())
         .max()
         .unwrap_or(0);
      let batch_size = tokenized.len();

      let mut all_token_ids = Vec::with_capacity(batch_size * max_len);
      let mut all_attention_masks = Vec::with_capacity(batch_size * max_len);

      for (token_ids, attention_mask) in tokenized {
         all_token_ids.extend(token_ids);
         all_token_ids.extend(vec![0u32; max_len - token_ids.len()]);
         all_attention_masks.extend(attention_mask);
         all_attention_masks.extend(vec![0u32; max_len - attention_mask.len()]);
      }

      let token_ids_tensor = Tensor::new(&all_token_ids[..], &self.device)
         .map_err(EmbeddingError::CreateTensor)?
         .reshape(&[batch_size, max_len])
         .map_err(EmbeddingError::Reshape)?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(EmbeddingError::CreateMask)?
         .reshape(&[batch_size, max_len])
         .map_err(EmbeddingError::Reshape)?;

      let Models(_, colbert) = self.models().await?;

      let embeddings = colbert
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(EmbeddingError::ForwardPass)?;

      let projected = colbert
         .projection
         .forward(&embeddings)
         .map_err(EmbeddingError::Projection)?;

      let projected_f32 = projected
         .to_dtype(DType::F32)
         .map_err(EmbeddingError::DtypeConversion)?;

      let all_embeddings: Vec<Vec<Vec<f32>>> =
         projected_f32.to_vec3().map_err(EmbeddingError::Convert)?;

      let mut results = Vec::with_capacity(batch_size);
      for (i, batch_emb) in all_embeddings.into_iter().enumerate() {
         let seq_len = tokenized[i].0.len();
         let mut token_embeddings = Vec::with_capacity(seq_len);

         for mut vec in batch_emb.into_iter().take(seq_len) {
            Self::normalize_l2(&mut vec);
            token_embeddings.push(vec);
         }

         results.push(token_embeddings);
      }

      Ok(results)
   }

   async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>> {
      if texts.is_empty() {
         return Ok(Vec::new());
      }

      let mut current_batch_size = self.adaptive_batch_size.load(Ordering::Relaxed);
      let mut all_results = Vec::with_capacity(texts.len());
      let mut offset = 0;

      while offset < texts.len() {
         let end = (offset + current_batch_size).min(texts.len());
         let batch_texts = &texts[offset..end];

         let dense_tokenized = self.tokenize_dense_batch(batch_texts).await?;
         let colbert_tokenized = self.tokenize_colbert_batch(batch_texts).await?;

         match self
            .try_compute_batch(&dense_tokenized, &colbert_tokenized)
            .await
         {
            Ok((dense_embeddings, colbert_embeddings)) => {
               for i in 0..batch_texts.len() {
                  let dense = dense_embeddings[i].clone();
                  let colbert_tokens = &colbert_embeddings[i];
                  let (colbert, colbert_scale) = Self::quantize_embeddings(colbert_tokens);
                  all_results.push(HybridEmbedding { dense, colbert, colbert_scale });
               }
               offset = end;
            },
            Err(e) => {
               let err_str = e.to_string();
               if is_oom_error(&err_str) && current_batch_size > MIN_BATCH_SIZE {
                  current_batch_size = self.reduce_batch_size();
               } else {
                  return Err(e);
               }
            },
         }
      }

      Ok(all_results)
   }

   async fn try_compute_batch(
      &self,
      dense_tokenized: &[(Vec<u32>, Vec<u32>)],
      colbert_tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>)> {
      let dense_embeddings = self
         .compute_dense_embeddings_batch_inner(dense_tokenized)
         .await?;
      let colbert_embeddings = self
         .compute_colbert_embeddings_batch_inner(colbert_tokenized)
         .await?;
      Ok((dense_embeddings, colbert_embeddings))
   }
}

#[async_trait::async_trait]
impl Embedder for CandleEmbedder {
   async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>> {
      Self::compute_hybrid(self, texts).await
   }

   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding> {
      let cfg = config::get();
      let query_text = if cfg.query_prefix.is_empty() {
         text.to_string()
      } else {
         format!("{}{}", cfg.query_prefix, text)
      };

      if cfg.debug_embed {
         tracing::info!("encoding query: {:?}", text);
      }

      let dense = self.compute_dense_embedding(&query_text).await?;
      let colbert = self.compute_colbert_embedding(&query_text).await?;

      if cfg.debug_embed {
         tracing::info!(
            "query embedding - dense_dim: {}, colbert_tokens: {}",
            dense.len(),
            colbert.len()
         );
      }

      Ok(QueryEmbedding { dense, colbert })
   }

   fn is_ready(&self) -> bool {
      self.models.get().is_some()
   }
}

impl Default for CandleEmbedder {
   fn default() -> Self {
      Self::new().expect("failed to create CandleEmbedder")
   }
}
