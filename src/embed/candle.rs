//! Candle-based embedding implementation using BERT models
//!
//! Provides GPU-accelerated text embedding using the Candle ML framework
//! with adaptive batching and automatic model management.

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
use candle_transformers::models::{
   bert::{BertModel, Config as BertConfig},
   modernbert::{Config as ModernBertConfig, ModernBert},
};
use hf_hub::{Repo, RepoType, api::tokio::Api};
use ndarray::Array2;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::{
   Str, config,
   embed::{Embedder, HybridEmbedding, QueryEmbedding},
   error::Result,
};

const MIN_BATCH_SIZE: usize = 1;

#[derive(Debug)]
pub struct Models(DenseModelState, ColbertModelState);

/// Candle-based embedder with GPU support and adaptive batching
///
/// Manages both dense and `ColBERT` models with lazy initialization
/// and automatic batch size reduction on OOM errors.
#[derive(Debug)]
pub struct CandleEmbedder {
   models:              OnceLock<Models>,
   init_lock:           Mutex<()>,
   device:              Device,
   adaptive_batch_size: AtomicUsize,
}

/// Model backend trait supporting BERT and `ModernBERT` architectures
trait DenseModelBackend: Send + Sync + fmt::Debug {
   fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor>;
   fn uses_mean_pooling(&self) -> bool;
}

struct BertBackend {
   model:  BertModel,
   device: Device,
}

impl fmt::Debug for BertBackend {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("BertBackend")
         .field("device", &self.device)
         .finish_non_exhaustive()
   }
}

impl DenseModelBackend for BertBackend {
   fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
      let token_type_ids = input_ids.zeros_like()?;
      self
         .model
         .forward(input_ids, &token_type_ids, Some(attention_mask))
   }

   fn uses_mean_pooling(&self) -> bool {
      false
   }
}

struct ModernBertBackend {
   model:  ModernBert,
   device: Device,
}

impl fmt::Debug for ModernBertBackend {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("ModernBertBackend")
         .field("device", &self.device)
         .finish_non_exhaustive()
   }
}

impl DenseModelBackend for ModernBertBackend {
   fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
      self.model.forward(input_ids, attention_mask)
   }

   fn uses_mean_pooling(&self) -> bool {
      true
   }
}

struct DenseModelState {
   name:      &'static str,
   model:     Box<dyn DenseModelBackend>,
   tokenizer: Tokenizer,
}

impl fmt::Debug for DenseModelState {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("DenseModelState")
         .field("name", &self.name)
         .field("model", &self.model)
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

/// Errors that can occur during embedding operations
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
      || err.contains("MTLBuffer")
      || err.contains("Metal")
      || err.contains("alloc")
}

const fn optimal_dtype(_device: &Device) -> DType {
   // BF16/F16 on some CUDA setups can yield NaNs; F32 is stable across devices.
   DType::F32
}

impl CandleEmbedder {
   /// Creates a new embedder with GPU support if available
   pub fn new() -> Result<Self> {
      let cfg = config::get();
      let device = if cfg.disable_gpu {
         Device::Cpu
      } else {
         Self::select_device()
      };

      let initial_batch = cfg.batch_size();

      Ok(Self {
         models: OnceLock::new(),
         init_lock: Mutex::new(()),
         device,
         adaptive_batch_size: AtomicUsize::new(initial_batch),
      })
   }

   fn select_device() -> Device {
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      {
         if let Ok(device) = Device::new_metal(0) {
            return device;
         }
      }
      #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
      {
         if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
               return device;
            }
         }
      }
      Device::Cpu
   }

   /// Returns the current adaptive batch size
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

      let dtype = optimal_dtype(device);

      if cfg.debug_models {
         tracing::info!("loading dense model from {:?} with {:?}", model_path, dtype);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(EmbeddingError::LoadTokenizer)?;

      let config_str =
         fs::read_to_string(model_path.join("config.json")).map_err(EmbeddingError::ReadConfig)?;

      // Detect model type from config
      let model_type: Option<String> = serde_json::from_str::<serde_json::Value>(&config_str)
         .ok()
         .and_then(|v| {
            v.get("model_type")
               .and_then(|t| t.as_str().map(String::from))
         });

      // SAFETY: VarBuilder::from_mmaped_safetensors is safe to use as it properly
      // handles memory mapping
      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], dtype, device)
            .map_err(EmbeddingError::LoadWeights)?
      };

      let model: Box<dyn DenseModelBackend> = if model_type.as_deref() == Some("modernbert") {
         let config: ModernBertConfig = serde_json::from_str(&config_str)?;
         // HF granite models have tensor names like "embeddings.tok_embeddings"
         // but Candle's ModernBERT expects "model.embeddings.tok_embeddings"
         // Strip the "model." prefix when looking up tensors
         let vb = vb.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string());
         let model = ModernBert::load(vb, &config).map_err(EmbeddingError::LoadModel)?;
         if cfg.debug_models {
            tracing::info!("loaded ModernBERT dense model");
         }
         Box::new(ModernBertBackend { model, device: device.clone() })
      } else {
         // Default to BERT/RoBERTa for backward compatibility
         let config: BertConfig = serde_json::from_str(&config_str)?;
         let model = BertModel::load(vb, &config).map_err(EmbeddingError::LoadModel)?;
         if cfg.debug_models {
            tracing::info!("loaded BERT/RoBERTa dense model");
         }
         Box::new(BertBackend { model, device: device.clone() })
      };

      if cfg.debug_models {
         tracing::info!("dense model loaded");
      }

      Ok(DenseModelState { name: cfg.dense_model.as_str(), model, tokenizer })
   }

   async fn load_colbert(device: &Device) -> Result<ColbertModelState> {
      let cfg = config::get();
      let model_path = Self::download_model(&cfg.colbert_model).await?;

      let dtype = optimal_dtype(device);

      if cfg.debug_models {
         tracing::info!("loading colbert model from {:?} with {:?}", model_path, dtype);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(EmbeddingError::LoadTokenizer)?;

      let config: BertConfig = serde_json::from_str(
         &fs::read_to_string(model_path.join("config.json")).map_err(EmbeddingError::ReadConfig)?,
      )?;

      // SAFETY: VarBuilder::from_mmaped_safetensors is safe to use as it properly
      // handles memory mapping
      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[model_path.join("model.safetensors")], dtype, device)
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

   fn tokenize_impl(
      tokenizer: &Tokenizer,
      text: &str,
      max_len: usize,
   ) -> Result<(Vec<u32>, Vec<u32>)> {
      let encoding = tokenizer.encode(text, true).map_err(EmbeddingError::from)?;
      let mut token_ids = encoding.get_ids().to_vec();
      let mut attention_mask = vec![1u32; token_ids.len()];
      if token_ids.len() > max_len {
         token_ids.truncate(max_len);
         attention_mask.truncate(max_len);
      }
      Ok((token_ids, attention_mask))
   }

   async fn tokenize_dense(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
      let Models(dense, _) = self.models().await?;
      let max_len = config::get().dense_max_length;
      Self::tokenize_impl(&dense.tokenizer, text, max_len)
   }

   async fn tokenize_dense_batch(&self, texts: &[Str]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let Models(dense, _) = self.models().await?;
      let max_len = config::get().dense_max_length;
      texts
         .iter()
         .map(|text| Self::tokenize_impl(&dense.tokenizer, text.as_str(), max_len))
         .collect()
   }

   async fn tokenize_colbert(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
      let Models(_, colbert) = self.models().await?;
      let max_len = config::get().colbert_max_length;
      Self::tokenize_impl(&colbert.tokenizer, text, max_len)
   }

   async fn tokenize_colbert_batch(&self, texts: &[Str]) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let Models(_, colbert) = self.models().await?;
      let max_len = config::get().colbert_max_length;
      texts
         .iter()
         .map(|text| Self::tokenize_impl(&colbert.tokenizer, text.as_str(), max_len))
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

   fn sanitize(embeddings: &mut [f32]) {
      for v in embeddings.iter_mut() {
         if !v.is_finite() {
            *v = 0.0;
         }
      }
   }

   fn quantize_embeddings(tokens: &Array2<f32>) -> (Vec<u8>, f64) {
      if tokens.is_empty() {
         return (Vec::new(), 1.0);
      }

      let values = tokens.as_slice().expect("matrix must be contiguous");
      let mut max_val = 0.0f32;
      for &val in values {
         if val.is_finite() {
            max_val = max_val.max(val.abs());
         }
      }

      if max_val == 0.0 || !max_val.is_finite() {
         return (vec![0; values.len()], 1.0);
      }

      let scale = max_val as f64 / 127.0;
      let inv_max = 127.0 / max_val;

      let mut quantized = Vec::with_capacity(values.len());
      quantized.extend(values.iter().map(|&x| (x * inv_max) as i8 as u8));

      (quantized, scale)
   }

   fn bucket_by_length(lengths: &[usize], bucket_size: usize) -> Vec<Vec<usize>> {
      if lengths.is_empty() {
         return Vec::new();
      }

      let mut pairs: Vec<(usize, usize)> = lengths
         .iter()
         .enumerate()
         .map(|(idx, &len)| (len.div_ceil(bucket_size), idx))
         .collect();

      pairs.sort_by_key(|(bucket, _)| *bucket);

      let mut buckets = Vec::new();
      let mut current_bucket = None;

      for (bucket, idx) in pairs {
         if current_bucket != Some(bucket) {
            buckets.push(Vec::new());
            current_bucket = Some(bucket);
         }
         buckets.last_mut().expect("bucket just pushed").push(idx);
      }

      buckets
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
         .model
         .forward(&token_ids_tensor, &attention_mask_tensor)
         .map_err(EmbeddingError::ForwardPass)?;

      let pooled = if dense.model.uses_mean_pooling() {
         // Mean pooling: average over non-padding tokens
         let batch_emb = embeddings.get(0).map_err(EmbeddingError::GetBatch)?;
         let mask_f32 = attention_mask_tensor
            .get(0)
            .map_err(EmbeddingError::GetBatch)?
            .to_dtype(DType::F32)
            .map_err(EmbeddingError::DtypeConversion)?
            .unsqueeze(1)
            .map_err(EmbeddingError::Unsqueeze)?;
         let masked = batch_emb
            .broadcast_mul(&mask_f32)
            .map_err(EmbeddingError::ForwardPass)?;
         let sum = masked.sum(0).map_err(EmbeddingError::ForwardPass)?;
         let count = mask_f32.sum_all().map_err(EmbeddingError::ForwardPass)?;
         sum.broadcast_div(&count)
            .map_err(EmbeddingError::ForwardPass)?
      } else {
         // CLS pooling: take first token
         embeddings
            .get(0)
            .map_err(EmbeddingError::GetBatch)?
            .get(0)
            .map_err(EmbeddingError::ExtractCls)?
      };

      let mut dense_vec: Vec<f32> = pooled
         .to_dtype(DType::F32)
         .map_err(EmbeddingError::DtypeConversion)?
         .to_device(&Device::Cpu)
         .map_err(EmbeddingError::Convert)?
         .to_vec1()
         .map_err(EmbeddingError::ConvertToVec)?;

      Self::normalize_l2(&mut dense_vec);
      Ok(dense_vec)
   }

   async fn compute_dense_embeddings_batch_inner(
      &self,
      indices: &[usize],
      tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<Array2<f32>> {
      if indices.is_empty() {
         return Ok(Array2::default((0, 0)));
      }

      let max_len = indices
         .iter()
         .map(|&i| tokenized[i].0.len())
         .max()
         .unwrap_or(0);
      let batch_size = indices.len();

      let mut all_token_ids = Vec::with_capacity(batch_size * max_len);
      let mut all_attention_masks = Vec::with_capacity(batch_size * max_len);

      for &idx in indices {
         let (token_ids, attention_mask) = &tokenized[idx];
         all_token_ids.extend(token_ids);
         all_token_ids.extend(std::iter::repeat_n(0u32, max_len - token_ids.len()));
         all_attention_masks.extend(attention_mask);
         all_attention_masks.extend(std::iter::repeat_n(0u32, max_len - attention_mask.len()));
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
         .model
         .forward(&token_ids_tensor, &attention_mask_tensor)
         .map_err(EmbeddingError::ForwardPass)?;

      let dim = config::get().dense_dim;
      let pooled = if dense.model.uses_mean_pooling() {
         // Mean pooling: average over non-padding tokens for each batch item
         // embeddings: [batch, seq_len, dim]
         // attention_mask: [batch, seq_len]
         let mask_f32 = attention_mask_tensor
            .to_dtype(DType::F32)
            .map_err(EmbeddingError::DtypeConversion)?
            .unsqueeze(2)
            .map_err(EmbeddingError::Unsqueeze)?;
         let masked = embeddings
            .broadcast_mul(&mask_f32)
            .map_err(EmbeddingError::ForwardPass)?;
         let sum = masked.sum(1).map_err(EmbeddingError::ForwardPass)?;
         let count = mask_f32
            .sum(1)
            .map_err(EmbeddingError::ForwardPass)?
            .clamp(1.0, f64::MAX)
            .map_err(EmbeddingError::ForwardPass)?;
         sum.broadcast_div(&count)
            .map_err(EmbeddingError::ForwardPass)?
      } else {
         // CLS pooling: take first token from each sequence
         embeddings
            .narrow(1, 0, 1)
            .map_err(EmbeddingError::ExtractCls)?
            .squeeze(1)
            .map_err(EmbeddingError::ExtractCls)?
      };

      let mut flat: Vec<f32> = pooled
         .to_dtype(DType::F32)
         .map_err(EmbeddingError::DtypeConversion)?
         .to_device(&Device::Cpu)
         .map_err(EmbeddingError::Convert)?
         .flatten_all()
         .map_err(EmbeddingError::Convert)?
         .to_vec1()
         .map_err(EmbeddingError::Convert)?;

      for batch_idx in 0..batch_size {
         let start = batch_idx * dim;
         Self::sanitize(&mut flat[start..start + dim]);
         Self::normalize_l2(&mut flat[start..start + dim]);
      }

      Ok(Array2::from_shape_vec((batch_size, dim), flat).expect("shape matches data"))
   }

   async fn compute_colbert_embedding(&self, text: &str) -> Result<Array2<f32>> {
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

      // RoBERTa models don't use token_type_ids, so pass zeros
      let token_type_ids = token_ids_tensor
         .zeros_like()
         .map_err(EmbeddingError::CreateMask)?;

      let Models(_, colbert) = self.models().await?;

      let embeddings = colbert
         .bert
         .forward(&token_ids_tensor, &token_type_ids, Some(&attention_mask_tensor))
         .map_err(EmbeddingError::ForwardPass)?;
      let projected = colbert
         .projection
         .forward(&embeddings)
         .map_err(EmbeddingError::Projection)?;

      let dim = config::get().colbert_dim;

      let batch_emb = projected.get(0).map_err(EmbeddingError::GetBatch)?;
      let mut data: Vec<f32> = batch_emb
         .to_dtype(DType::F32)
         .map_err(EmbeddingError::Convert)?
         .to_device(&Device::Cpu)
         .map_err(EmbeddingError::Convert)?
         .flatten_all()
         .map_err(EmbeddingError::Convert)?
         .to_vec1()
         .map_err(EmbeddingError::Convert)?;

      for i in 0..seq_len {
         Self::sanitize(&mut data[i * dim..(i + 1) * dim]);
         Self::normalize_l2(&mut data[i * dim..(i + 1) * dim]);
      }

      Ok(Array2::from_shape_vec((seq_len, dim), data).expect("shape matches data"))
   }

   async fn compute_colbert_embeddings_batch_inner(
      &self,
      indices: &[usize],
      tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<Vec<Array2<f32>>> {
      if indices.is_empty() {
         return Ok(Vec::new());
      }

      let max_len = indices
         .iter()
         .map(|&i| tokenized[i].0.len())
         .max()
         .unwrap_or(0);
      let batch_size = indices.len();

      let mut all_token_ids = Vec::with_capacity(batch_size * max_len);
      let mut all_attention_masks = Vec::with_capacity(batch_size * max_len);

      for &idx in indices {
         let (token_ids, attention_mask) = &tokenized[idx];
         all_token_ids.extend(token_ids);
         all_token_ids.extend(std::iter::repeat_n(0u32, max_len - token_ids.len()));
         all_attention_masks.extend(attention_mask);
         all_attention_masks.extend(std::iter::repeat_n(0u32, max_len - attention_mask.len()));
      }

      let token_ids_tensor = Tensor::new(&all_token_ids[..], &self.device)
         .map_err(EmbeddingError::CreateTensor)?
         .reshape(&[batch_size, max_len])
         .map_err(EmbeddingError::Reshape)?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(EmbeddingError::CreateMask)?
         .reshape(&[batch_size, max_len])
         .map_err(EmbeddingError::Reshape)?;

      // RoBERTa models don't use token_type_ids, so pass zeros
      let token_type_ids = token_ids_tensor
         .zeros_like()
         .map_err(EmbeddingError::CreateMask)?;

      let Models(_, colbert) = self.models().await?;

      let embeddings = colbert
         .bert
         .forward(&token_ids_tensor, &token_type_ids, Some(&attention_mask_tensor))
         .map_err(EmbeddingError::ForwardPass)?;

      let projected = colbert
         .projection
         .forward(&embeddings)
         .map_err(EmbeddingError::Projection)?;

      let projected_f32 = projected
         .to_dtype(DType::F32)
         .map_err(EmbeddingError::DtypeConversion)?
         .to_device(&Device::Cpu)
         .map_err(EmbeddingError::Convert)?;

      let dim = config::get().colbert_dim;
      let flat: Vec<f32> = projected_f32
         .flatten_all()
         .map_err(EmbeddingError::Convert)?
         .to_vec1()
         .map_err(EmbeddingError::Convert)?;

      let mut results = Vec::with_capacity(batch_size);
      for (i, &idx) in indices.iter().enumerate() {
         let seq_len = tokenized[idx].0.len();
         let base = i * max_len * dim;
         let end = base + seq_len * dim;
         let mut data = flat[base..end].to_vec();
         for chunk in data.chunks_mut(dim) {
            Self::normalize_l2(chunk);
         }
         results.push(Array2::from_shape_vec((seq_len, dim), data).expect("shape matches data"));
      }

      Ok(results)
   }

   async fn compute_hybrid(&self, texts: &[Str]) -> Result<Vec<HybridEmbedding>> {
      if texts.is_empty() {
         return Ok(Vec::new());
      }

      let dense_tokenized = self.tokenize_dense_batch(texts).await?;
      let colbert_tokenized = self.tokenize_colbert_batch(texts).await?;

      let combined_lengths: Vec<usize> = dense_tokenized
         .iter()
         .zip(colbert_tokenized.iter())
         .map(|((dense_ids, _), (colbert_ids, _))| dense_ids.len().max(colbert_ids.len()))
         .collect();

      let buckets = Self::bucket_by_length(&combined_lengths, 32);

      let mut results = vec![None; texts.len()];
      let mut current_batch_size = self.adaptive_batch_size.load(Ordering::Relaxed);

      for bucket_indices in &buckets {
         let mut offset = 0;

         while offset < bucket_indices.len() {
            let end = (offset + current_batch_size).min(bucket_indices.len());
            let batch_indices = &bucket_indices[offset..end];

            match self
               .try_compute_batch_indexed(batch_indices, &dense_tokenized, &colbert_tokenized)
               .await
            {
               Ok((dense_matrix, colbert_embeddings)) => {
                  for (i, &orig_idx) in batch_indices.iter().enumerate() {
                     let dense = dense_matrix.row(i).to_vec();
                     let colbert_tokens = &colbert_embeddings[i];
                     let (colbert, colbert_scale) = Self::quantize_embeddings(colbert_tokens);
                     results[orig_idx] = Some(HybridEmbedding { dense, colbert, colbert_scale });
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
      }

      Ok(results
         .into_iter()
         .map(|r| r.expect("all indices processed"))
         .collect())
   }

   async fn try_compute_batch_indexed(
      &self,
      indices: &[usize],
      dense_tokenized: &[(Vec<u32>, Vec<u32>)],
      colbert_tokenized: &[(Vec<u32>, Vec<u32>)],
   ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
      let dense_embeddings = self
         .compute_dense_embeddings_batch_inner(indices, dense_tokenized)
         .await?;
      let colbert_embeddings = self
         .compute_colbert_embeddings_batch_inner(indices, colbert_tokenized)
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
            colbert.nrows()
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
