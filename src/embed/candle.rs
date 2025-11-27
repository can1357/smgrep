use std::{path::PathBuf, sync::Arc};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use parking_lot::RwLock;
use tokenizers::Tokenizer;

use crate::{
   config::{DENSE_MODEL, QUERY_PREFIX, debug_embed, debug_models},
   embed::{Embedder, HybridEmbedding, QueryEmbedding},
   error::{Result, RsgrepError},
};

const MAX_SEQ_LEN_DENSE: usize = 256;
const MAX_SEQ_LEN_COLBERT: usize = 512;

pub struct CandleEmbedder {
   model:  Arc<RwLock<Option<ModelState>>>,
   device: Device,
}

struct ModelState {
   bert:      BertModel,
   tokenizer: Tokenizer,
}

impl CandleEmbedder {
   pub fn new() -> Result<Self> {
      let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

      Ok(Self { model: Arc::new(RwLock::new(None)), device })
   }

   fn ensure_model_loaded(&self) -> Result<()> {
      if self.model.read().is_some() {
         return Ok(());
      }

      let mut guard = self.model.write();
      if guard.is_some() {
         return Ok(());
      }

      let (bert, tokenizer) = Self::load_model(&self.device)?;
      *guard = Some(ModelState { bert, tokenizer });
      Ok(())
   }

   fn load_model(device: &Device) -> Result<(BertModel, Tokenizer)> {
      let model_path = Self::download_model()?;

      if debug_models() {
         tracing::info!("loading model from {:?}, device: {:?}", model_path, device);
      }

      let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
         .map_err(|e| RsgrepError::Embedding(format!("failed to load tokenizer: {}", e)))?;

      let config_path = model_path.join("config.json");
      let config: BertConfig = serde_json::from_str(
         &std::fs::read_to_string(&config_path)
            .map_err(|e| RsgrepError::Embedding(format!("failed to read config: {}", e)))?,
      )
      .map_err(|e| RsgrepError::Embedding(format!("failed to parse config: {}", e)))?;

      let weights_path = model_path.join("model.safetensors");
      let vb = unsafe {
         VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, device)
            .map_err(|e| RsgrepError::Embedding(format!("failed to load weights: {}", e)))?
      };

      let bert = BertModel::load(vb, &config)
         .map_err(|e| RsgrepError::Embedding(format!("failed to load model: {}", e)))?;

      if debug_models() {
         tracing::info!("model loaded successfully");
      }

      Ok((bert, tokenizer))
   }

   fn download_model() -> Result<PathBuf> {
      let cache_dir = crate::config::model_dir();
      std::fs::create_dir_all(&cache_dir)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create model cache: {}", e)))?;

      let api = Api::new()
         .map_err(|e| RsgrepError::Embedding(format!("failed to initialize hf_hub API: {}", e)))?;

      let repo = api.repo(Repo::new(DENSE_MODEL.to_string(), RepoType::Model));

      let model_files = ["config.json", "tokenizer.json", "model.safetensors"];
      let mut paths = Vec::new();

      for filename in &model_files {
         let path = repo.get(filename).map_err(|e| {
            RsgrepError::Embedding(format!(
               "failed to download {}: {}. Run 'rsgrep setup' to download models.",
               filename, e
            ))
         })?;
         paths.push(path);
      }

      paths[0]
         .parent()
         .ok_or_else(|| RsgrepError::Embedding("invalid model path".to_string()))
         .map(|p| p.to_path_buf())
   }

   fn tokenize(&self, text: &str, max_len: usize) -> Result<(Vec<u32>, Vec<u32>)> {
      let model_state = self.model.read();
      let state = model_state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("model not loaded".to_string()))?;

      let encoding = state
         .tokenizer
         .encode(text, true)
         .map_err(|e| RsgrepError::Embedding(format!("tokenization failed: {}", e)))?;

      let mut token_ids = encoding.get_ids().to_vec();
      let mut attention_mask = vec![1u32; token_ids.len()];

      if token_ids.len() > max_len {
         token_ids.truncate(max_len);
         attention_mask.truncate(max_len);
      }

      Ok((token_ids, attention_mask))
   }

   fn tokenize_batch(&self, texts: &[String], max_len: usize) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
      let model_state = self.model.read();
      let state = model_state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("model not loaded".to_string()))?;

      texts
         .iter()
         .map(|text| {
            let encoding = state
               .tokenizer
               .encode(text.as_str(), true)
               .map_err(|e| RsgrepError::Embedding(format!("tokenization failed: {}", e)))?;

            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = vec![1u32; token_ids.len()];

            if token_ids.len() > max_len {
               token_ids.truncate(max_len);
               attention_mask.truncate(max_len);
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

   fn compute_dense_embedding(&self, text: &str) -> Result<Vec<f32>> {
      let (token_ids, attention_mask) = self.tokenize(text, MAX_SEQ_LEN_DENSE)?;

      let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let model_state = self.model.read();
      let state = model_state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let cls_embedding = embeddings
         .get(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to get batch: {}", e)))?
         .get(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to extract CLS: {}", e)))?;

      let mut dense_vec: Vec<f32> = cls_embedding
         .to_vec1()
         .map_err(|e| RsgrepError::Embedding(format!("failed to convert to vec: {}", e)))?;

      Self::normalize_l2(&mut dense_vec);
      Ok(dense_vec)
   }

   fn compute_dense_embeddings_batch(
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
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let model_state = self.model.read();
      let state = model_state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let mut results = Vec::with_capacity(batch_size);
      for i in 0..batch_size {
         let cls_embedding = embeddings
            .get(i)
            .map_err(|e| RsgrepError::Embedding(format!("failed to get batch item {}: {}", i, e)))?
            .get(0)
            .map_err(|e| RsgrepError::Embedding(format!("failed to extract CLS: {}", e)))?;

         let mut dense_vec: Vec<f32> = cls_embedding
            .to_vec1()
            .map_err(|e| RsgrepError::Embedding(format!("failed to convert to vec: {}", e)))?;

         Self::normalize_l2(&mut dense_vec);
         results.push(dense_vec);
      }

      Ok(results)
   }

   fn compute_colbert_embedding(&self, text: &str) -> Result<Vec<Vec<f32>>> {
      let (token_ids, attention_mask) = self.tokenize(text, MAX_SEQ_LEN_COLBERT)?;

      let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .unsqueeze(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to unsqueeze: {}", e)))?;

      let model_state = self.model.read();
      let state = model_state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let seq_len = token_ids.len();
      let mut token_embeddings = Vec::with_capacity(seq_len);

      let batch_embeddings = embeddings
         .get(0)
         .map_err(|e| RsgrepError::Embedding(format!("failed to get batch: {}", e)))?;

      for i in 0..seq_len {
         let token_emb = batch_embeddings
            .get(i)
            .map_err(|e| RsgrepError::Embedding(format!("failed to extract token {}: {}", i, e)))?;

         let mut vec: Vec<f32> = token_emb
            .to_vec1()
            .map_err(|e| RsgrepError::Embedding(format!("failed to convert token {}: {}", i, e)))?;

         Self::normalize_l2(&mut vec);
         token_embeddings.push(vec);
      }

      Ok(token_embeddings)
   }

   fn compute_colbert_embeddings_batch(
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
         .map_err(|e| RsgrepError::Embedding(format!("failed to create tensor: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let attention_mask_tensor = Tensor::new(&all_attention_masks[..], &self.device)
         .map_err(|e| RsgrepError::Embedding(format!("failed to create mask: {}", e)))?
         .reshape(&[batch_size, max_len])
         .map_err(|e| RsgrepError::Embedding(format!("failed to reshape: {}", e)))?;

      let model_state = self.model.read();
      let state = model_state
         .as_ref()
         .ok_or_else(|| RsgrepError::Embedding("model not loaded".to_string()))?;

      let embeddings = state
         .bert
         .forward(&token_ids_tensor, &attention_mask_tensor, None)
         .map_err(|e| RsgrepError::Embedding(format!("forward pass failed: {}", e)))?;

      let mut results = Vec::with_capacity(batch_size);
      for i in 0..batch_size {
         let seq_len = tokenized[i].0.len();
         let mut token_embeddings = Vec::with_capacity(seq_len);

         let batch_embeddings = embeddings.get(i).map_err(|e| {
            RsgrepError::Embedding(format!("failed to get batch item {}: {}", i, e))
         })?;

         for j in 0..seq_len {
            let token_emb = batch_embeddings.get(j).map_err(|e| {
               RsgrepError::Embedding(format!(
                  "failed to extract token {} from batch {}: {}",
                  j, i, e
               ))
            })?;

            let mut vec: Vec<f32> = token_emb.to_vec1().map_err(|e| {
               RsgrepError::Embedding(format!(
                  "failed to convert token {} from batch {}: {}",
                  j, i, e
               ))
            })?;

            Self::normalize_l2(&mut vec);
            token_embeddings.push(vec);
         }

         results.push(token_embeddings);
      }

      Ok(results)
   }
}

#[async_trait::async_trait]
impl Embedder for CandleEmbedder {
   async fn compute_hybrid(&self, texts: &[String]) -> Result<Vec<HybridEmbedding>> {
      self.ensure_model_loaded()?;

      if texts.is_empty() {
         return Ok(Vec::new());
      }

      let dense_tokenized = self.tokenize_batch(texts, MAX_SEQ_LEN_DENSE)?;
      let colbert_tokenized = self.tokenize_batch(texts, MAX_SEQ_LEN_COLBERT)?;

      let dense_embeddings = self.compute_dense_embeddings_batch(&dense_tokenized)?;
      let colbert_embeddings = self.compute_colbert_embeddings_batch(&colbert_tokenized)?;

      let mut results = Vec::with_capacity(texts.len());
      for i in 0..texts.len() {
         let dense = dense_embeddings[i].clone();
         let colbert_tokens = &colbert_embeddings[i];
         let (colbert, colbert_scale) = Self::quantize_embeddings(colbert_tokens);

         results.push(HybridEmbedding { dense, colbert, colbert_scale });
      }

      Ok(results)
   }

   async fn encode_query(&self, text: &str) -> Result<QueryEmbedding> {
      self.ensure_model_loaded()?;

      let query_text = if QUERY_PREFIX.is_empty() {
         text.to_string()
      } else {
         format!("{}{}", QUERY_PREFIX, text)
      };

      if debug_embed() {
         tracing::info!("encoding query: {:?} (prefix: {:?})", text, QUERY_PREFIX);
      }

      let dense = self.compute_dense_embedding(&query_text)?;
      let colbert = self.compute_colbert_embedding(&query_text)?;

      if debug_embed() {
         tracing::info!("query embedding - colbert_len: {}", colbert.len());
      }

      Ok(QueryEmbedding { dense, colbert })
   }

   fn is_ready(&self) -> bool {
      self.model.read().is_some()
   }
}

impl Default for CandleEmbedder {
   fn default() -> Self {
      Self::new().expect("failed to create CandleEmbedder")
   }
}
