//! `ColBERT` `MaxSim` scoring with SIMD optimization and quantized inference.

use std::simd::{
   Simd,
   cmp::SimdPartialEq,
   f32x8,
   num::{SimdFloat, SimdInt},
};

use ndarray::Array2;

/// Computes `MaxSim` score between query and document token matrices.
///
/// For each query token, finds the maximum dot product with any document token
/// and sums these maxima.
pub fn max_sim(query: &Array2<f32>, doc: &Array2<f32>) -> f32 {
   if query.is_empty() || doc.is_empty() {
      return 0.0;
   }

   debug_assert_eq!(query.ncols(), doc.ncols(), "dimension mismatch");

   let mut total_score = 0.0;

   for q_idx in 0..query.nrows() {
      let q_view = query.row(q_idx);
      let q_row = q_view.as_slice().unwrap();
      let mut max_dot = f32::NEG_INFINITY;

      for d_idx in 0..doc.nrows() {
         let d_view = doc.row(d_idx);
         let d_row = d_view.as_slice().unwrap();
         let dot = dot_product_simd(q_row, d_row);
         max_dot = max_dot.max(dot);
      }

      total_score += max_dot;
   }

   total_score
}

/// Dequantizes int8 `ColBERT` embeddings to float32 matrix, skipping padding
/// tokens.
///
/// Converts quantized bytes using: `byte as i8 * scale`. All-zero tokens are
/// treated as padding and excluded.
pub fn dequantize_colbert(quantized: &[u8], scale: f64, dim: usize) -> Array2<f32> {
   if quantized.is_empty() || dim == 0 {
      return Array2::default((0, 0));
   }

   let valid_len = (quantized.len() / dim) * dim;
   let quantized = &quantized[..valid_len];
   let scale_f32 = scale as f32;
   let num_tokens = valid_len / dim;

   let mut data: Box<[f32]> = vec![0.0f32; valid_len].into_boxed_slice();
   let mut write_token = 0;

   for token_idx in 0..num_tokens {
      let src_start = token_idx * dim;
      let dst_start = write_token * dim;
      let mut is_padding = true;

      for i in 0..dim {
         let b = quantized[src_start + i];
         let val = (b as i8 as f32) * scale_f32;
         data[dst_start + i] = val;
         if val != 0.0 {
            is_padding = false;
         }
      }

      if !is_padding {
         write_token += 1;
      }
   }

   if write_token < num_tokens {
      let truncated_len = write_token * dim;
      let mut vec = data.into_vec();
      vec.truncate(truncated_len);
      data = vec.into_boxed_slice();
   }

   if write_token == 0 {
      return Array2::default((0, dim));
   }

   Array2::from_shape_vec((write_token, dim), data.into_vec())
      .expect("data length must match shape")
}

/// Computes `MaxSim` score directly on quantized document embeddings without
/// dequantization.
///
/// Streams dot products on-the-fly with SIMD acceleration, skipping all-zero
/// padding tokens.
pub fn max_sim_quantized(
   query: &Array2<f32>,
   doc_quantized: &[u8],
   doc_scale: f64,
   dim: usize,
) -> f32 {
   if query.is_empty() || doc_quantized.is_empty() || dim == 0 {
      return 0.0;
   }

   debug_assert_eq!(query.ncols(), dim, "dimension mismatch");

   let valid_len = (doc_quantized.len() / dim) * dim;
   let doc_quantized = &doc_quantized[..valid_len];
   let num_doc_tokens = valid_len / dim;
   let scale_factor = doc_scale as f32;
   const LANES: usize = 8;
   let chunk_count = dim / LANES;
   let remainder = dim % LANES;
   let mut q_chunks: Box<[Simd<f32, LANES>]> =
      vec![Simd::splat(0.0); chunk_count].into_boxed_slice();
   let scale_simd = Simd::splat(scale_factor);
   let zero_i8 = Simd::<i8, LANES>::splat(0);

   let mut total_score = 0.0;
   let mut d_buf = [0i8; LANES];

   for q_idx in 0..query.nrows() {
      let q_view = query.row(q_idx);
      let q_row = q_view.as_slice().unwrap();
      let mut max_dot = f32::NEG_INFINITY;
      let mut found_token = false;

      for (chunk_idx, slot) in q_chunks.iter_mut().enumerate() {
         let offset = chunk_idx * LANES;
         *slot = Simd::<f32, LANES>::from_slice(&q_row[offset..offset + LANES]);
      }
      let q_tail = &q_row[chunk_count * LANES..];

      for d_idx in 0..num_doc_tokens {
         let d_start = d_idx * dim;
         let d_slice = &doc_quantized[d_start..d_start + dim];

         let mut is_padding = true;
         let mut dot = 0.0f32;

         for (chunk_idx, q_chunk) in q_chunks.iter().enumerate() {
            let offset = chunk_idx * LANES;
            for lane in 0..LANES {
               d_buf[lane] = d_slice[offset + lane] as i8;
            }
            let d_chunk = Simd::<i8, LANES>::from_array(d_buf);
            if d_chunk.simd_ne(zero_i8).any() {
               is_padding = false;
            }
            let d_f32 = d_chunk.cast::<f32>() * scale_simd;
            dot += (*q_chunk * d_f32).reduce_sum();
         }

         if remainder > 0 {
            let base = chunk_count * LANES;
            for i in 0..remainder {
               let quantized_val = d_slice[base + i] as i8 as f32;
               if quantized_val != 0.0 {
                  is_padding = false;
               }
               dot += q_tail[i] * quantized_val * scale_factor;
            }
         }

         if !is_padding {
            found_token = true;
            max_dot = max_dot.max(dot);
         }
      }

      if found_token {
         total_score += max_dot;
      }
   }

   total_score
}

/// Dequantizes int8 `ColBERT` embeddings into a reusable scratch buffer.
///
/// Clears and reuses the provided buffer to avoid allocation. Returns number of
/// non-padding tokens written.
pub fn dequantize_colbert_scratch(
   quantized: &[u8],
   scale: f64,
   dim: usize,
   scratch: &mut Vec<f32>,
) -> usize {
   scratch.clear();

   if quantized.is_empty() || dim == 0 {
      return 0;
   }

   let valid_len = (quantized.len() / dim) * dim;
   let quantized = &quantized[..valid_len];
   let scale_f32 = scale as f32;
   let num_tokens = valid_len / dim;

   scratch.reserve(valid_len);

   let mut write_token = 0;

   for token_idx in 0..num_tokens {
      let src_start = token_idx * dim;
      let mut is_padding = true;

      let token_start_idx = scratch.len();
      for i in 0..dim {
         let b = quantized[src_start + i];
         let val = (b as i8 as f32) * scale_f32;
         scratch.push(val);
         if val != 0.0 {
            is_padding = false;
         }
      }

      if is_padding {
         scratch.truncate(token_start_idx);
      } else {
         write_token += 1;
      }
   }

   write_token
}

/// SIMD-optimized dot product using 8-lane vectors.
/// Falls back to scalar for tail elements that don't fit in a full SIMD lane.
#[inline]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
   debug_assert_eq!(a.len(), b.len(), "dot product requires equal-length vectors");

   const LANES: usize = 8;
   let len = a.len();
   let chunks = len / LANES;
   let remainder = len % LANES;

   let mut sum_vec = f32x8::splat(0.0);

   // SIMD vectorized portion
   for i in 0..chunks {
      let offset = i * LANES;
      let a_chunk = f32x8::from_slice(&a[offset..offset + LANES]);
      let b_chunk = f32x8::from_slice(&b[offset..offset + LANES]);
      sum_vec += a_chunk * b_chunk;
   }

   let mut sum = sum_vec.reduce_sum();

   // Scalar tail for remaining elements
   if remainder > 0 {
      let offset = chunks * LANES;
      for i in 0..remainder {
         sum += a[offset + i] * b[offset + i];
      }
   }

   sum
}

#[cfg(test)]
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
   debug_assert_eq!(a.len(), b.len(), "dot product requires equal-length vectors");
   a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
   use super::*;

   fn quantize_for_test(doc: &Array2<f32>) -> (Vec<u8>, f64) {
      let values = doc.as_slice().expect("matrix must be contiguous");
      if values.is_empty() {
         return (Vec::new(), 1.0);
      }

      let max_val = values.iter().fold(0.0f32, |max, &v| max.max(v.abs()));
      if max_val == 0.0 {
         return (vec![0; values.len()], 1.0);
      }

      let scale = max_val as f64 / 127.0;
      let inv_max = 127.0 / max_val;
      let quantized = values
         .iter()
         .map(|&v| {
            let scaled = (v * inv_max).round().clamp(-127.0, 127.0);
            scaled as i8 as u8
         })
         .collect();

      (quantized, scale)
   }

   fn matrix(data: Vec<f32>, dim: usize) -> Array2<f32> {
      let rows = data.len() / dim;
      Array2::from_shape_vec((rows, dim), data).unwrap()
   }

   #[test]
   fn test_dot_product() {
      let a = vec![1.0, 2.0, 3.0];
      let b = vec![4.0, 5.0, 6.0];
      let result = dot_product(&a, &b);
      assert!((result - 32.0).abs() < 1e-6);
   }

   #[test]
   fn test_dot_product_simd() {
      let a = vec![1.0, 2.0, 3.0];
      let b = vec![4.0, 5.0, 6.0];
      let result = dot_product_simd(&a, &b);
      assert!((result - 32.0).abs() < 1e-6);

      let a96: Vec<f32> = (0..96).map(|i| i as f32 * 0.1).collect();
      let b96: Vec<f32> = (0..96).map(|i| (95 - i) as f32 * 0.1).collect();
      let scalar_result = dot_product(&a96, &b96);
      let simd_result = dot_product_simd(&a96, &b96);
      let diff96 = (scalar_result - simd_result).abs();
      assert!(diff96 < 2e-4, "96-dim diff: {diff96}");

      let a13: Vec<f32> = (0..13).map(|i| i as f32).collect();
      let b13: Vec<f32> = (0..13).map(|i| (i + 1) as f32).collect();
      let scalar_result = dot_product(&a13, &b13);
      let simd_result = dot_product_simd(&a13, &b13);
      assert!((scalar_result - simd_result).abs() < 1e-4);
   }

   #[test]
   fn test_max_sim() {
      let query = matrix(vec![1.0, 0.0, 0.0, 1.0], 2);
      let doc = matrix(vec![0.9, 0.1, 0.1, 0.9], 2);
      let score = max_sim(&query, &doc);
      assert!((score - 1.8).abs() < 1e-6);
   }

   #[test]
   fn test_dequantize_colbert() {
      let quantized = vec![127, 0, -127i8 as u8, 64];
      let scale = 2.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;
      let result = dequantize_colbert(&quantized, scale, dim);

      assert_eq!(result.nrows(), 2);
      assert!((result[[0, 0]] - 2.0).abs() < 1e-6); // 127 * (2.0/127.0) = 2.0
      assert!((result[[0, 1]] - 0.0).abs() < 1e-6);
      assert!((result[[1, 0]] - (-2.0)).abs() < 1e-6); // -127 * (2.0/127.0) = -2.0
      assert!((result[[1, 1]] - (64.0 * 2.0 / 127.0)).abs() < 1e-6);
   }

   #[test]
   fn test_dequantize_colbert_with_padding() {
      let quantized = vec![127, 64, 0, 0];
      let scale = 1.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;
      let result = dequantize_colbert(&quantized, scale, dim);

      assert_eq!(result.nrows(), 1);
      assert!((result[[0, 0]] - 1.0).abs() < 1e-6); // 127 * (1.0/127.0) = 1.0
   }

   #[test]
   fn test_matrix_memory_layout() {
      let m = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
      assert_eq!(m.nrows(), 2);
      assert_eq!(m.ncols(), 3);
      assert_eq!(m.row(0).as_slice().unwrap(), &[1.0, 2.0, 3.0]);
      assert_eq!(m.row(1).as_slice().unwrap(), &[4.0, 5.0, 6.0]);
   }

   #[test]
   fn test_max_sim_quantized() {
      let query = matrix(vec![1.0, 0.0, 0.0, 1.0], 2);
      let quantized = vec![127, 0, 0, 127];
      let scale = 1.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;
      let score = max_sim_quantized(&query, &quantized, scale, dim);
      // dot([1,0], [1,0]) + dot([0,1], [0,1]) = 1 + 1 = 2
      assert!((score - 2.0).abs() < 1e-4);
   }

   #[test]
   fn test_max_sim_quantized_vs_dequantized() {
      let quantized = vec![127, 0, -127i8 as u8, 64];
      let scale = 2.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;

      let query = matrix(vec![1.0, 0.0, 0.0, 1.0], 2);

      let dequantized_doc = dequantize_colbert(&quantized, scale, dim);
      let score_deq = max_sim(&query, &dequantized_doc);

      let score_quantized = max_sim_quantized(&query, &quantized, scale, dim);

      assert!((score_deq - score_quantized).abs() < 1e-4);
   }

   #[test]
   fn test_quantized_pipeline_matches_float_score() {
      let dim = 2;
      let query = matrix(vec![0.8, -0.1, 0.2, 0.9], dim);
      let doc = matrix(vec![0.5, -0.25, 0.2, 0.8], dim);

      let (doc_quantized, scale) = quantize_for_test(&doc);

      let float_score = max_sim(&query, &doc);
      let quantized_score = max_sim_quantized(&query, &doc_quantized, scale, dim);

      assert!(
         (float_score - quantized_score).abs() < 1e-2,
         "float {float_score} vs quantized {quantized_score}"
      );
   }

   #[test]
   fn test_max_sim_quantized_with_padding() {
      let query = matrix(vec![1.0, 0.5], 2);
      let quantized = vec![127, 64, 0, 0, 100, 50];
      let scale = 1.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;

      let score = max_sim_quantized(&query, &quantized, scale, dim);

      // doc1: [127*scale, 64*scale] = [1.0, 64/127], doc3: [100*scale, 50*scale] =
      // [100/127, 50/127]
      let expected_doc1: f32 = 1.0f32.mul_add(1.0, (64.0 / 127.0) * 0.5);
      let expected_doc3: f32 = (100.0f32 / 127.0f32).mul_add(1.0f32, (50.0f32 / 127.0f32) * 0.5f32);

      assert!((score - expected_doc1.max(expected_doc3)).abs() < 1e-4);
   }

   #[test]
   fn test_dequantize_colbert_scratch() {
      let quantized = vec![127, 0, -127i8 as u8, 64];
      let scale = 2.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;
      let mut scratch = Vec::with_capacity(quantized.len());

      let num_tokens = dequantize_colbert_scratch(&quantized, scale, dim, &mut scratch);

      assert_eq!(num_tokens, 2);
      assert_eq!(scratch.len(), 4);
      assert!((scratch[0] - 2.0).abs() < 1e-6); // 127 * (2.0/127.0) = 2.0
      assert!((scratch[1] - 0.0).abs() < 1e-6);
      assert!((scratch[2] - (-2.0)).abs() < 1e-6); // -127 * (2.0/127.0) = -2.0
      assert!((scratch[3] - (64.0 * 2.0 / 127.0)).abs() < 1e-6);
   }

   #[test]
   fn test_dequantize_colbert_scratch_reuse() {
      let quantized1 = vec![127, 64];
      let quantized2 = vec![100, 50, -127i8 as u8, -64i8 as u8];
      let scale = 1.0;
      let dim = 2;
      let mut scratch = Vec::with_capacity(quantized2.len());

      dequantize_colbert_scratch(&quantized1, scale, dim, &mut scratch);
      assert_eq!(scratch.len(), 2);
      let cap_after_first = scratch.capacity();

      dequantize_colbert_scratch(&quantized2, scale, dim, &mut scratch);
      assert_eq!(scratch.len(), 4);
      assert_eq!(scratch.capacity(), cap_after_first);
   }

   #[test]
   fn test_dequantize_colbert_scratch_with_padding() {
      let quantized = vec![127, 64, 0, 0];
      let scale = 1.0 / 127.0; // scale = max_val / 127 from quantize_embeddings
      let dim = 2;
      let mut scratch = Vec::with_capacity(quantized.len());

      let num_tokens = dequantize_colbert_scratch(&quantized, scale, dim, &mut scratch);

      assert_eq!(num_tokens, 1);
      assert_eq!(scratch.len(), 2);
      assert!((scratch[0] - 1.0).abs() < 1e-6); // 127 * (1.0/127.0) = 1.0
   }
}
