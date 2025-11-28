pub fn max_sim(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
   let mut total_score = 0.0;

   for query_vec in query_tokens {
      let mut max_dot = f32::NEG_INFINITY;

      for doc_vec in doc_tokens {
         let dot = dot_product(query_vec, doc_vec);
         if dot > max_dot {
            max_dot = dot;
         }
      }

      total_score += max_dot;
   }

   total_score
}

pub fn dequantize_colbert(quantized: &[u8], scale: f64, dim: usize) -> Vec<Vec<f32>> {
   if quantized.is_empty() || dim == 0 {
      return Vec::new();
   }

   let seq_len = quantized.len() / dim;
   let mut result = Vec::with_capacity(seq_len);

   for chunk in quantized.chunks(dim) {
      let mut is_padding = true;
      let token: Vec<f32> = chunk
         .iter()
         .map(|&b| {
            let val = (b as i8 as f32 / 127.0) * scale as f32;
            if val != 0.0 {
               is_padding = false;
            }
            val
         })
         .collect();

      if !is_padding {
         result.push(token);
      }
   }

   result
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
   a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn test_dot_product() {
      let a = vec![1.0, 2.0, 3.0];
      let b = vec![4.0, 5.0, 6.0];
      let result = dot_product(&a, &b);
      assert!((result - 32.0).abs() < 1e-6);
   }

   #[test]
   fn test_max_sim() {
      let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
      let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
      let score = max_sim(&query, &doc);
      assert!((score - 1.8).abs() < 1e-6);
   }

   #[test]
   fn test_dequantize_colbert() {
      let quantized = vec![127, 0, -127i8 as u8, 64];
      let scale = 2.0;
      let dim = 2;
      let result = dequantize_colbert(&quantized, scale, dim);

      assert_eq!(result.len(), 2);
      assert!((result[0][0] - 2.0).abs() < 1e-6);
      assert!((result[0][1] - 0.0).abs() < 1e-6);
      assert!((result[1][0] - (-2.0)).abs() < 1e-6);
      assert!((result[1][1] - (64.0 / 127.0 * 2.0)).abs() < 1e-6);
   }

   #[test]
   fn test_dequantize_colbert_with_padding() {
      let quantized = vec![127, 64, 0, 0];
      let scale = 1.0;
      let dim = 2;
      let result = dequantize_colbert(&quantized, scale, dim);

      assert_eq!(result.len(), 1);
      assert!((result[0][0] - 1.0).abs() < 1e-6);
   }
}
