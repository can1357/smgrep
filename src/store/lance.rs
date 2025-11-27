use std::{
   collections::{HashMap, HashSet},
   fs,
   path::{Path, PathBuf},
   sync::Arc,
};

use arrow_array::{
   Array, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Float64Array,
   LargeBinaryArray, LargeStringArray, RecordBatch, RecordBatchReader, StringArray, UInt32Array,
   builder::{
      BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, LargeBinaryBuilder,
      LargeStringBuilder, StringBuilder, UInt32Builder,
   },
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::{
   Connection, Table, connect,
   index::{Index, scalar::FullTextSearchQuery},
   query::{ExecutableQuery, QueryBase, Select},
};
use parking_lot::RwLock;

use crate::{
   Str, config,
   error::Result,
   meta::FileHash,
   store,
   types::{ChunkType, SearchResponse, SearchResult, SearchStatus, StoreInfo, VectorRecord},
};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
   #[error("invalid database path")]
   InvalidDatabasePath,

   #[error("failed to connect to database: {0}")]
   Connect(#[source] lancedb::Error),

   #[error("failed to reopen table after migration: {0}")]
   ReopenTableAfterMigration(#[source] lancedb::Error),

   #[error("failed to create table: {0}")]
   CreateTable(#[source] lancedb::Error),

   #[error("failed to sample table for migration check: {0}")]
   SampleTableForMigration(#[source] lancedb::Error),

   #[error("failed to read sample batch: {0}")]
   ReadSampleBatch(#[source] lancedb::Error),

   #[error("failed to read existing data for migration: {0}")]
   ReadExistingDataForMigration(#[source] lancedb::Error),

   #[error("failed to collect existing batches: {0}")]
   CollectExistingBatches(#[source] lancedb::Error),

   #[error("failed to drop old table during migration: {0}")]
   DropOldTableDuringMigration(#[source] lancedb::Error),

   #[error("failed to create new table during migration: {0}")]
   CreateNewTableDuringMigration(#[source] lancedb::Error),

   #[error("failed to add migrated records: {0}")]
   AddMigratedRecords(#[source] lancedb::Error),

   #[error("failed to create empty batch: {0}")]
   CreateEmptyBatch(#[source] ArrowError),

   #[error("empty batch")]
   EmptyBatch,

   #[error("failed to create record batch: {0}")]
   CreateRecordBatch(#[source] ArrowError),

   #[error("failed to add records: {0}")]
   AddRecords(#[source] lancedb::Error),

   #[error("failed to create vector query: {0}")]
   CreateVectorQuery(#[source] lancedb::Error),

   #[error("failed to execute code search: {0}")]
   ExecuteCodeSearch(#[source] lancedb::Error),

   #[error("failed to execute doc search: {0}")]
   ExecuteDocSearch(#[source] lancedb::Error),

   #[error("failed to collect code results: {0}")]
   CollectCodeResults(#[source] lancedb::Error),

   #[error("failed to collect doc results: {0}")]
   CollectDocResults(#[source] lancedb::Error),

   #[error("missing path column")]
   MissingPathColumn,

   #[error("path column type mismatch")]
   PathColumnTypeMismatch,

   #[error("missing start_line column")]
   MissingStartLineColumn,

   #[error("start_line type mismatch")]
   StartLineTypeMismatch,

   #[error("content column type mismatch")]
   ContentColumnTypeMismatch,

   #[error("vector column type mismatch")]
   VectorColumnTypeMismatch,

   #[error("vector values type mismatch")]
   VectorValuesTypeMismatch,

   #[error("failed to delete file: {0}")]
   DeleteFile(#[source] lancedb::Error),

   #[error("failed to delete files: {0}")]
   DeleteFiles(#[source] lancedb::Error),

   #[error("failed to drop table: {0}")]
   DropTable(#[source] lancedb::Error),

   #[error("failed to count rows: {0}")]
   CountRows(#[source] lancedb::Error),

   #[error("failed to execute query: {0}")]
   ExecuteQuery(#[source] lancedb::Error),

   #[error("failed to collect results: {0}")]
   CollectResults(#[source] lancedb::Error),

   #[error("index already exists")]
   IndexAlreadyExists,

   #[error("failed to create FTS index: {0}")]
   CreateFtsIndex(#[source] lancedb::Error),

   #[error("failed to create vector index: {0}")]
   CreateVectorIndex(#[source] lancedb::Error),
}

pub enum RecordBatchOnce {
   Batch(RecordBatch),
   Taken(SchemaRef),
   __Invalid,
}

impl RecordBatchOnce {
   pub const fn new(batch: RecordBatch) -> Self {
      Self::Batch(batch)
   }

   pub fn take(&mut self) -> Result<RecordBatch, SchemaRef> {
      let prev = std::mem::replace(self, Self::__Invalid);
      match prev {
         Self::Batch(batch) => {
            *self = Self::Taken(batch.schema());
            Ok(batch)
         },
         Self::Taken(schema) => {
            *self = Self::Taken(schema.clone());
            Err(schema)
         },
         Self::__Invalid => {
            unreachable!()
         },
      }
   }
}

impl Iterator for RecordBatchOnce {
   type Item = Result<RecordBatch, ArrowError>;

   fn next(&mut self) -> Option<Self::Item> {
      self.take().ok().map(Ok)
   }
}

impl RecordBatchReader for RecordBatchOnce {
   fn schema(&self) -> arrow_schema::SchemaRef {
      match self {
         Self::Batch(batch) => batch.schema(),
         Self::Taken(schema) => schema.clone(),
         Self::__Invalid => unreachable!(),
      }
   }
}

pub struct LanceStore {
   connections: RwLock<HashMap<String, Arc<Connection>>>,
   data_dir:    PathBuf,
}

impl LanceStore {
   pub fn new() -> Result<Self> {
      let data_dir = config::data_dir().join("data");
      fs::create_dir_all(&data_dir)?;

      Ok(Self { connections: RwLock::new(HashMap::new()), data_dir })
   }

   async fn get_connection(&self, store_id: &str) -> Result<Arc<Connection>> {
      {
         let connections = self.connections.read();
         if let Some(conn) = connections.get(store_id) {
            return Ok(Arc::clone(conn));
         }
      }

      let db_path = self.data_dir.join(store_id);
      fs::create_dir_all(&db_path)?;

      let conn = connect(db_path.to_str().ok_or(StoreError::InvalidDatabasePath)?)
         .execute()
         .await
         .map_err(StoreError::Connect)?;

      let conn = Arc::new(conn);
      self
         .connections
         .write()
         .insert(store_id.to_string(), Arc::clone(&conn));

      Ok(conn)
   }

   async fn get_table(&self, store_id: &str) -> Result<Table> {
      let conn = self.get_connection(store_id).await?;

      let table = if let Ok(table) = conn.open_table(store_id).execute().await {
         Self::check_and_migrate_table(&conn, store_id, &table).await?;
         conn
            .open_table(store_id)
            .execute()
            .await
            .map_err(StoreError::ReopenTableAfterMigration)?
      } else {
         let schema = Self::create_schema();
         let empty_batch = Self::create_empty_batch(&schema)?;

         conn
            .create_table(store_id, RecordBatchOnce::new(empty_batch))
            .execute()
            .await
            .map_err(StoreError::CreateTable)?
      };
      Ok(table)
   }

   async fn check_and_migrate_table(
      conn: &Connection,
      store_id: &str,
      table: &Table,
   ) -> Result<()> {
      let mut stream = table
         .query()
         .limit(1)
         .execute()
         .await
         .map_err(StoreError::SampleTableForMigration)?;

      let sample_batch = match stream.try_next().await {
         Ok(Some(batch)) => batch,
         Ok(None) => return Ok(()),
         Err(e) => {
            return Err(StoreError::ReadSampleBatch(e).into());
         },
      };

      let Some(vector_col) = sample_batch.column_by_name("vector") else {
         return Ok(());
      };

      let Some(vector_list) = vector_col.as_any().downcast_ref::<FixedSizeListArray>() else {
         return Ok(());
      };

      let sample_vector_len = vector_list.value_length() as usize;

      if sample_vector_len == config::get().dense_dim {
         return Ok(());
      }

      let mut all_stream = table
         .query()
         .execute()
         .await
         .map_err(StoreError::ReadExistingDataForMigration)?;

      let mut existing_batches: Vec<RecordBatch> = Vec::new();
      while let Some(batch) = all_stream
         .try_next()
         .await
         .map_err(StoreError::CollectExistingBatches)?
      {
         existing_batches.push(batch);
      }

      conn
         .drop_table(store_id, &[])
         .await
         .map_err(StoreError::DropOldTableDuringMigration)?;

      let schema = Self::create_schema();
      let empty_batch = Self::create_empty_batch(&schema)?;

      let new_table = conn
         .create_table(store_id, RecordBatchOnce::new(empty_batch))
         .execute()
         .await
         .map_err(StoreError::CreateNewTableDuringMigration)?;

      if !existing_batches.is_empty() {
         let mut migrated_records = Vec::new();

         for batch in existing_batches {
            for row_idx in 0..batch.num_rows() {
               let id = batch
                  .column_by_name("id")
                  .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                  .map(|arr| arr.value(row_idx).to_string())
                  .unwrap_or_default();

               let path: PathBuf = batch
                  .column_by_name("path")
                  .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                  .map(|arr| arr.value(row_idx).into())
                  .unwrap_or_default();

               let hash = batch
                  .column_by_name("hash")
                  .and_then(|col| col.as_any().downcast_ref::<BinaryArray>())
                  .and_then(|arr| FileHash::from_slice(arr.value(row_idx)))
                  .unwrap_or_default();

               let content: Str = batch
                  .column_by_name("content")
                  .and_then(|col| {
                     col.as_any()
                        .downcast_ref::<LargeStringArray>()
                        .map(|arr| arr.value(row_idx).to_string())
                        .or_else(|| {
                           col.as_any()
                              .downcast_ref::<StringArray>()
                              .map(|arr| arr.value(row_idx).to_string())
                        })
                  })
                  .unwrap_or_default()
                  .into();

               let start_line = batch
                  .column_by_name("start_line")
                  .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
                  .map_or(0, |arr| arr.value(row_idx));

               let end_line = batch
                  .column_by_name("end_line")
                  .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
                  .map_or(start_line, |arr| arr.value(row_idx));

               let old_vector_col = batch.column_by_name("vector").unwrap();
               let old_vector_list = old_vector_col
                  .as_any()
                  .downcast_ref::<FixedSizeListArray>()
                  .unwrap();
               let old_vector_values = old_vector_list.value(row_idx);
               let old_vector_floats = old_vector_values
                  .as_any()
                  .downcast_ref::<Float32Array>()
                  .unwrap();

               let old_vec: Vec<f32> = (0..old_vector_floats.len())
                  .map(|i| old_vector_floats.value(i))
                  .collect();

               let new_vector = Self::normalize_vector(&old_vec);

               let colbert = batch
                  .column_by_name("colbert")
                  .and_then(|col| {
                     if col.is_null(row_idx) {
                        None
                     } else {
                        col.as_any()
                           .downcast_ref::<LargeBinaryArray>()
                           .map(|arr| arr.value(row_idx).to_vec())
                     }
                  })
                  .unwrap_or_default();

               let colbert_scale = batch
                  .column_by_name("colbert_scale")
                  .and_then(|col| {
                     if col.is_null(row_idx) {
                        None
                     } else {
                        col.as_any()
                           .downcast_ref::<Float64Array>()
                           .map(|arr| arr.value(row_idx))
                     }
                  })
                  .unwrap_or(1.0);

               let chunk_index = batch.column_by_name("chunk_index").and_then(|col| {
                  if col.is_null(row_idx) {
                     None
                  } else {
                     col.as_any()
                        .downcast_ref::<UInt32Array>()
                        .map(|arr| arr.value(row_idx))
                  }
               });

               let is_anchor = batch.column_by_name("is_anchor").and_then(|col| {
                  if col.is_null(row_idx) {
                     None
                  } else {
                     col.as_any()
                        .downcast_ref::<BooleanArray>()
                        .map(|arr| arr.value(row_idx))
                  }
               });

               let chunk_type = batch.column_by_name("chunk_type").and_then(|col| {
                  if col.is_null(row_idx) {
                     None
                  } else {
                     col.as_any()
                        .downcast_ref::<StringArray>()
                        .map(|arr| Self::parse_chunk_type(arr.value(row_idx)))
                  }
               });

               let context_prev: Option<Str> =
                  batch.column_by_name("context_prev").and_then(|col| {
                     if col.is_null(row_idx) {
                        None
                     } else {
                        col.as_any()
                           .downcast_ref::<StringArray>()
                           .map(|arr| Str::copy_from_str(arr.value(row_idx)))
                     }
                  });

               let context_next: Option<Str> =
                  batch.column_by_name("context_next").and_then(|col| {
                     if col.is_null(row_idx) {
                        None
                     } else {
                        col.as_any()
                           .downcast_ref::<StringArray>()
                           .map(|arr| Str::copy_from_str(arr.value(row_idx)))
                     }
                  });

               migrated_records.push(VectorRecord {
                  id,
                  path,
                  hash,
                  content,
                  start_line,
                  end_line,
                  vector: new_vector,
                  colbert,
                  colbert_scale,
                  chunk_index,
                  is_anchor,
                  chunk_type,
                  context_prev,
                  context_next,
               });
            }
         }

         if !migrated_records.is_empty() {
            let migrated_batch = Self::records_to_batch(migrated_records)?;
            new_table
               .add(RecordBatchOnce::new(migrated_batch))
               .execute()
               .await
               .map_err(StoreError::AddMigratedRecords)?;
         }
      }

      Ok(())
   }

   fn normalize_vector(old_vector: &[f32]) -> Vec<f32> {
      let dim = config::get().dense_dim;
      let mut new_vector = vec![0.0; dim];
      let copy_len = old_vector.len().min(dim);
      new_vector[..copy_len].copy_from_slice(&old_vector[..copy_len]);
      new_vector
   }

   fn create_schema() -> Arc<Schema> {
      Arc::new(Schema::new(vec![
         Field::new("id", DataType::Utf8, false),
         Field::new("path", DataType::Utf8, false),
         Field::new("hash", DataType::Binary, false),
         Field::new("content", DataType::LargeUtf8, false),
         Field::new("start_line", DataType::UInt32, false),
         Field::new("end_line", DataType::UInt32, false),
         Field::new(
            "vector",
            DataType::FixedSizeList(
               Arc::new(Field::new("item", DataType::Float32, true)),
               config::get().dense_dim as i32,
            ),
            false,
         ),
         Field::new("colbert", DataType::LargeBinary, true),
         Field::new("colbert_scale", DataType::Float64, true),
         Field::new("chunk_index", DataType::UInt32, true),
         Field::new("is_anchor", DataType::Boolean, true),
         Field::new("chunk_type", DataType::Utf8, true),
         Field::new("context_prev", DataType::Utf8, true),
         Field::new("context_next", DataType::Utf8, true),
      ]))
   }

   fn create_empty_batch(schema: &Arc<Schema>) -> Result<RecordBatch> {
      let id_array = StringBuilder::new().finish();
      let path_array = StringBuilder::new().finish();
      let hash_array = BinaryBuilder::new().finish();
      let content_array = LargeStringBuilder::new().finish();
      let start_line_array = UInt32Builder::new().finish();
      let end_line_array = UInt32Builder::new().finish();

      let vector_values = Float32Builder::new().finish();
      let vector_array = FixedSizeListArray::new(
         Arc::new(Field::new("item", DataType::Float32, true)),
         config::get().dense_dim as i32,
         Arc::new(vector_values),
         None,
      );

      let colbert_array = LargeBinaryBuilder::new().finish();
      let colbert_scale_array = Float64Builder::new().finish();
      let chunk_index_array = UInt32Builder::new().finish();
      let is_anchor_array = BooleanBuilder::new().finish();
      let chunk_type_array = StringBuilder::new().finish();
      let context_prev_array = StringBuilder::new().finish();
      let context_next_array = StringBuilder::new().finish();

      Ok(RecordBatch::try_new(schema.clone(), vec![
         Arc::new(id_array),
         Arc::new(path_array),
         Arc::new(hash_array),
         Arc::new(content_array),
         Arc::new(start_line_array),
         Arc::new(end_line_array),
         Arc::new(vector_array),
         Arc::new(colbert_array),
         Arc::new(colbert_scale_array),
         Arc::new(chunk_index_array),
         Arc::new(is_anchor_array),
         Arc::new(chunk_type_array),
         Arc::new(context_prev_array),
         Arc::new(context_next_array),
      ])
      .map_err(StoreError::CreateEmptyBatch)?)
   }

   fn records_to_batch(records: Vec<VectorRecord>) -> Result<RecordBatch> {
      if records.is_empty() {
         return Err(StoreError::EmptyBatch.into());
      }

      let schema = Self::create_schema();
      let _len = records.len();

      let mut id_builder = StringBuilder::new();
      let mut path_builder = StringBuilder::new();
      let mut hash_builder = BinaryBuilder::new();
      let mut content_builder = LargeStringBuilder::new();
      let mut start_line_builder = UInt32Builder::new();
      let mut end_line_builder = UInt32Builder::new();
      let mut vector_builder = Float32Builder::new();
      let mut colbert_builder = LargeBinaryBuilder::new();
      let mut colbert_scale_builder = Float64Builder::new();
      let mut chunk_index_builder = UInt32Builder::new();
      let mut is_anchor_builder = BooleanBuilder::new();
      let mut chunk_type_builder = StringBuilder::new();
      let mut context_prev_builder = StringBuilder::new();
      let mut context_next_builder = StringBuilder::new();

      for record in records {
         id_builder.append_value(&record.id);
         path_builder.append_value(store::path_to_store_key(&record.path));
         hash_builder.append_value(record.hash);
         content_builder.append_value(&record.content);
         start_line_builder.append_value(record.start_line);
         end_line_builder.append_value(record.end_line);

         let dim = config::get().dense_dim;
         if record.vector.len() != dim {
            return Err(StoreError::VectorColumnTypeMismatch.into());
         }

         for &val in &record.vector {
            vector_builder.append_value(val);
         }

         colbert_builder.append_value(&record.colbert);
         colbert_scale_builder.append_value(record.colbert_scale);

         if let Some(idx) = record.chunk_index {
            chunk_index_builder.append_value(idx);
         } else {
            chunk_index_builder.append_null();
         }

         if let Some(is_anchor) = record.is_anchor {
            is_anchor_builder.append_value(is_anchor);
         } else {
            is_anchor_builder.append_null();
         }

         if let Some(chunk_type) = record.chunk_type {
            let chunk_type_str = match chunk_type {
               ChunkType::Function => "function",
               ChunkType::Class => "class",
               ChunkType::Interface => "interface",
               ChunkType::Method => "method",
               ChunkType::TypeAlias => "type_alias",
               ChunkType::Block => "block",
               ChunkType::Other => "other",
            };
            chunk_type_builder.append_value(chunk_type_str);
         } else {
            chunk_type_builder.append_null();
         }

         if let Some(prev) = &record.context_prev {
            context_prev_builder.append_value(prev);
         } else {
            context_prev_builder.append_null();
         }

         if let Some(next) = &record.context_next {
            context_next_builder.append_value(next);
         } else {
            context_next_builder.append_null();
         }
      }

      let id_array = id_builder.finish();
      let path_array = path_builder.finish();
      let hash_array = hash_builder.finish();
      let content_array = content_builder.finish();
      let start_line_array = start_line_builder.finish();
      let end_line_array = end_line_builder.finish();

      let vector_values_array = vector_builder.finish();
      let vector_array = FixedSizeListArray::new(
         Arc::new(Field::new("item", DataType::Float32, true)),
         config::get().dense_dim as i32,
         Arc::new(vector_values_array),
         None,
      );

      let colbert_array = colbert_builder.finish();
      let colbert_scale_array = colbert_scale_builder.finish();
      let chunk_index_array = chunk_index_builder.finish();
      let is_anchor_array = is_anchor_builder.finish();
      let chunk_type_array = chunk_type_builder.finish();
      let context_prev_array = context_prev_builder.finish();
      let context_next_array = context_next_builder.finish();

      Ok(RecordBatch::try_new(schema, vec![
         Arc::new(id_array),
         Arc::new(path_array),
         Arc::new(hash_array),
         Arc::new(content_array),
         Arc::new(start_line_array),
         Arc::new(end_line_array),
         Arc::new(vector_array),
         Arc::new(colbert_array),
         Arc::new(colbert_scale_array),
         Arc::new(chunk_index_array),
         Arc::new(is_anchor_array),
         Arc::new(chunk_type_array),
         Arc::new(context_prev_array),
         Arc::new(context_next_array),
      ])
      .map_err(StoreError::CreateRecordBatch)?)
   }

   fn parse_chunk_type(s: &str) -> ChunkType {
      match s {
         "function" => ChunkType::Function,
         "class" => ChunkType::Class,
         "interface" => ChunkType::Interface,
         "method" => ChunkType::Method,
         "type_alias" => ChunkType::TypeAlias,
         "block" => ChunkType::Block,
         _ => ChunkType::Other,
      }
   }

   fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
      let len = a.len().min(b.len());
      let mut dot = 0.0;
      for i in 0..len {
         dot += a[i] * b[i];
      }
      dot
   }

   fn maxsim(query_matrix: &[Vec<f32>], doc_matrix: &[Vec<f32>]) -> f32 {
      let mut total = 0.0;
      for q_vec in query_matrix {
         let mut max_score = f32::MIN;
         for d_vec in doc_matrix {
            let score = Self::cosine_similarity(q_vec, d_vec);
            if score > max_score {
               max_score = score;
            }
         }
         total += max_score;
      }
      total
   }

   fn decode_colbert(colbert_bytes: &[u8], scale: f64) -> Vec<Vec<f32>> {
      let dim = config::get().colbert_dim;
      let mut result = Vec::new();
      let mut i = 0;

      while i + dim <= colbert_bytes.len() {
         let mut row = Vec::with_capacity(dim);
         let mut is_padding = true;

         for j in 0..dim {
            let byte_val = colbert_bytes[i + j] as i8;
            let val = (byte_val as f32 / 127.0) * scale as f32;
            if val != 0.0 {
               is_padding = false;
            }
            row.push(val);
         }

         if !is_padding {
            result.push(row);
         }

         i += dim;
      }

      result
   }
}

impl Default for LanceStore {
   fn default() -> Self {
      Self::new().expect("failed to create LanceStore")
   }
}

#[async_trait::async_trait]
impl super::Store for LanceStore {
   async fn insert_batch(&self, store_id: &str, records: Vec<VectorRecord>) -> Result<()> {
      if records.is_empty() {
         return Ok(());
      }

      let table = self.get_table(store_id).await?;
      let batch = Self::records_to_batch(records)?;

      table
         .add(RecordBatchOnce::new(batch))
         .execute()
         .await
         .map_err(StoreError::AddRecords)?;

      Ok(())
   }

   async fn search(&self, params: store::SearchParams<'_>) -> Result<SearchResponse> {
      let Ok(table) = self.get_table(params.store_id).await else {
         return Ok(SearchResponse {
            results:  vec![],
            status:   SearchStatus::Ready,
            progress: None,
         });
      };

      let anchor_filter = "(is_anchor IS NULL OR is_anchor = false)";
      let doc_clause =
         "(path LIKE '%.md' OR path LIKE '%.mdx' OR path LIKE '%.txt' OR path LIKE '%.json')";
      let code_clause = format!("NOT {doc_clause}");

      let mut code_filter = format!("{code_clause} AND {anchor_filter}");
      let mut doc_filter = format!("{doc_clause} AND {anchor_filter}");
      let base_filter = if let Some(filter) = params.path_filter {
         let filter_str = store::path_to_store_key(filter);
         let path_clause = format!("path LIKE '{filter_str}%'");
         code_filter = format!("{path_clause} AND {code_clause} AND {anchor_filter}");
         doc_filter = format!("{path_clause} AND {doc_clause} AND {anchor_filter}");
         Some(format!("{path_clause} AND {anchor_filter}"))
      } else {
         Some(anchor_filter.to_string())
      };

      let code_results_stream = table
         .query()
         .nearest_to(params.query_vector)
         .map_err(StoreError::CreateVectorQuery)?
         .limit(300)
         .only_if(&code_filter)
         .execute()
         .await
         .map_err(StoreError::ExecuteCodeSearch)?;

      let doc_results_stream = table
         .query()
         .nearest_to(params.query_vector)
         .map_err(StoreError::CreateVectorQuery)?
         .only_if(&doc_filter)
         .limit(50)
         .execute()
         .await
         .map_err(StoreError::ExecuteDocSearch)?;

      let code_batches: Vec<RecordBatch> = code_results_stream
         .try_collect()
         .await
         .map_err(StoreError::CollectCodeResults)?;

      let doc_batches: Vec<RecordBatch> = doc_results_stream
         .try_collect()
         .await
         .map_err(StoreError::CollectDocResults)?;

      let fts_query = FullTextSearchQuery::new(params.query_text.to_owned());
      let mut fts_query_builder = table.query().full_text_search(fts_query);

      if let Some(ref filter) = base_filter {
         fts_query_builder = fts_query_builder.only_if(filter);
      }

      let fts_batches: Vec<RecordBatch> = match fts_query_builder.limit(50).execute().await {
         Ok(stream) => stream.try_collect().await.unwrap_or_default(),
         Err(_) => vec![],
      };

      let mut candidates = Vec::new();
      let mut seen_keys = HashMap::new();

      for batch in code_batches
         .iter()
         .chain(doc_batches.iter())
         .chain(fts_batches.iter())
      {
         let path_col = batch
            .column_by_name("path")
            .ok_or(StoreError::MissingPathColumn)?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(StoreError::PathColumnTypeMismatch)?;

         let start_line_col = batch
            .column_by_name("start_line")
            .ok_or(StoreError::MissingStartLineColumn)?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or(StoreError::StartLineTypeMismatch)?;

         for i in 0..batch.num_rows() {
            if path_col.is_null(i) {
               continue;
            }

            let path = path_col.value(i).to_string();
            let start_line = start_line_col.value(i);

            let key = format!("{path}:{start_line}");
            if seen_keys.contains_key(&key) {
               continue;
            }
            seen_keys.insert(key, ());

            candidates.push((i, batch.clone()));
         }
      }

      let mut scored_results = Vec::new();

      for (row_idx, batch) in candidates {
         let path: PathBuf = batch
            .column_by_name("path")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row_idx)
            .into();

         let content_col = batch.column_by_name("content").unwrap();
         let content = if let Some(str_array) = content_col.as_any().downcast_ref::<StringArray>() {
            str_array.value(row_idx).to_string()
         } else if let Some(large_str_array) =
            content_col.as_any().downcast_ref::<LargeStringArray>()
         {
            large_str_array.value(row_idx).to_string()
         } else {
            return Err(StoreError::ContentColumnTypeMismatch.into());
         };

         let start_line = batch
            .column_by_name("start_line")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(row_idx);

         let end_line = batch
            .column_by_name("end_line")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(row_idx);

         let chunk_type = batch.column_by_name("chunk_type").and_then(|col| {
            if col.is_null(row_idx) {
               None
            } else {
               col.as_any()
                  .downcast_ref::<StringArray>()
                  .map(|arr| Self::parse_chunk_type(arr.value(row_idx)))
            }
         });

         let is_anchor = batch.column_by_name("is_anchor").and_then(|col| {
            if col.is_null(row_idx) {
               None
            } else {
               col.as_any()
                  .downcast_ref::<BooleanArray>()
                  .map(|arr| arr.value(row_idx))
            }
         });

         let vector_list = batch
            .column_by_name("vector")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or(StoreError::VectorColumnTypeMismatch)?;
         let vector_values = vector_list.value(row_idx);
         let vector_floats = vector_values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(StoreError::VectorValuesTypeMismatch)?;

         let doc_vector: Vec<f32> = (0..vector_floats.len())
            .map(|i| vector_floats.value(i))
            .collect();

         let mut score = Self::cosine_similarity(params.query_vector, &doc_vector);

         if params.rerank
            && !params.query_colbert.is_empty()
            && let Some(colbert_col) = batch.column_by_name("colbert")
            && !colbert_col.is_null(row_idx)
         {
            let colbert_binary = if let Some(large_binary_array) =
               colbert_col.as_any().downcast_ref::<LargeBinaryArray>()
            {
               large_binary_array.value(row_idx)
            } else {
               &[]
            };

            if !colbert_binary.is_empty() {
               let scale = if let Some(scale_col) = batch.column_by_name("colbert_scale") {
                  if scale_col.is_null(row_idx) {
                     1.0
                  } else {
                     scale_col
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .map_or(1.0, |arr| arr.value(row_idx))
                  }
               } else {
                  1.0
               };

               let doc_matrix = Self::decode_colbert(colbert_binary, scale);
               if !doc_matrix.is_empty() {
                  score = Self::maxsim(params.query_colbert, &doc_matrix);
               }
            }
         }

         let mut full_content = String::new();
         let mut context_prev_lines = 0u32;

         if let Some(prev_col) = batch.column_by_name("context_prev")
            && !prev_col.is_null(row_idx)
            && let Some(prev_str) = prev_col.as_any().downcast_ref::<StringArray>()
         {
            let prev_content = prev_str.value(row_idx);
            context_prev_lines = prev_content.lines().count() as u32;
            full_content.push_str(prev_content);
         }
         full_content.push_str(&content);
         if let Some(next_col) = batch.column_by_name("context_next")
            && !next_col.is_null(row_idx)
            && let Some(next_str) = next_col.as_any().downcast_ref::<StringArray>()
         {
            full_content.push_str(next_str.value(row_idx));
         }

         let adjusted_start_line = start_line.saturating_sub(context_prev_lines);

         scored_results.push(SearchResult {
            path,
            content: full_content.into(),
            score,
            start_line: adjusted_start_line,
            num_lines: end_line.saturating_sub(start_line).max(1),
            chunk_type,
            is_anchor,
         });
      }

      scored_results.sort_by(|a, b| {
         b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
      });
      scored_results.truncate(params.limit);

      Ok(SearchResponse { results: scored_results, status: SearchStatus::Ready, progress: None })
   }

   async fn delete_file(&self, store_id: &str, file_path: &Path) -> Result<()> {
      let table = self.get_table(store_id).await?;
      let escaped = store::path_to_store_key(file_path);
      table
         .delete(&format!("path = '{escaped}'"))
         .await
         .map_err(StoreError::DeleteFile)?;

      Ok(())
   }

   async fn delete_files(&self, store_id: &str, file_paths: &[PathBuf]) -> Result<()> {
      if file_paths.is_empty() {
         return Ok(());
      }

      let table = self.get_table(store_id).await?;
      let unique_paths: Vec<_> = file_paths
         .iter()
         .collect::<HashSet<_>>()
         .into_iter()
         .collect();

      const BATCH_SIZE: usize = 900;
      for chunk in unique_paths.chunks(BATCH_SIZE) {
         let escaped: Vec<String> = chunk
            .iter()
            .map(|p| format!("'{}'", store::path_to_store_key(p)))
            .collect();
         let predicate = format!("path IN ({})", escaped.join(","));

         table
            .delete(&predicate)
            .await
            .map_err(StoreError::DeleteFiles)?;
      }

      Ok(())
   }

   async fn delete_store(&self, store_id: &str) -> Result<()> {
      let conn = self.get_connection(store_id).await?;

      conn
         .drop_table(store_id, &[])
         .await
         .map_err(StoreError::DropTable)?;

      self.connections.write().remove(store_id);

      Ok(())
   }

   async fn get_info(&self, store_id: &str) -> Result<StoreInfo> {
      let table = self.get_table(store_id).await?;
      let row_count = table
         .count_rows(None)
         .await
         .map_err(StoreError::CountRows)?;

      Ok(StoreInfo {
         store_id:  store_id.to_string(),
         row_count: row_count as u64,
         path:      self.data_dir.join(store_id),
      })
   }

   async fn list_files(&self, store_id: &str) -> Result<Vec<PathBuf>> {
      let Ok(table) = self.get_table(store_id).await else {
         return Ok(vec![]);
      };

      let stream_result = table
         .query()
         .only_if("is_anchor = true")
         .select(Select::Columns(vec!["path".to_string()]))
         .execute()
         .await;

      let stream = match stream_result {
         Ok(s) => s,
         Err(_) => table
            .query()
            .select(Select::Columns(vec!["path".to_string()]))
            .execute()
            .await
            .map_err(StoreError::ExecuteQuery)?,
      };

      let batches: Vec<RecordBatch> = stream
         .try_collect()
         .await
         .map_err(StoreError::CollectResults)?;

      let mut paths = Vec::new();
      let mut seen = HashSet::new();

      for batch in batches {
         if let Some(path_col) = batch.column_by_name("path")
            && let Some(path_array) = path_col.as_any().downcast_ref::<StringArray>()
         {
            for i in 0..path_array.len() {
               if !path_array.is_null(i) {
                  let path: PathBuf = path_array.value(i).into();
                  if seen.insert(path.clone()) {
                     paths.push(path);
                  }
               }
            }
         }
      }

      Ok(paths)
   }

   async fn is_empty(&self, store_id: &str) -> Result<bool> {
      let Ok(table) = self.get_table(store_id).await else {
         return Ok(true);
      };

      let row_count = table
         .count_rows(None)
         .await
         .map_err(StoreError::CountRows)?;

      Ok(row_count == 0)
   }

   async fn create_fts_index(&self, store_id: &str) -> Result<()> {
      let table = self.get_table(store_id).await?;

      table
         .create_index(&["content"], Index::FTS(Default::default()))
         .execute()
         .await
         .map_err(|e| {
            if matches!(e, lancedb::Error::TableAlreadyExists { .. }) {
               return StoreError::IndexAlreadyExists;
            }
            StoreError::CreateFtsIndex(e)
         })?;

      Ok(())
   }

   async fn create_vector_index(&self, store_id: &str) -> Result<()> {
      let table = self.get_table(store_id).await?;

      let row_count = table
         .count_rows(None)
         .await
         .map_err(StoreError::CountRows)?;

      if row_count < 4000 {
         return Ok(());
      }

      let num_partitions = (row_count / 100).clamp(8, 64) as u32;

      let index = Index::IvfPq(
         lancedb::index::vector::IvfPqIndexBuilder::default().num_partitions(num_partitions),
      );

      table
         .create_index(&["vector"], index)
         .execute()
         .await
         .map_err(|e| {
            if matches!(e, lancedb::Error::TableAlreadyExists { .. }) {
               return StoreError::IndexAlreadyExists;
            }
            StoreError::CreateVectorIndex(e)
         })?;

      Ok(())
   }

   async fn get_file_hashes(&self, store_id: &str) -> Result<HashMap<PathBuf, FileHash>> {
      let Ok(table) = self.get_table(store_id).await else {
         return Ok(HashMap::new());
      };

      let stream_result = table
         .query()
         .only_if("is_anchor = true")
         .select(Select::Columns(vec!["path".to_string(), "hash".to_string()]))
         .execute()
         .await;

      let stream = match stream_result {
         Ok(s) => s,
         Err(_) => table
            .query()
            .select(Select::Columns(vec!["path".to_string(), "hash".to_string()]))
            .execute()
            .await
            .map_err(StoreError::ExecuteQuery)?,
      };

      let batches: Vec<RecordBatch> = stream
         .try_collect()
         .await
         .map_err(StoreError::CollectResults)?;

      let mut hashes = HashMap::new();

      for batch in batches {
         if let (Some(path_col), Some(hash_col)) =
            (batch.column_by_name("path"), batch.column_by_name("hash"))
            && let (Some(path_array), Some(hash_array)) = (
               path_col.as_any().downcast_ref::<StringArray>(),
               hash_col.as_any().downcast_ref::<BinaryArray>(),
            )
         {
            for i in 0..path_array.len() {
               if !path_array.is_null(i) && !hash_array.is_null(i) {
                  let path = path_array.value(i).into();
                  let hash = FileHash::from_slice(hash_array.value(i)).expect("invalid hash");
                  hashes.insert(path, hash);
               }
            }
         }
      }

      Ok(hashes)
   }
}
