use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{Result, error::IpcError, types::SearchResponse};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
   Search { query: String, limit: usize, path: Option<PathBuf>, rerank: bool },
   Health,
   Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
   Search(SearchResponse),
   Health { status: ServerStatus },
   Shutdown { success: bool },
   Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
   pub indexing: bool,
   pub progress: u8,
   pub files:    usize,
}

pub struct SocketBuffer {
   buf: SmallVec<[u8; 2048]>,
}

impl Extend<u8> for &mut SocketBuffer {
   fn extend<I: IntoIterator<Item = u8>>(&mut self, iter: I) {
      self.buf.extend(iter);
   }
}

impl Default for SocketBuffer {
   fn default() -> Self {
      Self::new()
   }
}

impl SocketBuffer {
   pub fn new() -> Self {
      Self { buf: SmallVec::new() }
   }

   #[allow(
      clippy::future_not_send,
      reason = "Generic async function with references - Send bound would be too restrictive for \
                trait"
   )]
   pub async fn send<W, T>(&mut self, writer: &mut W, msg: &T) -> Result<()>
   where
      W: AsyncWrite + Unpin,
      T: Serialize,
   {
      self.buf.clear();
      self.buf.resize(4, 0u8);
      _ = postcard::to_extend(msg, &mut *self).map_err(IpcError::Serialize)?;
      let payload_len = (self.buf.len() - 4) as u32;
      *self.buf.first_chunk_mut().unwrap() = payload_len.to_le_bytes();
      writer.write_all(&self.buf).await.map_err(IpcError::Write)?;
      writer.flush().await.map_err(IpcError::Write)?;
      Ok(())
   }

   pub async fn recv<'de, R, T>(&'de mut self, reader: &mut R) -> Result<T>
   where
      R: AsyncRead + Unpin,
      T: Deserialize<'de>,
   {
      let mut len_buf = [0u8; 4];
      reader
         .read_exact(&mut len_buf)
         .await
         .map_err(IpcError::Read)?;
      let len = u32::from_le_bytes(len_buf) as usize;

      if len > 16 * 1024 * 1024 {
         return Err(IpcError::MessageTooLarge(len).into());
      }

      self.buf.resize(len, 0u8);
      reader
         .read_exact(self.buf.as_mut_slice())
         .await
         .map_err(IpcError::Read)?;
      postcard::from_bytes(&self.buf).map_err(|e| IpcError::Deserialize(e).into())
   }
}
