use std::{
   fs, io,
   path::PathBuf,
   pin::Pin,
   task::{self, Poll},
};

use tokio::{
   io::ReadBuf,
   net::{UnixListener as TokioUnixListener, UnixStream as TokioUnixStream},
};

use super::SocketError;
use crate::{Result, config};

pub fn socket_path(store_id: &str) -> PathBuf {
   config::socket_dir().join(format!("{store_id}.sock"))
}

pub fn list_running_servers() -> Vec<String> {
   let dir = config::socket_dir();
   if !dir.exists() {
      return Vec::new();
   }

   fs::read_dir(dir)
      .into_iter()
      .flatten()
      .filter_map(|e| e.ok())
      .filter(|e| e.path().extension().is_some_and(|ext| ext == "sock"))
      .filter_map(|e| {
         e.path()
            .file_stem()
            .and_then(|s| s.to_str())
            .map(String::from)
      })
      .collect()
}

pub fn remove_socket(store_id: &str) {
   let _ = fs::remove_file(socket_path(store_id));
}

pub struct Listener {
   inner: TokioUnixListener,
   path:  PathBuf,
}

impl Listener {
   pub async fn bind(store_id: &str) -> Result<Self> {
      let path = socket_path(store_id);

      if let Some(parent) = path.parent() {
         fs::create_dir_all(parent).map_err(SocketError::CreateDir)?;
      }

      if path.exists() {
         if Stream::connect(store_id).await.is_ok() {
            return Err(SocketError::AlreadyRunning.into());
         }
         fs::remove_file(&path).map_err(SocketError::RemoveStale)?;
      }

      let inner = TokioUnixListener::bind(&path).map_err(SocketError::Bind)?;
      Ok(Self { inner, path })
   }

   pub async fn accept(&self) -> Result<Stream> {
      let (stream, _) = self.inner.accept().await.map_err(SocketError::Accept)?;
      Ok(Stream { inner: stream })
   }

   pub fn local_addr(&self) -> String {
      self.path.display().to_string()
   }
}

impl Drop for Listener {
   fn drop(&mut self) {
      let _ = fs::remove_file(&self.path);
   }
}

#[repr(transparent)]
pub struct Stream {
   inner: TokioUnixStream,
}

impl Stream {
   pub async fn connect(store_id: &str) -> Result<Self> {
      let path = socket_path(store_id);
      let inner = TokioUnixStream::connect(&path)
         .await
         .map_err(SocketError::Connect)?;
      Ok(Self { inner })
   }
}

impl tokio::io::AsyncRead for Stream {
   fn poll_read(
      mut self: Pin<&mut Self>,
      cx: &mut task::Context<'_>,
      buf: &mut ReadBuf<'_>,
   ) -> Poll<io::Result<()>> {
      Pin::new(&mut self.inner).poll_read(cx, buf)
   }
}

impl tokio::io::AsyncWrite for Stream {
   fn poll_write(
      mut self: Pin<&mut Self>,
      cx: &mut task::Context<'_>,
      buf: &[u8],
   ) -> Poll<io::Result<usize>> {
      Pin::new(&mut self.inner).poll_write(cx, buf)
   }

   fn poll_flush(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<io::Result<()>> {
      Pin::new(&mut self.inner).poll_flush(cx)
   }

   fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<io::Result<()>> {
      Pin::new(&mut self.inner).poll_shutdown(cx)
   }
}
