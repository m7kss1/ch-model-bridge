use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use protocol::passport::ModelKind;
use protocol::wire::{self, Request, Response};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

use crate::registry::Registry;

/// Binary channel for `bridge-client`: one length-prefixed request frame in,
/// one response frame out. Requests land in the same per-model dispatcher as
/// HTTP, so batches merge across both channels.
pub async fn serve(path: PathBuf, registry: Arc<Registry>) -> anyhow::Result<()> {
    // A previous run's socket file would fail the bind; nothing else may own
    // this path.
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).with_context(|| path.display().to_string())?;
    tracing::info!(socket = %path.display(), "binary channel listening");
    loop {
        let (stream, _) = listener.accept().await?;
        let registry = Arc::clone(&registry);
        tokio::spawn(async move {
            if let Err(e) = handle(stream, registry).await {
                tracing::debug!("socket client: {e}");
            }
        });
    }
}

async fn handle(mut stream: UnixStream, registry: Arc<Registry>) -> anyhow::Result<()> {
    loop {
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            // A closed connection between frames is the client's normal exit.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf);
        if len > wire::MAX_FRAME {
            anyhow::bail!("frame of {len} bytes exceeds the limit");
        }
        let mut payload = vec![0u8; len as usize];
        stream.read_exact(&mut payload).await?;

        let response = process(&payload, &registry).await;
        let encoded = wire::encode_response(&response);
        stream
            .write_all(&(encoded.len() as u32).to_le_bytes())
            .await?;
        stream.write_all(&encoded).await?;
        stream.flush().await?;
    }
}

async fn process(payload: &[u8], registry: &Registry) -> Response {
    let request = match wire::decode_request(payload) {
        Ok(request) => request,
        Err(message) => return Response::Error(message),
    };
    match request {
        Request::Embed { model, texts } => {
            let Some(entry) = registry.resolve(&model) else {
                return unknown_model(registry, &model);
            };
            if entry.card.kind != ModelKind::Embedding {
                return Response::Error(format!("model `{model}` is not an embedding model"));
            }
            let dim = entry.card.dim.unwrap_or(0) as u32;
            if texts.is_empty() {
                return Response::Embed {
                    dim,
                    vectors: Vec::new(),
                };
            }
            match entry.handle.clone().embed(texts).await {
                Ok(reply) => {
                    let mut vectors = Vec::with_capacity(reply.vectors.len() * dim as usize);
                    for vector in reply.vectors {
                        vectors.extend(vector);
                    }
                    Response::Embed { dim, vectors }
                }
                Err(e) => Response::Error(e.to_string()),
            }
        }
        Request::Rerank { model, pairs } => {
            let Some(entry) = registry.resolve(&model) else {
                return unknown_model(registry, &model);
            };
            if entry.card.kind != ModelKind::Rerank {
                return Response::Error(format!("model `{model}` is not a rerank model"));
            }
            if pairs.is_empty() {
                return Response::Rerank { scores: Vec::new() };
            }
            match entry.handle.clone().rerank(pairs).await {
                Ok(reply) => Response::Rerank {
                    scores: reply.scores,
                },
                Err(e) => Response::Error(e.to_string()),
            }
        }
        Request::Evaluate {
            model,
            n_features,
            values,
        } => {
            let Some(entry) = registry.resolve(&model) else {
                return unknown_model(registry, &model);
            };
            if entry.card.kind != ModelKind::Tabular {
                return Response::Error(format!("model `{model}` is not a tabular model"));
            }
            let expected = entry.card.n_features.unwrap_or(0);
            if n_features as usize != expected {
                return Response::Error(format!(
                    "rows have {n_features} features, model `{model}` expects {expected}"
                ));
            }
            if values.is_empty() {
                return Response::Evaluate { scores: Vec::new() };
            }
            let rows = values.len() / expected.max(1);
            match entry.handle.clone().evaluate(values, rows).await {
                Ok(scores) => Response::Evaluate { scores },
                Err(e) => Response::Error(e.to_string()),
            }
        }
    }
}

fn unknown_model(registry: &Registry, name: &str) -> Response {
    Response::Error(format!(
        "unknown model `{name}`; available: {}",
        registry.names().join(", ")
    ))
}
