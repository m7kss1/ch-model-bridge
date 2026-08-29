use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use protocol::passport::ModelKind;

use crate::metrics::Metrics;
use crate::registry::Registry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<Registry>,
    pub metrics: Arc<Metrics>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/v1/evaluate", post(evaluate))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok\n"
}

async fn models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({ "object": "list", "data": state.registry.cards() }))
}

async fn metrics(State(state): State<AppState>) -> String {
    state.metrics.render()
}

/// `input` accepts a single string or an array: ClickHouse `aiEmbed` sends an
/// array with one entry per row of a block.
#[derive(Deserialize)]
struct EmbeddingsRequest {
    model: String,
    input: EmbeddingsInput,
    #[serde(default)]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    model: String,
    data: Vec<EmbeddingItem>,
    usage: Usage,
}

#[derive(Serialize)]
struct EmbeddingItem {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

/// Errors follow the OpenAI error envelope: clients, including ClickHouse,
/// surface `error.message` to the user as-is.
fn api_error(status: StatusCode, message: impl Into<String>) -> ApiError {
    (
        status,
        Json(json!({ "error": { "message": message.into(), "type": "invalid_request_error" } })),
    )
}

fn unknown_model(registry: &Registry, name: &str) -> ApiError {
    api_error(
        StatusCode::BAD_REQUEST,
        format!(
            "unknown model `{name}`; available: {}",
            registry.names().join(", ")
        ),
    )
}

async fn embeddings(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let texts = match request.input {
        EmbeddingsInput::One(text) => vec![text],
        EmbeddingsInput::Many(texts) => texts,
    };
    if texts.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "`input` must not be empty",
        ));
    }

    let entry = state
        .registry
        .resolve(&request.model)
        .ok_or_else(|| unknown_model(&state.registry, &request.model))?;
    if entry.card.kind != ModelKind::Embedding {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "model `{}` is a rerank model, not an embedding one",
                request.model
            ),
        ));
    }

    if let Some(dimensions) = request.dimensions {
        if Some(dimensions) != entry.card.dim {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "model `{}` produces {}-dimensional vectors and does not support `dimensions` = {}",
                    request.model,
                    entry.card.dim.unwrap_or(0),
                    dimensions
                ),
            ));
        }
    }

    let handle = entry.handle.clone();
    let model = request.model;

    let output = handle
        .embed(texts)
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data = output
        .vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingItem {
            object: "embedding",
            index,
            embedding,
        })
        .collect();

    Ok(Json(EmbeddingsResponse {
        object: "list",
        model,
        data,
        usage: Usage {
            prompt_tokens: output.prompt_tokens,
            total_tokens: output.prompt_tokens,
        },
    }))
}

#[derive(Deserialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<usize>,
}

#[derive(Serialize)]
struct RerankResponse {
    model: String,
    /// Sorted by `relevance_score` descending; `index` refers to the position
    /// in the request `documents`.
    results: Vec<RerankResult>,
    usage: Usage,
}

#[derive(Serialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

async fn rerank(
    State(state): State<AppState>,
    Json(request): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, ApiError> {
    if request.documents.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "`documents` must not be empty",
        ));
    }

    let entry = state
        .registry
        .resolve(&request.model)
        .ok_or_else(|| unknown_model(&state.registry, &request.model))?;
    if entry.card.kind != ModelKind::Rerank {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "model `{}` is an embedding model, not a rerank one",
                request.model
            ),
        ));
    }

    let pairs: Vec<(String, String)> = request
        .documents
        .iter()
        .map(|doc| (request.query.clone(), doc.clone()))
        .collect();

    let handle = entry.handle.clone();
    let output = handle
        .rerank(pairs)
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut results: Vec<RerankResult> = output
        .scores
        .into_iter()
        .enumerate()
        .map(|(index, relevance_score)| RerankResult {
            index,
            relevance_score,
        })
        .collect();
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(top_n) = request.top_n {
        results.truncate(top_n);
    }

    Ok(Json(RerankResponse {
        model: request.model,
        results,
        usage: Usage {
            prompt_tokens: output.prompt_tokens,
            total_tokens: output.prompt_tokens,
        },
    }))
}

#[derive(Deserialize)]
struct EvaluateRequest {
    model: String,
    /// Feature rows in the model's training-time feature order.
    rows: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct EvaluateResponse {
    model: String,
    /// One score per request row, in request order.
    scores: Vec<f32>,
}

async fn evaluate(
    State(state): State<AppState>,
    Json(request): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    if request.rows.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "`rows` must not be empty",
        ));
    }

    let entry = state
        .registry
        .resolve(&request.model)
        .ok_or_else(|| unknown_model(&state.registry, &request.model))?;
    if entry.card.kind != ModelKind::Tabular {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("model `{}` is not a tabular model", request.model),
        ));
    }

    let n_features = entry.card.n_features.unwrap_or(0);
    let mut values = Vec::with_capacity(request.rows.len() * n_features);
    for (index, row) in request.rows.iter().enumerate() {
        if row.len() != n_features {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "row {index} has {} features, model `{}` expects {n_features}",
                    row.len(),
                    request.model
                ),
            ));
        }
        values.extend_from_slice(row);
    }

    let handle = entry.handle.clone();
    let rows = request.rows.len();
    let scores = handle
        .evaluate(values, rows)
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(EvaluateResponse {
        model: request.model,
        scores,
    }))
}
