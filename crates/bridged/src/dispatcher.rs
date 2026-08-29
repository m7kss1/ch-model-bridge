use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;
use lru::LruCache;
use tokio::sync::{mpsc, oneshot};

use crate::engine::{Embedder, Evaluator, Reranker};
use crate::metrics::Metrics;

/// How long the worker waits for more requests before running a partial batch.
/// Small enough to be invisible in a single request's latency, large enough to
/// merge requests that arrive together.
const BATCH_WINDOW: Duration = Duration::from_millis(2);

pub struct EmbedReply {
    pub vectors: Vec<Vec<f32>>,
    /// Tokens actually run through the model for this request; cache hits
    /// contribute zero.
    pub prompt_tokens: usize,
}

pub struct RerankReply {
    pub scores: Vec<f32>,
    pub prompt_tokens: usize,
}

enum Job {
    Embed {
        texts: Vec<String>,
        reply: oneshot::Sender<anyhow::Result<EmbedReply>>,
    },
    Rerank {
        pairs: Vec<(String, String)>,
        reply: oneshot::Sender<anyhow::Result<RerankReply>>,
    },
    Evaluate {
        /// Row-major feature matrix; the worker knows `n_features` from its
        /// engine, so only the row count is carried alongside.
        values: Vec<f32>,
        rows: usize,
        reply: oneshot::Sender<anyhow::Result<Vec<f32>>>,
    },
}

impl Job {
    fn size(&self) -> usize {
        match self {
            Job::Embed { texts, .. } => texts.len(),
            Job::Rerank { pairs, .. } => pairs.len(),
            Job::Evaluate { rows, .. } => *rows,
        }
    }
}

/// Requests to one model go through one worker task: batches are formed across
/// every client (HTTP, socket, concurrent queries), so the hardware sees a
/// steady stream of full batches instead of per-caller dribble. The bounded
/// queue is the backpressure: when the model does not keep up, senders wait.
#[derive(Clone)]
pub struct ModelHandle {
    tx: mpsc::Sender<Job>,
}

impl ModelHandle {
    pub async fn embed(&self, texts: Vec<String>) -> anyhow::Result<EmbedReply> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Embed { texts, reply })
            .await
            .map_err(|_| anyhow!("model worker stopped"))?;
        rx.await
            .map_err(|_| anyhow!("model worker dropped the request"))?
    }

    pub async fn rerank(&self, pairs: Vec<(String, String)>) -> anyhow::Result<RerankReply> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Rerank { pairs, reply })
            .await
            .map_err(|_| anyhow!("model worker stopped"))?;
        rx.await
            .map_err(|_| anyhow!("model worker dropped the request"))?
    }

    pub async fn evaluate(&self, values: Vec<f32>, rows: usize) -> anyhow::Result<Vec<f32>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Evaluate {
                values,
                rows,
                reply,
            })
            .await
            .map_err(|_| anyhow!("model worker stopped"))?;
        rx.await
            .map_err(|_| anyhow!("model worker dropped the request"))?
    }
}

pub fn spawn_embed_worker(
    engine: Arc<dyn Embedder>,
    max_batch: usize,
    cache_entries: usize,
    metrics: Arc<Metrics>,
) -> ModelHandle {
    let (tx, rx) = mpsc::channel(max_batch.max(1) * 4);
    tokio::spawn(embed_worker(
        rx,
        engine,
        max_batch.max(1),
        cache_entries,
        metrics,
    ));
    ModelHandle { tx }
}

pub fn spawn_rerank_worker(
    engine: Arc<dyn Reranker>,
    max_batch: usize,
    metrics: Arc<Metrics>,
) -> ModelHandle {
    let (tx, rx) = mpsc::channel(max_batch.max(1) * 4);
    tokio::spawn(rerank_worker(rx, engine, max_batch.max(1), metrics));
    ModelHandle { tx }
}

pub fn spawn_evaluate_worker(
    engine: Arc<dyn Evaluator>,
    max_batch: usize,
    metrics: Arc<Metrics>,
) -> ModelHandle {
    let (tx, rx) = mpsc::channel(max_batch.max(1) * 4);
    tokio::spawn(evaluate_worker(rx, engine, max_batch.max(1), metrics));
    ModelHandle { tx }
}

/// Pulls the first job, then keeps collecting until the batch window closes or
/// `max_batch` items are queued.
async fn collect_batch(rx: &mut mpsc::Receiver<Job>, first: Job, max_batch: usize) -> Vec<Job> {
    let mut jobs = vec![first];
    let mut queued = jobs[0].size();
    let deadline = tokio::time::Instant::now() + BATCH_WINDOW;
    while queued < max_batch {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(job)) => {
                queued += job.size();
                jobs.push(job);
            }
            _ => break,
        }
    }
    jobs
}

async fn embed_worker(
    mut rx: mpsc::Receiver<Job>,
    engine: Arc<dyn Embedder>,
    max_batch: usize,
    cache_entries: usize,
    metrics: Arc<Metrics>,
) {
    // The worker is the cache's only user, so no lock is needed. The key is
    // the text alone: the cache lives inside one (model, revision) worker.
    let mut cache: LruCache<String, Arc<(Vec<f32>, usize)>> =
        LruCache::new(NonZeroUsize::new(cache_entries.max(1)).unwrap());

    while let Some(first) = rx.recv().await {
        let jobs = collect_batch(&mut rx, first, max_batch).await;
        Metrics::add(&metrics.embed_requests, jobs.len() as u64);

        // Deduplicate texts across jobs and resolve cache hits up front; the
        // model only sees what is genuinely new. Hits are taken out of the
        // cache *before* anything is inserted: a batch carrying more unique
        // texts than the cache holds would otherwise evict entries this very
        // batch still needs. `get` also refreshes their recency.
        let mut unique: Vec<String> = Vec::new();
        let mut position: HashMap<String, usize> = HashMap::new();
        let mut cached: HashMap<String, Arc<(Vec<f32>, usize)>> = HashMap::new();
        for job in &jobs {
            let Job::Embed { texts, .. } = job else {
                continue;
            };
            for text in texts {
                if position.contains_key(text) || cached.contains_key(text) {
                    continue;
                }
                if let Some(entry) = cache.get(text) {
                    cached.insert(text.clone(), Arc::clone(entry));
                    continue;
                }
                position.insert(text.clone(), unique.len());
                unique.push(text.clone());
            }
        }

        let mut computed: Vec<Arc<(Vec<f32>, usize)>> = Vec::with_capacity(unique.len());
        let mut failure: Option<String> = None;
        for chunk in unique.chunks(max_batch) {
            let engine = Arc::clone(&engine);
            let chunk_owned = chunk.to_vec();
            let result = tokio::task::spawn_blocking(move || engine.embed(&chunk_owned)).await;
            match result {
                Ok(Ok(output)) => {
                    Metrics::add(&metrics.embed_batches, 1);
                    Metrics::add(&metrics.texts_embedded, chunk.len() as u64);
                    for (vector, tokens) in output.vectors.into_iter().zip(output.tokens) {
                        computed.push(Arc::new((vector, tokens)));
                    }
                }
                Ok(Err(e)) => {
                    failure = Some(e.to_string());
                    break;
                }
                Err(e) => {
                    failure = Some(format!("inference task: {e}"));
                    break;
                }
            }
        }

        if let Some(message) = failure {
            Metrics::add(&metrics.errors, 1);
            for job in jobs {
                if let Job::Embed { reply, .. } = job {
                    let _ = reply.send(Err(anyhow!("{message}")));
                }
            }
            continue;
        }

        for (text, index) in &position {
            cache.put(text.clone(), Arc::clone(&computed[*index]));
        }

        for job in jobs {
            let Job::Embed { texts, reply } = job else {
                continue;
            };
            let mut vectors = Vec::with_capacity(texts.len());
            let mut prompt_tokens = 0usize;
            let mut hits = 0u64;
            for text in &texts {
                match position.get(text) {
                    Some(index) => {
                        let entry = &computed[*index];
                        vectors.push(entry.0.clone());
                        prompt_tokens += entry.1;
                    }
                    None => {
                        // Resolved from the cache before this batch ran, so
                        // no later eviction can invalidate it.
                        let entry = cached.get(text).expect("text neither computed nor cached");
                        vectors.push(entry.0.clone());
                        hits += 1;
                    }
                }
            }
            Metrics::add(&metrics.cache_hits, hits);
            let _ = reply.send(Ok(EmbedReply {
                vectors,
                prompt_tokens,
            }));
        }
    }
}

async fn rerank_worker(
    mut rx: mpsc::Receiver<Job>,
    engine: Arc<dyn Reranker>,
    max_batch: usize,
    metrics: Arc<Metrics>,
) {
    // No cache here: (query, document) pairs almost never repeat, unlike
    // single texts.
    while let Some(first) = rx.recv().await {
        let jobs = collect_batch(&mut rx, first, max_batch).await;
        Metrics::add(&metrics.rerank_requests, jobs.len() as u64);

        let mut all_pairs: Vec<(String, String)> = Vec::new();
        let mut spans: Vec<usize> = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let Job::Rerank { pairs, .. } = job else {
                continue;
            };
            spans.push(pairs.len());
            all_pairs.extend(pairs.iter().cloned());
        }

        let mut scores: Vec<f32> = Vec::with_capacity(all_pairs.len());
        let mut tokens: Vec<usize> = Vec::with_capacity(all_pairs.len());
        let mut failure: Option<String> = None;
        for chunk in all_pairs.chunks(max_batch) {
            let engine = Arc::clone(&engine);
            let chunk_owned = chunk.to_vec();
            let result =
                tokio::task::spawn_blocking(move || engine.score_pairs(&chunk_owned)).await;
            match result {
                Ok(Ok(output)) => {
                    Metrics::add(&metrics.rerank_batches, 1);
                    Metrics::add(&metrics.pairs_scored, chunk.len() as u64);
                    scores.extend(output.scores);
                    tokens.extend(output.tokens);
                }
                Ok(Err(e)) => {
                    failure = Some(e.to_string());
                    break;
                }
                Err(e) => {
                    failure = Some(format!("inference task: {e}"));
                    break;
                }
            }
        }

        if let Some(message) = failure {
            Metrics::add(&metrics.errors, 1);
            for job in jobs {
                if let Job::Rerank { reply, .. } = job {
                    let _ = reply.send(Err(anyhow!("{message}")));
                }
            }
            continue;
        }

        let mut offset = 0usize;
        for (job, span) in jobs.into_iter().zip(spans) {
            let Job::Rerank { reply, .. } = job else {
                continue;
            };
            let job_scores = scores[offset..offset + span].to_vec();
            let prompt_tokens = tokens[offset..offset + span].iter().sum();
            offset += span;
            let _ = reply.send(Ok(RerankReply {
                scores: job_scores,
                prompt_tokens,
            }));
        }
    }
}

async fn evaluate_worker(
    mut rx: mpsc::Receiver<Job>,
    engine: Arc<dyn Evaluator>,
    max_batch: usize,
    metrics: Arc<Metrics>,
) {
    let n_features = engine.n_features();
    // No cache: feature rows are continuous values and effectively never
    // repeat.
    while let Some(first) = rx.recv().await {
        let jobs = collect_batch(&mut rx, first, max_batch).await;
        Metrics::add(&metrics.evaluate_requests, jobs.len() as u64);

        let mut all_values: Vec<f32> = Vec::new();
        let mut spans: Vec<usize> = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let Job::Evaluate { values, rows, .. } = job else {
                continue;
            };
            spans.push(*rows);
            all_values.extend_from_slice(values);
        }

        let mut scores: Vec<f32> = Vec::with_capacity(all_values.len() / n_features.max(1));
        let mut failure: Option<String> = None;
        for chunk in all_values.chunks(max_batch * n_features) {
            let engine = Arc::clone(&engine);
            let chunk_owned = chunk.to_vec();
            let result = tokio::task::spawn_blocking(move || engine.evaluate(&chunk_owned)).await;
            match result {
                Ok(Ok(output)) => {
                    Metrics::add(&metrics.evaluate_batches, 1);
                    Metrics::add(&metrics.rows_evaluated, output.len() as u64);
                    scores.extend(output);
                }
                Ok(Err(e)) => {
                    failure = Some(e.to_string());
                    break;
                }
                Err(e) => {
                    failure = Some(format!("inference task: {e}"));
                    break;
                }
            }
        }

        if let Some(message) = failure {
            Metrics::add(&metrics.errors, 1);
            for job in jobs {
                if let Job::Evaluate { reply, .. } = job {
                    let _ = reply.send(Err(anyhow!("{message}")));
                }
            }
            continue;
        }

        let mut offset = 0usize;
        for (job, span) in jobs.into_iter().zip(spans) {
            let Job::Evaluate { reply, .. } = job else {
                continue;
            };
            let job_scores = scores[offset..offset + span].to_vec();
            offset += span;
            let _ = reply.send(Ok(job_scores));
        }
    }
}
