use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct EmbedOutput {
    pub vectors: Vec<Vec<f32>>,
    /// Tokens fed to the model, one count per input text: the dispatcher
    /// attributes usage to the request each text came from.
    pub tokens: Vec<usize>,
}

/// A loaded embedding model. Implementations return exactly one vector per
/// input text, in input order, L2-normalized to unit length.
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> anyhow::Result<EmbedOutput>;
    fn dim(&self) -> usize;
}

pub struct RerankOutput {
    pub scores: Vec<f32>,
    /// One token count per scored pair.
    pub tokens: Vec<usize>,
}

/// A loaded cross-encoder. Scores each (query, document) pair independently;
/// higher means more relevant. Scores are raw model logits: comparable within
/// one model, not across models.
pub trait Reranker: Send + Sync {
    fn score_pairs(&self, pairs: &[(String, String)]) -> anyhow::Result<RerankOutput>;
}

/// A loaded tabular model: one score per feature row. `values` is row-major
/// with exactly `n_features` floats per row; feature order is part of the
/// model contract and callers must match the training-time order.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, values: &[f32]) -> anyhow::Result<Vec<f32>>;
    fn n_features(&self) -> usize;
}

/// Development stand-in for when no real model is configured: maps a text to a
/// deterministic unit vector derived from its hash. The vectors carry no
/// semantics — nearest-neighbor results over them are meaningless — so this is
/// only good for exercising formats, channels and the ClickHouse integration.
pub struct StubEngine {
    dim: usize,
}

impl StubEngine {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for StubEngine {
    fn embed(&self, texts: &[String]) -> anyhow::Result<EmbedOutput> {
        let vectors = texts
            .iter()
            .map(|text| pseudo_embedding(text, self.dim))
            .collect();
        // The stub has no tokenizer; a word count stands in for a token count.
        let tokens = texts
            .iter()
            .map(|t| t.split_whitespace().count().max(1))
            .collect();
        Ok(EmbedOutput { vectors, tokens })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

fn pseudo_embedding(text: &str, dim: usize) -> Vec<f32> {
    // `DefaultHasher::new` uses fixed keys, so the same text yields the same
    // vector across runs — callers rely on that for repeatable plumbing tests.
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let mut state = hasher.finish();

    let mut vector: Vec<f32> = (0..dim)
        .map(|_| {
            state = splitmix64(state);
            ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect();

    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut vector {
        *x /= norm;
    }
    vector
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
