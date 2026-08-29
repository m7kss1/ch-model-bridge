use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context};
use protocol::passport::{ModelKind, Passport};
use serde::Serialize;

use crate::dispatcher::{
    spawn_embed_worker, spawn_evaluate_worker, spawn_rerank_worker, ModelHandle,
};
use crate::engine::{Embedder, Evaluator, Reranker};
use crate::metrics::Metrics;
use crate::onnx::{OnnxCrossEncoder, OnnxEmbedder, OnnxTabular};

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub name: String,
    pub kind: ModelKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim: Option<usize>,
    /// Tabular models only: how many floats each feature row must contain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_features: Option<usize>,
    pub revision: u32,
    pub max_batch: usize,
    pub backend: &'static str,
}

pub struct Entry {
    pub card: ModelCard,
    pub handle: ModelHandle,
}

/// Name-to-model dispatch table, built once at startup. Registration spawns
/// the model's worker task, so the registry owns the lifetime of the whole
/// serving pipeline.
pub struct Registry {
    entries: HashMap<String, Entry>,
    metrics: Arc<Metrics>,
    cache_entries: usize,
}

impl Registry {
    pub fn new(metrics: Arc<Metrics>, cache_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            metrics,
            cache_entries,
        }
    }

    pub fn register_embedder(
        &mut self,
        name: &str,
        backend: &'static str,
        revision: u32,
        max_batch: usize,
        engine: Arc<dyn Embedder>,
    ) {
        let card = ModelCard {
            name: name.to_string(),
            kind: ModelKind::Embedding,
            dim: Some(engine.dim()),
            n_features: None,
            revision,
            max_batch,
            backend,
        };
        let handle = spawn_embed_worker(
            engine,
            max_batch,
            self.cache_entries,
            Arc::clone(&self.metrics),
        );
        self.entries
            .insert(name.to_string(), Entry { card, handle });
    }

    pub fn register_reranker(
        &mut self,
        name: &str,
        backend: &'static str,
        revision: u32,
        max_batch: usize,
        engine: Arc<dyn Reranker>,
    ) {
        let card = ModelCard {
            name: name.to_string(),
            kind: ModelKind::Rerank,
            dim: None,
            n_features: None,
            revision,
            max_batch,
            backend,
        };
        let handle = spawn_rerank_worker(engine, max_batch, Arc::clone(&self.metrics));
        self.entries
            .insert(name.to_string(), Entry { card, handle });
    }

    pub fn register_evaluator(
        &mut self,
        name: &str,
        backend: &'static str,
        revision: u32,
        max_batch: usize,
        engine: Arc<dyn Evaluator>,
    ) {
        let card = ModelCard {
            name: name.to_string(),
            kind: ModelKind::Tabular,
            dim: None,
            n_features: Some(engine.n_features()),
            revision,
            max_batch,
            backend,
        };
        let handle = spawn_evaluate_worker(engine, max_batch, Arc::clone(&self.metrics));
        self.entries
            .insert(name.to_string(), Entry { card, handle });
    }

    /// Loads every `*.toml` passport under `dir`, verifying checksums before a
    /// model is allowed into memory. A bad passport fails startup: serving a
    /// subset of the configured models silently is worse than not starting.
    pub fn load_models_dir(&mut self, dir: &Path) -> anyhow::Result<usize> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| dir.display().to_string())?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
            .collect();
        paths.sort();

        for path in &paths {
            let passport = Passport::load(path)?;
            let model_dir = passport.resolved_dir(path);
            let started = std::time::Instant::now();
            passport
                .verify(&model_dir)
                .with_context(|| format!("passport {}", path.display()))?;
            match passport.kind {
                ModelKind::Embedding => {
                    let engine = OnnxEmbedder::load(&model_dir)
                        .with_context(|| format!("loading `{}`", passport.name))?;
                    self.register_embedder(
                        &passport.name,
                        "onnx",
                        passport.revision,
                        passport.max_batch,
                        Arc::new(engine),
                    );
                }
                ModelKind::Rerank => {
                    let engine = OnnxCrossEncoder::load(&model_dir)
                        .with_context(|| format!("loading `{}`", passport.name))?;
                    self.register_reranker(
                        &passport.name,
                        "onnx",
                        passport.revision,
                        passport.max_batch,
                        Arc::new(engine),
                    );
                }
                ModelKind::Tabular => {
                    let engine = OnnxTabular::load(&model_dir)
                        .with_context(|| format!("loading `{}`", passport.name))?;
                    self.register_evaluator(
                        &passport.name,
                        "onnx",
                        passport.revision,
                        passport.max_batch,
                        Arc::new(engine),
                    );
                }
            }
            tracing::info!(
                model = passport.name,
                kind = ?passport.kind,
                revision = passport.revision,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "model verified and loaded"
            );
        }

        if paths.is_empty() {
            bail!("{}: no passports found", dir.display());
        }
        Ok(paths.len())
    }

    pub fn resolve(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    pub fn cards(&self) -> Vec<&ModelCard> {
        let mut cards: Vec<_> = self.entries.values().map(|entry| &entry.card).collect();
        cards.sort_by(|a, b| a.name.cmp(&b.name));
        cards
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.entries.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}
