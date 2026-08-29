use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide counters, rendered by `/metrics` in the Prometheus text
/// format. Counters only: gauges and histograms can be added when something
/// actually needs them.
#[derive(Default)]
pub struct Metrics {
    pub embed_requests: AtomicU64,
    pub rerank_requests: AtomicU64,
    pub evaluate_requests: AtomicU64,
    pub texts_embedded: AtomicU64,
    pub pairs_scored: AtomicU64,
    pub rows_evaluated: AtomicU64,
    pub embed_batches: AtomicU64,
    pub rerank_batches: AtomicU64,
    pub evaluate_batches: AtomicU64,
    pub cache_hits: AtomicU64,
    pub errors: AtomicU64,
}

impl Metrics {
    pub fn add(counter: &AtomicU64, value: u64) {
        counter.fetch_add(value, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        for (name, counter) in [
            ("model_bridge_embed_requests_total", &self.embed_requests),
            ("model_bridge_rerank_requests_total", &self.rerank_requests),
            (
                "model_bridge_evaluate_requests_total",
                &self.evaluate_requests,
            ),
            ("model_bridge_texts_embedded_total", &self.texts_embedded),
            ("model_bridge_pairs_scored_total", &self.pairs_scored),
            ("model_bridge_rows_evaluated_total", &self.rows_evaluated),
            ("model_bridge_embed_batches_total", &self.embed_batches),
            ("model_bridge_rerank_batches_total", &self.rerank_batches),
            (
                "model_bridge_evaluate_batches_total",
                &self.evaluate_batches,
            ),
            ("model_bridge_cache_hits_total", &self.cache_hits),
            ("model_bridge_errors_total", &self.errors),
        ] {
            out.push_str(&format!(
                "# TYPE {name} counter\n{name} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }
        out
    }
}
