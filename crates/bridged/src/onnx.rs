use std::path::Path;
use std::sync::{Condvar, Mutex};

use anyhow::{anyhow, bail};
use ort::session::builder::{GraphOptimizationLevel, PrepackedWeights};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::engine::{EmbedOutput, Embedder, Evaluator, RerankOutput, Reranker};

/// Truncation fallback when neither the passport nor `tokenizer.json` names a
/// limit: the position budget of the BERT and XLM-R family. Long-context
/// encoders (nomic, jina, e5-mistral) must carry `max_tokens` in their
/// passport, or their 8k inputs are cut here.
const DEFAULT_MAX_TOKENS: usize = 512;

/// Raw first output of a transformer run: shape, data, per-row token counts
/// and the attention mask.
type RawOutput = (Vec<usize>, Vec<f32>, Vec<usize>, Vec<i64>);

/// A fixed set of interchangeable sessions over one model. `Session::run`
/// needs exclusive access, so this is what turns `sessions = k` in a passport
/// into k concurrent inference streams: a caller borrows whichever session is
/// free and parks when all k are busy — callers already sit on the blocking
/// pool, so parking the thread is fine.
struct SessionPool {
    idle: Mutex<Vec<Session>>,
    returned: Condvar,
}

impl SessionPool {
    /// Builds `count` sessions from `model_path` with shared configuration.
    /// The sessions share one prepacked-weights container, so the weight
    /// buffers that ONNX Runtime pre-packs (the bulk of a transformer) are
    /// held once, not `count` times.
    fn build(model_path: &Path, count: usize) -> anyhow::Result<Self> {
        let count = count.max(1);
        // `ort::Error` is not `Send + Sync`, so it cannot ride through `?`
        // into `anyhow`; every ort call site converts the error via its
        // message.
        let builder = Session::builder().map_err(|e| anyhow!("session builder: {e}"))?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("session options: {e}"))?;
        if count > 1 {
            // The pool must not multiply the thread footprint: split the
            // machine between the sessions, so k concurrent runs use about
            // as many threads as one session with the ort default would.
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            builder = builder
                .with_intra_threads((cores / count).max(1))
                .map_err(|e| anyhow!("session threads: {e}"))?
                .with_prepacked_weights(&PrepackedWeights::new())
                .map_err(|e| anyhow!("session weights: {e}"))?;
        }
        let mut sessions = Vec::with_capacity(count);
        for _ in 0..count {
            sessions.push(
                builder
                    .clone()
                    .commit_from_file(model_path)
                    .map_err(|e| anyhow!("{}: {e}", model_path.display()))?,
            );
        }
        Ok(Self {
            idle: Mutex::new(sessions),
            returned: Condvar::new(),
        })
    }

    /// Borrows a session for the duration of `f`. A panicking `f` retires the
    /// borrowed session instead of returning a possibly corrupt one.
    fn with<R>(&self, f: impl FnOnce(&mut Session) -> R) -> R {
        let mut idle = self.idle.lock().expect("session pool poisoned");
        let mut session = loop {
            match idle.pop() {
                Some(session) => break session,
                None => idle = self.returned.wait(idle).expect("session pool poisoned"),
            }
        };
        drop(idle);
        let result = f(&mut session);
        self.idle
            .lock()
            .expect("session pool poisoned")
            .push(session);
        self.returned.notify_one();
        result
    }

    /// Runs `f` against one session outside the borrow discipline, for
    /// load-time introspection before the pool starts serving.
    fn inspect<R>(&self, f: impl FnOnce(&Session) -> R) -> R {
        let idle = self.idle.lock().expect("session pool poisoned");
        f(idle.first().expect("a pool is never empty"))
    }
}

struct LoadedModel {
    tokenizer: Tokenizer,
    pool: SessionPool,
    needs_token_type_ids: bool,
    /// The effective truncation limit, for the model card.
    max_tokens: usize,
}

impl LoadedModel {
    /// Loads `model.onnx` with its `tokenizer.json` from `dir` into a pool of
    /// `sessions` sessions. The pair is inseparable: encoding text with a
    /// tokenizer from another model does not fail, it silently degrades every
    /// produced result.
    fn load(dir: &Path, sessions: usize, max_tokens: Option<usize>) -> anyhow::Result<Self> {
        let tokenizer_path = dir.join("tokenizer.json");
        let model_path = dir.join("model.onnx");

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("{}: {e}", tokenizer_path.display()))?;

        // Model inputs must be rectangular and fit the position limit, while
        // `tokenizer.json` files commonly ship with no padding or truncation
        // sections at all.
        if tokenizer.get_padding().is_none() {
            // The pad id must be the model's own: XLM-R uses `<pad>`, BERT
            // uses `[PAD]`, and the `PaddingParams` default of id 0 is a real
            // token in some vocabularies.
            let (pad_token, pad_id) = ["<pad>", "[PAD]"]
                .iter()
                .find_map(|t| tokenizer.token_to_id(t).map(|id| (t.to_string(), id)))
                .unwrap_or_else(|| ("[PAD]".to_string(), 0));
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id,
                pad_token,
                ..Default::default()
            }));
        }
        // The truncation limit, in priority order: the passport (the operator
        // knows the model's context), then whatever `tokenizer.json` ships,
        // then the conservative family default. Only the limit is overridden;
        // a shipped strategy, stride or direction survives.
        let mut truncation = tokenizer
            .get_truncation()
            .cloned()
            .unwrap_or(TruncationParams {
                max_length: DEFAULT_MAX_TOKENS,
                ..Default::default()
            });
        if let Some(limit) = max_tokens {
            truncation.max_length = limit;
        }
        let max_tokens = truncation.max_length;
        tokenizer
            .with_truncation(Some(truncation))
            .map_err(|e| anyhow!("truncation setup: {e}"))?;

        let pool = SessionPool::build(&model_path, sessions)?;
        let needs_token_type_ids = pool.inspect(|session| {
            session
                .inputs()
                .iter()
                .any(|input| input.name() == "token_type_ids")
        });

        Ok(Self {
            tokenizer,
            pool,
            needs_token_type_ids,
            max_tokens,
        })
    }

    /// Runs the transformer and returns the raw first output as
    /// (shape, data, per-row token counts, mask). The first output is the
    /// token states or logits depending on the export; callers validate the
    /// shape.
    fn run(&self, encodings: &[Encoding]) -> anyhow::Result<RawOutput> {
        let batch = encodings.len();
        let seq = encodings.first().map_or(0, |e| e.get_ids().len());

        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for encoding in encodings {
            ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
            mask.extend(encoding.get_attention_mask().iter().map(|&m| m as i64));
        }
        let row_tokens: Vec<usize> = encodings
            .iter()
            .map(|e| e.get_attention_mask().iter().filter(|&&m| m == 1).count())
            .collect();

        let tensor = |data: Vec<i64>| {
            Tensor::from_array(([batch, seq], data)).map_err(|e| anyhow!("input tensor: {e}"))
        };
        let ids_tensor = tensor(ids)?;
        let mask_tensor = tensor(mask.clone())?;

        let (shape, data) = self.pool.with(|session| -> anyhow::Result<_> {
            let run_result = if self.needs_token_type_ids {
                let type_ids = tensor(vec![0i64; batch * seq])?;
                session.run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                    "token_type_ids" => type_ids
                ])
            } else {
                session.run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor
                ])
            };
            let outputs = run_result.map_err(|e| anyhow!("inference: {e}"))?;

            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("output tensor: {e}"))?;
            let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            Ok((shape, data.to_vec()))
        })?;
        Ok((shape, data, row_tokens, mask))
    }
}

/// Sentence embedder: mean-pools token states over the attention mask and
/// L2-normalizes.
pub struct OnnxEmbedder {
    model: LoadedModel,
    dim: usize,
}

impl OnnxEmbedder {
    pub fn load(dir: &Path, sessions: usize, max_tokens: Option<usize>) -> anyhow::Result<Self> {
        let model = LoadedModel::load(dir, sessions, max_tokens)?;
        let mut embedder = Self { model, dim: 0 };
        // The output dimensionality is taken from an actual run: graph
        // metadata frequently declares the hidden dimension as dynamic.
        embedder.dim = embedder
            .embed(std::slice::from_ref(&"probe".to_string()))?
            .vectors[0]
            .len();
        Ok(embedder)
    }

    /// The effective truncation limit, for the model card.
    pub fn max_tokens(&self) -> usize {
        self.model.max_tokens
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, texts: &[String]) -> anyhow::Result<EmbedOutput> {
        let encodings = self
            .model
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let (shape, data, tokens, mask) = self.model.run(&encodings)?;
        let [batch, seq, hidden] = shape[..] else {
            bail!("expected a [batch, seq, hidden] output, got shape {shape:?}");
        };

        let mut vectors = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut vector = vec![0f32; hidden];
            let mut count = 0usize;
            for t in 0..seq {
                if mask[b * seq + t] == 0 {
                    continue;
                }
                count += 1;
                let offset = (b * seq + t) * hidden;
                for (v, x) in vector.iter_mut().zip(&data[offset..offset + hidden]) {
                    *v += *x;
                }
            }
            let scale = 1.0 / count.max(1) as f32;
            for v in &mut vector {
                *v *= scale;
            }
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for v in &mut vector {
                *v /= norm;
            }
            vectors.push(vector);
        }

        Ok(EmbedOutput { vectors, tokens })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Cross-encoder reranker: encodes (query, document) pairs jointly and reads a
/// single relevance logit per pair.
pub struct OnnxCrossEncoder {
    model: LoadedModel,
}

impl OnnxCrossEncoder {
    pub fn load(dir: &Path, sessions: usize, max_tokens: Option<usize>) -> anyhow::Result<Self> {
        let reranker = Self {
            model: LoadedModel::load(dir, sessions, max_tokens)?,
        };
        // Validates the output shape once at load time instead of on the first
        // user query.
        reranker.score_pairs(std::slice::from_ref(&(
            "probe".to_string(),
            "probe".to_string(),
        )))?;
        Ok(reranker)
    }

    /// The effective truncation limit, for the model card.
    pub fn max_tokens(&self) -> usize {
        self.model.max_tokens
    }
}

/// Arbitrary ONNX model over numeric features (kind `tabular`): classic ML
/// scoring, exported from sklearn / XGBoost / LightGBM / CatBoost. There is
/// no tokenizer; the input is a [batch, n_features] float matrix.
pub struct OnnxTabular {
    pool: SessionPool,
    input_name: String,
    n_features: usize,
}

impl OnnxTabular {
    pub fn load(dir: &Path, sessions: usize) -> anyhow::Result<Self> {
        let model_path = dir.join("model.onnx");
        let pool = SessionPool::build(&model_path, sessions)?;

        let (input_name, n_features) = pool.inspect(|session| {
            let [input] = session.inputs() else {
                bail!(
                    "a tabular model must have exactly one input, this one has {}",
                    session.inputs().len()
                );
            };
            // sklearn classifiers export two outputs (label + probabilities);
            // that shape is ambiguous to score, so it is rejected here with the
            // export-time fix instead of failing on the first query.
            if session.outputs().len() != 1 {
                bail!(
                    "a tabular model must have exactly one output, this one has {}; \
                     export a regressor, or reduce a classifier to its probability \
                     column at conversion time",
                    session.outputs().len()
                );
            }
            let shape = input
                .dtype()
                .tensor_shape()
                .ok_or_else(|| anyhow!("model input `{}` is not a tensor", input.name()))?;
            // The feature count comes from the graph itself ([-1, n] input), so a
            // passport does not have to duplicate it.
            let n_features = match shape[..] {
                [_, n] if n > 0 => n as usize,
                _ => bail!("expected a [batch, n_features] input, model declares {shape:?}"),
            };
            Ok((input.name().to_string(), n_features))
        })?;

        let tabular = Self {
            input_name,
            pool,
            n_features,
        };
        // Validates the output shape once at load time on a dummy row.
        tabular.evaluate(&vec![0f32; n_features])?;
        Ok(tabular)
    }
}

impl Evaluator for OnnxTabular {
    fn evaluate(&self, values: &[f32]) -> anyhow::Result<Vec<f32>> {
        if !values.len().is_multiple_of(self.n_features) {
            bail!(
                "{} values do not form rows of {} features",
                values.len(),
                self.n_features
            );
        }
        let batch = values.len() / self.n_features;
        let tensor = Tensor::from_array(([batch, self.n_features], values.to_vec()))
            .map_err(|e| anyhow!("input tensor: {e}"))?;

        let (shape, data) = self.pool.with(|session| -> anyhow::Result<_> {
            let outputs = session
                .run(ort::inputs![self.input_name.as_str() => tensor])
                .map_err(|e| anyhow!("inference: {e}"))?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("output tensor: {e}"))?;
            let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            Ok((shape, data.to_vec()))
        })?;

        match shape[..] {
            [n] if n == batch => Ok(data),
            [n, 1] if n == batch => Ok(data),
            _ => bail!("expected a [batch] or [batch, 1] score output, got shape {shape:?}"),
        }
    }

    fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Reranker for OnnxCrossEncoder {
    fn score_pairs(&self, pairs: &[(String, String)]) -> anyhow::Result<RerankOutput> {
        let encodings = self
            .model
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let (shape, data, tokens, _mask) = self.model.run(&encodings)?;

        let scores = match shape[..] {
            [batch] if batch == pairs.len() => data,
            [batch, 1] if batch == pairs.len() => data,
            _ => bail!("expected a [batch] or [batch, 1] logit output, got shape {shape:?}"),
        };

        Ok(RerankOutput { scores, tokens })
    }
}
