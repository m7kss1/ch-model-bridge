//! Regressions against the real ONNX models. These are the numbers the README
//! publishes: vector dimensions, cross-channel parity, the ranking the demos
//! show, and tabular scores against the reference the training script wrote.
//!
//! The tier needs downloaded models (`model-bridge fetch`). Without them the
//! tests skip, unless `MODEL_BRIDGE_REQUIRE_TIERS=models` says they must not.

use functional_tests::{
    error_text, require_model, require_models, rowbinary, run_cli, Daemon, Request, Response,
    TempDir,
};
use serde_json::json;

const E5: &str = "multilingual-e5-small";
const RERANKER: &str = "bge-reranker-base";
const FRAUD: &str = "fraud-demo";

/// The support tickets the README ranks, in its order.
const TICKETS: [&str; 7] = [
    "I sent money a week ago and it still has not arrived",
    "My payment is stuck in processing for five days",
    "I cannot log in to the app after the update",
    "I want to set up automatic payments for my internet bill",
    "The fee was charged twice for one transfer",
    "My card got blocked after three wrong PIN attempts",
    "The app crashes when I open the transaction history",
];

/// A daemon serving exactly the models a test needs: passports are issued by
/// the CLI into a scratch directory, so nothing depends on the repository's
/// own `models.d`.
struct Fixture {
    _scratch: TempDir,
    daemon: Daemon,
}

impl std::ops::Deref for Fixture {
    type Target = Daemon;
    fn deref(&self) -> &Daemon {
        &self.daemon
    }
}

fn fixture(models: &[(&str, &str)], passport_args: &[&str], daemon_args: &[&str]) -> Fixture {
    let scratch = TempDir::new("models");
    let passports = scratch.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();
    let root = functional_tests::models_dir().expect("models directory");

    for (name, kind) in models {
        let dir = root.join(name);
        let mut args = vec![
            "passport",
            dir.to_str().unwrap(),
            "--name",
            name,
            "--kind",
            kind,
            "--passports",
            passports.to_str().unwrap(),
        ];
        args.extend_from_slice(passport_args);
        let run = run_cli(scratch.path(), &args);
        assert_eq!(
            run.code,
            Some(0),
            "passport for {name} failed: {}",
            run.stderr
        );
    }

    let daemon = Daemon::builder()
        .models_dir(&passports)
        .args(daemon_args.iter().map(|arg| arg.to_string()))
        .start();
    Fixture {
        _scratch: scratch,
        daemon,
    }
}

fn embed_http(daemon: &Daemon, model: &str, texts: &[&str]) -> Vec<Vec<f32>> {
    let body = daemon
        .post_json("/v1/embeddings", json!({"model": model, "input": texts}))
        .expect_status(200)
        .json();
    body["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .expect("embedding")
                .iter()
                .map(|v| v.as_f64().expect("float") as f32)
                .collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(vector: &[f32]) -> f32 {
    cosine(vector, vector).sqrt()
}

// ------------------------------------------------------------------ embedding

#[test]
fn the_embedding_model_reports_a_complete_card() {
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);

    let card = daemon.get("/v1/models").expect_status(200).json()["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| card["name"] == E5)
        .expect("the model must be listed")
        .clone();

    assert_eq!(card["kind"], "embedding");
    assert_eq!(card["dim"], 384, "e5-small produces 384 dimensions");
    assert_eq!(card["backend"], "onnx");
    assert_eq!(card["revision"], 1);
    assert_eq!(card["max_batch"], 64);
    assert!(
        daemon.logs().contains("model verified and loaded"),
        "the daemon must report what it verified:\n{}",
        daemon.logs()
    );
}

#[test]
fn embeddings_are_normalized_and_reproducible_across_processes() {
    require_model!(E5);
    let text = "query: where did my transfer go";

    let first = {
        let daemon = fixture(&[(E5, "embedding")], &[], &[]);
        embed_http(&daemon, E5, &[text]).remove(0)
    };
    assert_eq!(first.len(), 384);
    assert!(
        (norm(&first) - 1.0).abs() < 1e-5,
        "vectors must be L2-normalized, got length {}",
        norm(&first)
    );

    let second = {
        let daemon = fixture(&[(E5, "embedding")], &[], &[]);
        embed_http(&daemon, E5, &[text]).remove(0)
    };
    assert_eq!(
        first, second,
        "a pinned model revision must produce bit-identical vectors across runs"
    );
}

#[test]
fn every_channel_returns_the_same_vector_bit_for_bit() {
    // The README's headline claim: `localEmbed` over the UDF pipe and
    // `aiEmbed` over HTTP are the same numbers, not merely similar ones.
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);
    let texts = ["passage: the fee was charged twice", "query: double charge"];

    let over_http = embed_http(&daemon, E5, &texts);

    let Response::Embed { dim, vectors } = daemon.call(&Request::Embed {
        model: E5.to_string(),
        texts: texts.iter().map(|t| t.to_string()).collect(),
    }) else {
        panic!("the socket channel did not answer with vectors");
    };
    assert_eq!(dim, 384);

    let rows: Vec<Vec<u8>> = texts
        .iter()
        .map(|text| {
            let mut row = rowbinary::string(E5);
            row.extend(rowbinary::string(text));
            row
        })
        .collect();
    let stdout = daemon
        .run_client("embed", &rowbinary::block(&rows))
        .expect_success();
    let over_udf = rowbinary::read_f32_arrays(&stdout, texts.len());

    for row in 0..texts.len() {
        assert_eq!(
            over_http[row],
            vectors[row * 384..(row + 1) * 384].to_vec(),
            "row {row}: HTTP and the socket disagree"
        );
        assert_eq!(
            over_http[row], over_udf[row],
            "row {row}: HTTP and the UDF client disagree"
        );
    }
}

#[test]
fn semantic_neighbours_outrank_lexical_ones() {
    // Retrieval regression: the top hit for a paraphrased query must be the
    // ticket that answers it, though they share no words.
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);

    let query = embed_http(&daemon, E5, &["query: where did my transfer go"]).remove(0);
    let passages: Vec<String> = TICKETS.iter().map(|t| format!("passage: {t}")).collect();
    let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
    let vectors = embed_http(&daemon, E5, &refs);

    let mut ranked: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| (cosine(&query, vector), index))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    assert_eq!(
        ranked[0].1, 0,
        "expected the money-transfer ticket on top, got `{}` (ranking: {ranked:?})",
        TICKETS[ranked[0].1]
    );
    assert!(
        ranked[0].0 > 0.75,
        "similarity {} is far outside the band e5-small produces",
        ranked[0].0
    );
}

#[test]
fn the_e5_prefixes_change_the_vector() {
    // e5 was trained with query:/passage: prefixes; if they were stripped
    // somewhere in the pipeline, retrieval quality would silently drop.
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);
    let vectors = embed_http(&daemon, E5, &["query: refund", "passage: refund", "refund"]);

    assert_ne!(vectors[0], vectors[1]);
    assert_ne!(vectors[0], vectors[2]);
}

#[test]
fn text_longer_than_the_position_limit_is_truncated_not_refused() {
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);
    let long = "transfer ".repeat(4000);

    let vectors = embed_http(&daemon, E5, &[&long]);
    assert_eq!(vectors[0].len(), 384);
    assert!((norm(&vectors[0]) - 1.0).abs() < 1e-5);
}

#[test]
fn the_passport_batch_size_bounds_the_model_runs() {
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &["--max-batch", "2"], &[]);

    let texts: Vec<String> = (0..6).map(|i| format!("row {i}")).collect();
    daemon
        .post_json("/v1/embeddings", json!({"model": E5, "input": texts}))
        .expect_status(200);

    assert_eq!(daemon.get("/v1/models").json()["data"][0]["max_batch"], 2);
    assert_eq!(
        daemon.metric("model_bridge_embed_batches_total"),
        3,
        "six texts at max_batch=2 are exactly three model runs"
    );
}

#[test]
fn repeated_texts_never_reach_the_model_twice() {
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);
    let request = json!({"model": E5, "input": ["query: refund", "passage: refund"]});

    daemon
        .post_json("/v1/embeddings", request.clone())
        .expect_status(200);
    let second = daemon
        .post_json("/v1/embeddings", request)
        .expect_status(200)
        .json();

    assert_eq!(daemon.metric("model_bridge_texts_embedded_total"), 2);
    assert_eq!(daemon.metric("model_bridge_cache_hits_total"), 2);
    assert_eq!(second["usage"]["prompt_tokens"], 0);
}

#[test]
fn concurrent_queries_share_batches_on_a_real_model() {
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);
    const CLIENTS: usize = 24;

    let barrier = std::sync::Barrier::new(CLIENTS);
    std::thread::scope(|scope| {
        for client in 0..CLIENTS {
            let barrier = &barrier;
            let daemon = &daemon.daemon;
            scope.spawn(move || {
                barrier.wait();
                daemon
                    .post_json(
                        "/v1/embeddings",
                        json!({"model": E5, "input": format!("passage: ticket {client}")}),
                    )
                    .expect_status(200);
            });
        }
    });

    let requests = daemon.metric("model_bridge_embed_requests_total");
    let batches = daemon.metric("model_bridge_embed_batches_total");
    assert_eq!(requests, CLIENTS as u64);
    assert!(
        batches < requests,
        "{requests} concurrent requests ran as {batches} batches: they were not merged"
    );
}

#[test]
fn a_block_that_mixes_models_keeps_its_row_order() {
    // One SQL query may name several models; the client groups rows per model
    // and must put the answers back in the original order.
    require_model!(E5);
    let daemon = fixture(&[(E5, "embedding")], &[], &[]);
    let texts = ["first", "second", "third", "fourth"];

    let rows: Vec<Vec<u8>> = texts
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let model = if index % 2 == 0 { E5 } else { "stub" };
            let mut row = rowbinary::string(model);
            row.extend(rowbinary::string(text));
            row
        })
        .collect();

    let stdout = daemon
        .run_client("embed", &rowbinary::block(&rows))
        .expect_success();
    let mixed = rowbinary::read_f32_arrays(&stdout, texts.len());

    for (index, text) in texts.iter().enumerate() {
        let model = if index % 2 == 0 { E5 } else { "stub" };
        let alone = embed_http(&daemon, model, &[text]).remove(0);
        assert_eq!(
            mixed[index], alone,
            "row {index} was answered by the wrong model or landed out of order"
        );
    }
}

// --------------------------------------------------------------------- rerank

#[test]
fn the_reranker_puts_the_direct_answer_first() {
    require_model!(RERANKER);
    let daemon = fixture(&[(RERANKER, "rerank")], &[], &[]);

    let body = daemon
        .post_json(
            "/v1/rerank",
            json!({
                "model": RERANKER,
                "query": "transfer has not reached the recipient",
                "documents": TICKETS,
            }),
        )
        .expect_status(200)
        .json();

    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), TICKETS.len());
    assert_eq!(
        results[0]["index"],
        0,
        "the direct answer must rank first, got `{}`",
        TICKETS[results[0]["index"].as_u64().unwrap() as usize]
    );

    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["relevance_score"].as_f64().unwrap())
        .collect();
    for pair in scores.windows(2) {
        assert!(pair[0] >= pair[1], "results are not sorted: {scores:?}");
    }
    assert!(
        scores[0] - scores[1] > 3.0,
        "the direct answer should stand clear of the rest: {scores:?}"
    );
}

#[test]
fn rerank_truncates_to_top_n() {
    require_model!(RERANKER);
    let daemon = fixture(&[(RERANKER, "rerank")], &[], &[]);

    let body = daemon
        .post_json(
            "/v1/rerank",
            json!({
                "model": RERANKER,
                "query": "transfer has not reached the recipient",
                "documents": TICKETS,
                "top_n": 3,
            }),
        )
        .expect_status(200)
        .json();

    assert_eq!(body["results"].as_array().unwrap().len(), 3);
    assert_eq!(body["results"][0]["index"], 0);
}

#[test]
fn rerank_over_the_udf_channel_matches_http() {
    require_model!(RERANKER);
    let daemon = fixture(&[(RERANKER, "rerank")], &[], &[]);
    let query = "transfer has not reached the recipient";

    let http = daemon
        .post_json(
            "/v1/rerank",
            json!({"model": RERANKER, "query": query, "documents": TICKETS}),
        )
        .expect_status(200)
        .json();
    let mut by_index = vec![0f32; TICKETS.len()];
    for result in http["results"].as_array().unwrap() {
        by_index[result["index"].as_u64().unwrap() as usize] =
            result["relevance_score"].as_f64().unwrap() as f32;
    }

    let rows: Vec<Vec<u8>> = TICKETS
        .iter()
        .map(|document| {
            let mut row = rowbinary::string(RERANKER);
            row.extend(rowbinary::string(query));
            row.extend(rowbinary::string(document));
            row
        })
        .collect();
    let stdout = daemon
        .run_client("rerank", &rowbinary::block(&rows))
        .expect_success();

    assert_eq!(
        rowbinary::read_f32_scalars(&stdout, TICKETS.len()),
        by_index,
        "the UDF channel and HTTP disagree on the scores"
    );
}

#[test]
fn a_rerank_model_is_not_an_embedding_model() {
    require_model!(RERANKER);
    let daemon = fixture(&[(RERANKER, "rerank")], &[], &[]);

    let response = daemon
        .post_json("/v1/embeddings", json!({"model": RERANKER, "input": "x"}))
        .expect_status(400);
    assert_eq!(
        response.error_message(),
        format!("model `{RERANKER}` is a rerank model, not an embedding one")
    );
}

// -------------------------------------------------------------------- tabular

#[test]
fn tabular_scores_match_the_training_reference() {
    require_model!(FRAUD);
    let Some(reference) = functional_tests::fraud_reference() else {
        // Its own tier name on purpose: the reference scores come out of the
        // training script, so an operator who merely installed the model has
        // no way to produce them, and must not be failed for it.
        functional_tests::skip(
            "tabular-reference",
            "no tmp/fraud-expected.json; run examples/train_fraud_model.py",
        );
        return;
    };

    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    let rows: Vec<Vec<f32>> = expected["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    let reference_scores: Vec<f32> = expected["expected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let daemon = fixture(&[(FRAUD, "tabular")], &[], &[]);
    let body = daemon
        .post_json("/v1/evaluate", json!({"model": FRAUD, "rows": rows}))
        .expect_status(200)
        .json();

    let scores: Vec<f32> = body["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(scores.len(), reference_scores.len());

    let worst = scores
        .iter()
        .zip(&reference_scores)
        .map(|(got, want)| (got - want).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-5,
        "ONNX drifted from the sklearn reference by {worst} over {} rows",
        scores.len()
    );
}

#[test]
fn the_tabular_model_learned_the_amount_by_hour_interaction() {
    // The demo's whole point: the same amount is scored differently at night.
    require_model!(FRAUD);
    let daemon = fixture(&[(FRAUD, "tabular")], &[], &[]);

    let body = daemon
        .post_json(
            "/v1/evaluate",
            json!({"model": FRAUD, "rows": [
                [4800.0, 2.0, 0.0, 0.1],   // night
                [4800.0, 14.0, 0.0, 0.1],  // same amount, afternoon
                [9500.0, 3.0, 1.0, 0.9],   // night, new device, risky merchant
            ]}),
        )
        .expect_status(200)
        .json();

    let scores: Vec<f64> = body["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    assert!(
        scores[0] > scores[1] + 0.5,
        "the night transaction must score far above the daytime one: {scores:?}"
    );
    assert!(
        scores[2] > scores[0],
        "the worst row must score highest: {scores:?}"
    );
}

#[test]
fn a_wrong_feature_count_is_refused_on_every_channel() {
    require_model!(FRAUD);
    let daemon = fixture(&[(FRAUD, "tabular")], &[], &[]);

    let http = daemon
        .post_json(
            "/v1/evaluate",
            json!({"model": FRAUD, "rows": [[1.0, 2.0, 3.0]]}),
        )
        .expect_status(400);
    assert_eq!(
        http.error_message(),
        format!("row 0 has 3 features, model `{FRAUD}` expects 4")
    );

    let socket = daemon.call(&Request::Evaluate {
        model: FRAUD.to_string(),
        n_features: 3,
        values: vec![1.0, 2.0, 3.0],
    });
    assert_eq!(
        error_text(&socket),
        format!("rows have 3 features, model `{FRAUD}` expects 4")
    );

    let mut row = rowbinary::string(FRAUD);
    row.extend(rowbinary::f32_array(&[1.0, 2.0, 3.0]));
    let stderr = daemon
        .run_client("evaluate", &rowbinary::block(&[row]))
        .expect_failure();
    assert!(stderr.contains("expects 4"), "{stderr}");
}

#[test]
fn evaluate_over_the_udf_channel_matches_the_json_channel() {
    // The UDF pipe carries features at the model's own width, so the scores
    // must be the ones the JSON channel produces from the same float32 rows.
    require_model!(FRAUD);
    let daemon = fixture(&[(FRAUD, "tabular")], &[], &[]);
    let features = [[9500.0f32, 3.0, 1.0, 0.9], [250.0, 11.0, 0.0, 0.2]];

    let rows: Vec<Vec<u8>> = features
        .iter()
        .map(|row| {
            let mut encoded = rowbinary::string(FRAUD);
            encoded.extend(rowbinary::f32_array(row));
            encoded
        })
        .collect();
    let stdout = daemon
        .run_client("evaluate", &rowbinary::block(&rows))
        .expect_success();
    let over_udf = rowbinary::read_f32_scalars(&stdout, features.len());

    let http = daemon
        .post_json("/v1/evaluate", json!({"model": FRAUD, "rows": features}))
        .expect_status(200)
        .json();
    let over_http: Vec<f32> = http["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    assert_eq!(over_udf, over_http);
}

// ------------------------------------------------------------------ fail-close

#[test]
fn a_tampered_real_model_stops_the_daemon() {
    let source = require_model!(FRAUD);
    let scratch = TempDir::new("tampered-real");
    let model = scratch.child("fraud-demo");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::copy(source.join("model.onnx"), model.join("model.onnx")).unwrap();

    let passports = scratch.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();
    let run = run_cli(
        scratch.path(),
        &[
            "passport",
            model.to_str().unwrap(),
            "--name",
            FRAUD,
            "--kind",
            "tabular",
            "--passports",
            passports.to_str().unwrap(),
        ],
    );
    assert_eq!(run.code, Some(0), "{}", run.stderr);

    // One flipped byte in a model that otherwise loads fine.
    functional_tests::corrupt_file(&model.join("model.onnx"));

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("checksum mismatch"), "{logs}");
}

#[test]
fn one_unusable_model_stops_the_whole_daemon() {
    // No partial serving: a daemon that answers for some models and silently
    // omits others is worse than one that refuses to start.
    let fraud = require_model!(FRAUD);
    let scratch = TempDir::new("partial-serving");
    let passports = scratch.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();

    let run = run_cli(
        scratch.path(),
        &[
            "passport",
            fraud.to_str().unwrap(),
            "--name",
            FRAUD,
            "--kind",
            "tabular",
            "--passports",
            passports.to_str().unwrap(),
        ],
    );
    assert_eq!(run.code, Some(0), "{}", run.stderr);

    // A second passport whose files were never there.
    std::fs::write(
        passports.join("zz-missing.toml"),
        "name = \"zz-missing\"\nkind = \"tabular\"\ndir = \"/nonexistent\"\nrevision = 1\n\n\
         [sha256]\n\"model.onnx\" = \"00\"\n",
    )
    .unwrap();

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("zz-missing"), "{logs}");
}

#[test]
fn the_model_flag_loads_without_a_passport_and_says_so() {
    let dir = require_model!(E5);
    let daemon = Daemon::builder()
        .arg("--model")
        .arg(format!("unverified={}", dir.display()))
        .start();

    assert!(
        daemon
            .logs()
            .contains("loading without a passport: files are not checksum-verified"),
        "an unverified load must be loud:\n{}",
        daemon.logs()
    );
    let vectors = embed_http(&daemon, "unverified", &["query: hello"]);
    assert_eq!(vectors[0].len(), 384);
    assert_eq!(daemon.get("/v1/models").json()["data"][0]["revision"], 0);
}

#[test]
fn the_models_directory_the_tier_uses_is_the_documented_one() {
    // Guards the tier itself: a typo in MODEL_BRIDGE_MODELS_DIR would
    // otherwise turn every model test into a silent skip.
    let root = require_models!();
    assert!(
        root.join(E5).exists() || root.join(FRAUD).exists() || root.join(RERANKER).exists(),
        "{} holds none of the catalog models",
        root.display()
    );
}
