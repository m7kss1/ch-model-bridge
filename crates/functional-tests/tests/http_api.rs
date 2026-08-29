//! The HTTP surface: what ClickHouse's `aiEmbed` and any OpenAI-compatible
//! client see. Every test drives a freshly spawned daemon serving the built-in
//! `stub` embedder, so the surface is covered without a single model file.

use functional_tests::Daemon;
use serde_json::json;

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn embedding(value: &serde_json::Value, index: usize) -> Vec<f32> {
    value["data"][index]["embedding"]
        .as_array()
        .unwrap_or_else(|| panic!("no embedding at index {index} in {value}"))
        .iter()
        .map(|v| v.as_f64().expect("float") as f32)
        .collect()
}

#[test]
fn health_answers_plain_ok() {
    let daemon = Daemon::stub();
    let response = daemon.get("/health").expect_status(200);
    assert_eq!(response.body, "ok\n");
}

#[test]
fn models_endpoint_describes_the_loaded_models() {
    let daemon = Daemon::stub();
    let body = daemon.get("/v1/models").expect_status(200).json();

    assert_eq!(body["object"], "list");
    let stub = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .find(|card| card["name"] == "stub")
        .expect("the built-in stub model is always registered");

    assert_eq!(stub["kind"], "embedding");
    assert_eq!(stub["dim"], 384);
    assert_eq!(stub["revision"], 0);
    assert_eq!(stub["backend"], "stub");
    assert_eq!(stub["max_batch"], 64);
    assert!(
        stub.get("n_features").is_none(),
        "an embedding card must not carry a feature count: {stub}"
    );
}

#[test]
fn embeddings_accept_a_single_string() {
    let daemon = Daemon::stub();
    let body = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": "hello"}))
        .expect_status(200)
        .json();

    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "stub");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["data"][0]["object"], "embedding");
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(embedding(&body, 0).len(), 384);
}

#[test]
fn embeddings_preserve_input_order_and_index_every_item() {
    let daemon = Daemon::stub();
    let inputs = ["first", "second", "third", "fourth"];
    let body = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": inputs}))
        .expect_status(200)
        .json();

    assert_eq!(body["data"].as_array().unwrap().len(), inputs.len());
    for index in 0..inputs.len() {
        assert_eq!(body["data"][index]["index"], index);
    }

    // Order is meaningful: each vector must be the one for its own text.
    for (index, text) in inputs.iter().enumerate() {
        let alone = daemon
            .post_json("/v1/embeddings", json!({"model": "stub", "input": [text]}))
            .expect_status(200)
            .json();
        assert_eq!(
            embedding(&body, index),
            embedding(&alone, 0),
            "vector {index} does not belong to `{text}`"
        );
    }
}

#[test]
fn embeddings_are_unit_vectors() {
    let daemon = Daemon::stub();
    let body = daemon
        .post_json(
            "/v1/embeddings",
            json!({"model": "stub", "input": ["one", "two words", ""]}),
        )
        .expect_status(200)
        .json();

    for index in 0..3 {
        let length = norm(&embedding(&body, index));
        assert!(
            (length - 1.0).abs() < 1e-5,
            "vector {index} has length {length}, expected 1.0"
        );
    }
}

#[test]
fn embeddings_are_deterministic_across_daemons() {
    let first = {
        let daemon = Daemon::stub();
        embedding(
            &daemon
                .post_json(
                    "/v1/embeddings",
                    json!({"model": "stub", "input": "repeat"}),
                )
                .expect_status(200)
                .json(),
            0,
        )
    };
    let second = {
        let daemon = Daemon::stub();
        embedding(
            &daemon
                .post_json(
                    "/v1/embeddings",
                    json!({"model": "stub", "input": "repeat"}),
                )
                .expect_status(200)
                .json(),
            0,
        )
    };
    assert_eq!(
        first, second,
        "the same text must embed identically in a fresh process"
    );
}

#[test]
fn distinct_texts_get_distinct_vectors() {
    let daemon = Daemon::stub();
    let body = daemon
        .post_json(
            "/v1/embeddings",
            json!({"model": "stub", "input": ["alpha", "beta"]}),
        )
        .expect_status(200)
        .json();
    assert_ne!(embedding(&body, 0), embedding(&body, 1));
}

#[test]
fn usage_counts_tokens_and_cache_hits_cost_nothing() {
    let daemon = Daemon::stub();
    let request = json!({"model": "stub", "input": ["two words", "one"]});

    // The stub has no tokenizer and counts words, so the arithmetic here is
    // exact: 2 + 1.
    let first = daemon
        .post_json("/v1/embeddings", request.clone())
        .expect_status(200)
        .json();
    assert_eq!(first["usage"]["prompt_tokens"], 3);
    assert_eq!(first["usage"]["total_tokens"], 3);

    let second = daemon
        .post_json("/v1/embeddings", request)
        .expect_status(200)
        .json();
    assert_eq!(
        second["usage"]["prompt_tokens"], 0,
        "a fully cached request must bill no tokens"
    );
}

#[test]
fn dimensions_may_restate_the_model_dimension_but_not_change_it() {
    let daemon = Daemon::stub();
    daemon
        .post_json(
            "/v1/embeddings",
            json!({"model": "stub", "input": "x", "dimensions": 384}),
        )
        .expect_status(200);

    let rejected = daemon
        .post_json(
            "/v1/embeddings",
            json!({"model": "stub", "input": "x", "dimensions": 256}),
        )
        .expect_status(400);
    assert!(
        rejected
            .error_message()
            .contains("produces 384-dimensional vectors"),
        "unhelpful message: {}",
        rejected.error_message()
    );
}

#[test]
fn empty_input_is_rejected() {
    let daemon = Daemon::stub();
    let response = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": []}))
        .expect_status(400);
    assert_eq!(response.error_message(), "`input` must not be empty");
}

#[test]
fn unknown_model_is_rejected_with_the_available_ones() {
    let daemon = Daemon::stub();
    let response = daemon
        .post_json("/v1/embeddings", json!({"model": "nope", "input": "x"}))
        .expect_status(400);

    let message = response.error_message();
    assert!(message.starts_with("unknown model `nope`"), "{message}");
    assert!(message.contains("available: stub"), "{message}");
    assert_eq!(
        response.json()["error"]["type"],
        "invalid_request_error",
        "clients key off the OpenAI error envelope"
    );
}

#[test]
fn embedding_models_are_not_accepted_by_rerank_or_evaluate() {
    let daemon = Daemon::stub();

    let rerank = daemon
        .post_json(
            "/v1/rerank",
            json!({"model": "stub", "query": "q", "documents": ["d"]}),
        )
        .expect_status(400);
    assert_eq!(
        rerank.error_message(),
        "model `stub` is an embedding model, not a rerank one"
    );

    let evaluate = daemon
        .post_json("/v1/evaluate", json!({"model": "stub", "rows": [[1.0]]}))
        .expect_status(400);
    assert_eq!(
        evaluate.error_message(),
        "model `stub` is not a tabular model"
    );
}

#[test]
fn rerank_and_evaluate_reject_empty_work() {
    let daemon = Daemon::stub();

    let rerank = daemon
        .post_json(
            "/v1/rerank",
            json!({"model": "stub", "query": "q", "documents": []}),
        )
        .expect_status(400);
    assert_eq!(rerank.error_message(), "`documents` must not be empty");

    let evaluate = daemon
        .post_json("/v1/evaluate", json!({"model": "stub", "rows": []}))
        .expect_status(400);
    assert_eq!(evaluate.error_message(), "`rows` must not be empty");
}

#[test]
fn malformed_requests_are_refused_without_taking_the_daemon_down() {
    let daemon = Daemon::stub();

    let broken_json = daemon.post_raw("/v1/embeddings", "application/json", b"{not json");
    assert!(
        (400..500).contains(&broken_json.status),
        "malformed JSON returned {}",
        broken_json.status
    );

    let missing_field = daemon.post_json("/v1/embeddings", serde_json::json!({"input": "x"}));
    assert!(
        (400..500).contains(&missing_field.status),
        "a request without `model` returned {}",
        missing_field.status
    );

    // Still serving.
    daemon.get("/health").expect_status(200);
}

#[test]
fn unknown_routes_are_404() {
    let daemon = Daemon::stub();
    assert_eq!(daemon.get("/v1/chat/completions").status, 404);
    assert_eq!(daemon.get("/nope").status, 404);
}

#[test]
fn metrics_expose_every_counter_in_prometheus_format() {
    let daemon = Daemon::stub();
    let body = daemon.get("/metrics").expect_status(200).body;

    for counter in [
        "model_bridge_embed_requests_total",
        "model_bridge_rerank_requests_total",
        "model_bridge_evaluate_requests_total",
        "model_bridge_texts_embedded_total",
        "model_bridge_pairs_scored_total",
        "model_bridge_rows_evaluated_total",
        "model_bridge_embed_batches_total",
        "model_bridge_rerank_batches_total",
        "model_bridge_evaluate_batches_total",
        "model_bridge_cache_hits_total",
        "model_bridge_errors_total",
    ] {
        assert!(
            body.contains(&format!("# TYPE {counter} counter\n{counter} ")),
            "{counter} is missing or not in the Prometheus text format:\n{body}"
        );
    }
}

#[test]
fn metrics_count_requests_texts_and_batches() {
    let daemon = Daemon::stub();
    assert_eq!(daemon.metric("model_bridge_embed_requests_total"), 0);

    daemon
        .post_json(
            "/v1/embeddings",
            json!({"model": "stub", "input": ["a", "b", "c"]}),
        )
        .expect_status(200);

    assert_eq!(daemon.metric("model_bridge_embed_requests_total"), 1);
    assert_eq!(daemon.metric("model_bridge_texts_embedded_total"), 3);
    assert_eq!(daemon.metric("model_bridge_embed_batches_total"), 1);
    assert_eq!(daemon.metric("model_bridge_errors_total"), 0);
}

#[test]
fn repeated_texts_are_served_from_the_cache() {
    let daemon = Daemon::stub();
    let request = json!({"model": "stub", "input": ["cached", "cached too"]});

    daemon
        .post_json("/v1/embeddings", request.clone())
        .expect_status(200);
    daemon
        .post_json("/v1/embeddings", request)
        .expect_status(200);

    assert_eq!(
        daemon.metric("model_bridge_texts_embedded_total"),
        2,
        "the second request must not reach the model"
    );
    assert_eq!(daemon.metric("model_bridge_cache_hits_total"), 2);
}

#[test]
fn duplicate_texts_inside_one_request_reach_the_model_once() {
    let daemon = Daemon::stub();
    let input: Vec<&str> = std::iter::repeat_n("same", 10).collect();

    let body = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": input}))
        .expect_status(200)
        .json();

    assert_eq!(body["data"].as_array().unwrap().len(), 10);
    assert_eq!(
        daemon.metric("model_bridge_texts_embedded_total"),
        1,
        "ten copies of one text are one unit of work"
    );
    for index in 1..10 {
        assert_eq!(embedding(&body, 0), embedding(&body, index));
    }
}

#[test]
fn the_cache_is_bounded_by_cache_entries() {
    let daemon = Daemon::builder().arg("--cache-entries").arg("1").start();

    for text in ["a", "b", "a"] {
        daemon
            .post_json("/v1/embeddings", json!({"model": "stub", "input": text}))
            .expect_status(200);
    }

    assert_eq!(
        daemon.metric("model_bridge_texts_embedded_total"),
        3,
        "a one-entry cache must have evicted `a` before it was asked for again"
    );
}

#[test]
fn concurrent_requests_are_merged_into_shared_batches() {
    let daemon = Daemon::stub();
    const CLIENTS: usize = 64;

    let barrier = std::sync::Barrier::new(CLIENTS);
    std::thread::scope(|scope| {
        for client in 0..CLIENTS {
            let barrier = &barrier;
            let daemon = &daemon;
            scope.spawn(move || {
                barrier.wait();
                daemon
                    .post_json(
                        "/v1/embeddings",
                        json!({"model": "stub", "input": format!("client {client}")}),
                    )
                    .expect_status(200);
            });
        }
    });

    let requests = daemon.metric("model_bridge_embed_requests_total");
    let batches = daemon.metric("model_bridge_embed_batches_total");
    assert_eq!(requests, CLIENTS as u64);
    assert_eq!(
        daemon.metric("model_bridge_texts_embedded_total"),
        CLIENTS as u64
    );
    assert!(
        batches < requests,
        "{requests} concurrent requests produced {batches} model runs: nothing was batched"
    );
}

#[test]
fn a_large_batch_returns_one_vector_per_input() {
    let daemon = Daemon::stub();
    let inputs: Vec<String> = (0..500).map(|i| format!("row {i}")).collect();

    let body = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": inputs}))
        .expect_status(200)
        .json();

    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 500);
    for index in 0..500 {
        assert_eq!(embedding(&body, index).len(), 384);
    }
    // max_batch for the stub is 64, so 500 texts are eight model runs.
    assert_eq!(daemon.metric("model_bridge_embed_batches_total"), 8);
}

#[test]
fn stub_dimension_is_configurable() {
    let daemon = Daemon::builder().arg("--stub-dim").arg("8").start();
    let body = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": "x"}))
        .expect_status(200)
        .json();

    let vector = embedding(&body, 0);
    assert_eq!(vector.len(), 8);
    assert!((norm(&vector) - 1.0).abs() < 1e-5);
    assert_eq!(daemon.get("/v1/models").json()["data"][0]["dim"], 8);
}

#[test]
fn unicode_survives_the_round_trip() {
    let daemon = Daemon::stub();
    let inputs = ["привет мир", "🐈 emoji", "日本語テキスト"];
    let body = daemon
        .post_json("/v1/embeddings", json!({"model": "stub", "input": inputs}))
        .expect_status(200)
        .json();

    assert_eq!(body["data"].as_array().unwrap().len(), 3);
    assert_ne!(embedding(&body, 0), embedding(&body, 1));
    assert_eq!(body["usage"]["prompt_tokens"], 2 + 2 + 1);
}

#[test]
fn a_batch_that_overflows_the_cache_does_not_kill_the_worker() {
    // Regression: a batch holding a cached text plus more unique texts than
    // the cache has room for used to evict that text mid-batch, panic the
    // model worker on delivery, and leave the model answering `model worker
    // stopped` forever while /health stayed green.
    let daemon = Daemon::builder().args(["--cache-entries", "2"]).start();

    let seed = |text: &str| {
        daemon
            .post_json("/v1/embeddings", json!({"model": "stub", "input": text}))
            .expect_status(200)
            .json()
    };
    let anchor = embedding(&seed("anchor"), 0);

    for round in 0..15 {
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let barrier = &barrier;
            let daemon = &daemon;
            let anchor = &anchor;
            scope.spawn(move || {
                barrier.wait();
                let flood: Vec<String> = (0..8).map(|i| format!("flood {round} {i}")).collect();
                daemon
                    .post_json("/v1/embeddings", json!({"model": "stub", "input": flood}))
                    .expect_status(200);
            });
            scope.spawn(move || {
                barrier.wait();
                let body = daemon
                    .post_json(
                        "/v1/embeddings",
                        json!({"model": "stub", "input": "anchor"}),
                    )
                    .expect_status(200)
                    .json();
                assert_eq!(
                    &embedding(&body, 0),
                    anchor,
                    "a cache hit came back as a different vector"
                );
            });
        });
        // The flood may have evicted the anchor legitimately; re-seed it so
        // the next round races a genuine cache hit again.
        seed("anchor");
    }

    let after = seed("the worker must still answer");
    assert_eq!(embedding(&after, 0).len(), 384);
}
