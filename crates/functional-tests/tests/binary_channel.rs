//! The unix-socket channel: length-prefixed frames, one response per request.
//! This is what a pooled `bridge-client` process speaks, so its error paths
//! decide whether a ClickHouse query fails loudly or silently returns
//! nonsense.

use functional_tests::{error_text, wire_call, Daemon, Request, Response};
use protocol::wire::{self, MAX_FRAME};
use serde_json::json;

fn embed(model: &str, texts: &[&str]) -> Request {
    Request::Embed {
        model: model.to_string(),
        texts: texts.iter().map(|t| t.to_string()).collect(),
    }
}

#[test]
fn embedding_over_the_socket_matches_the_http_channel() {
    let daemon = Daemon::stub();

    let Response::Embed { dim, vectors } = daemon.call(&embed("stub", &["hello", "world"])) else {
        panic!("expected an embed response");
    };
    assert_eq!(dim, 384);
    assert_eq!(vectors.len(), 2 * 384);

    let http = daemon
        .post_json(
            "/v1/embeddings",
            json!({"model": "stub", "input": ["hello", "world"]}),
        )
        .expect_status(200)
        .json();

    for row in 0..2 {
        let over_http: Vec<f32> = http["data"][row]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(
            &vectors[row * 384..(row + 1) * 384],
            over_http.as_slice(),
            "row {row} differs between the socket and HTTP channels"
        );
    }
}

#[test]
fn an_empty_request_still_reports_the_dimension() {
    let daemon = Daemon::stub();
    let Response::Embed { dim, vectors } = daemon.call(&embed("stub", &[])) else {
        panic!("expected an embed response");
    };
    assert_eq!(dim, 384);
    assert!(vectors.is_empty());
}

#[test]
fn unknown_models_are_named_along_with_the_available_ones() {
    let daemon = Daemon::stub();
    let response = daemon.call(&embed("ghost", &["x"]));
    let message = error_text(&response);
    assert!(message.starts_with("unknown model `ghost`"), "{message}");
    assert!(message.contains("available: stub"), "{message}");
}

#[test]
fn a_model_of_the_wrong_kind_is_refused_on_every_task() {
    let daemon = Daemon::stub();

    let rerank = daemon.call(&Request::Rerank {
        model: "stub".to_string(),
        pairs: vec![("q".to_string(), "d".to_string())],
    });
    assert_eq!(error_text(&rerank), "model `stub` is not a rerank model");

    let evaluate = daemon.call(&Request::Evaluate {
        model: "stub".to_string(),
        n_features: 2,
        values: vec![1.0, 2.0],
    });
    assert_eq!(error_text(&evaluate), "model `stub` is not a tabular model");
}

#[test]
fn a_protocol_version_mismatch_says_which_side_speaks_what() {
    let daemon = Daemon::stub();
    let mut frame = wire::encode_request(&embed("stub", &["x"]));
    frame[0] = 99;

    let response = daemon.connect().call_raw(&frame);
    assert_eq!(
        error_text(&response),
        "protocol version mismatch: daemon speaks v1, client sent v99"
    );
}

#[test]
fn an_unknown_task_kind_is_named() {
    let daemon = Daemon::stub();
    let mut frame = wire::encode_request(&embed("stub", &["x"]));
    frame[1] = 42;

    let response = daemon.connect().call_raw(&frame);
    assert_eq!(error_text(&response), "unknown task kind 42");
}

#[test]
fn a_truncated_frame_is_rejected_instead_of_being_guessed_at() {
    let daemon = Daemon::stub();
    let frame = wire::encode_request(&embed("stub", &["hello"]));

    let response = daemon.connect().call_raw(&frame[..frame.len() - 3]);
    assert_eq!(error_text(&response), "frame truncated");
}

#[test]
fn trailing_bytes_are_rejected() {
    let daemon = Daemon::stub();
    let mut frame = wire::encode_request(&embed("stub", &["hello"]));
    frame.extend_from_slice(b"junk");

    let response = daemon.connect().call_raw(&frame);
    assert_eq!(error_text(&response), "4 trailing bytes in frame");
}

#[test]
fn an_invalid_utf8_model_name_is_rejected() {
    let daemon = Daemon::stub();
    let mut frame = vec![protocol::PROTOCOL_VERSION, 0];
    frame.extend_from_slice(&4u32.to_le_bytes());
    frame.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
    frame.extend_from_slice(&0u32.to_le_bytes());

    let response = daemon.connect().call_raw(&frame);
    assert!(
        error_text(&response).starts_with("invalid utf-8"),
        "{}",
        error_text(&response)
    );
}

#[test]
fn a_rejected_frame_does_not_poison_the_connection() {
    let daemon = Daemon::stub();
    let mut socket = daemon.connect();

    let mut broken = wire::encode_request(&embed("stub", &["x"]));
    broken[1] = 42;
    assert_eq!(
        error_text(&socket.call_raw(&broken)),
        "unknown task kind 42"
    );

    // The pooled client keeps one connection for the life of the process; a
    // decoding error must not cost it that connection.
    let Response::Embed { dim, .. } = socket.call(&embed("stub", &["x"])) else {
        panic!("the connection stopped working after a rejected frame");
    };
    assert_eq!(dim, 384);
}

#[test]
fn one_connection_serves_many_requests_in_order() {
    let daemon = Daemon::stub();
    let mut socket = daemon.connect();

    let mut seen = Vec::new();
    for index in 0..25 {
        let text = format!("request {index}");
        let Response::Embed { vectors, .. } = socket.call(&embed("stub", &[&text])) else {
            panic!("request {index} did not produce vectors");
        };
        assert_eq!(vectors.len(), 384);
        seen.push(vectors);
    }
    for index in 1..seen.len() {
        assert_ne!(seen[0], seen[index], "responses got crossed between frames");
    }
}

#[test]
fn concurrent_connections_are_served_independently() {
    let daemon = Daemon::stub();
    const CLIENTS: usize = 16;

    let barrier = std::sync::Barrier::new(CLIENTS);
    std::thread::scope(|scope| {
        for client in 0..CLIENTS {
            let barrier = &barrier;
            let socket = daemon.socket();
            scope.spawn(move || {
                let text = format!("client {client}");
                barrier.wait();
                let Response::Embed { dim, vectors } = wire_call(socket, &embed("stub", &[&text]))
                else {
                    panic!("client {client} got no vectors");
                };
                assert_eq!(dim, 384);
                assert_eq!(vectors.len(), 384);
            });
        }
    });

    assert_eq!(
        daemon.metric("model_bridge_embed_requests_total"),
        CLIENTS as u64
    );
    assert!(daemon.metric("model_bridge_embed_batches_total") < CLIENTS as u64);
}

#[test]
fn a_frame_over_the_size_limit_closes_the_connection() {
    let daemon = Daemon::stub();
    let mut socket = daemon.connect();
    socket.send_raw(MAX_FRAME + 1, b"");

    assert!(
        socket.read_frame().is_err(),
        "an oversized length prefix must not be allocated for"
    );
    // The daemon itself keeps serving everyone else.
    daemon.get("/health").expect_status(200);
    let Response::Embed { dim, .. } = daemon.call(&embed("stub", &["still here"])) else {
        panic!("the daemon stopped serving after an oversized frame");
    };
    assert_eq!(dim, 384);
}

#[test]
fn a_length_prefix_longer_than_the_payload_ends_the_connection() {
    let daemon = Daemon::stub();
    let mut socket = daemon.connect();
    socket.send_raw(1024, b"short");
    socket.shutdown_write();

    assert!(
        socket.read_frame().is_err(),
        "a frame that never arrives must end the connection, not hang forever"
    );
    daemon.get("/health").expect_status(200);
}

#[test]
fn the_daemon_refuses_the_socket_channel_when_it_is_not_configured() {
    let daemon = Daemon::builder().without_socket().start();
    assert!(
        !daemon.socket().exists(),
        "no --socket must mean no socket file"
    );
    assert!(!daemon.logs().contains("binary channel listening"));
    daemon.get("/health").expect_status(200);
}
