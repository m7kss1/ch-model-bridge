//! `bridge-client` as ClickHouse runs it: a pooled process fed RowBinary
//! blocks on stdin, answering on stdout, failing loudly on stderr. These tests
//! stand in for the database, so the pipe contract is checked without one.

use std::io::{Read, Write};
use std::process::{Command, Stdio};

use functional_tests::{bin, rowbinary, Daemon};
use serde_json::json;

/// Bytes one `Array(Float32)` of `dim` floats occupies in the reply.
fn row_size(dim: usize) -> usize {
    rowbinary::varuint(dim as u64).len() + dim * 4
}

fn embed_row(model: &str, text: &str) -> Vec<u8> {
    let mut row = rowbinary::string(model);
    row.extend(rowbinary::string(text));
    row
}

#[test]
fn a_block_of_rows_matches_the_http_channel() {
    let daemon = Daemon::stub();
    let texts = ["first row", "second row", "third row"];
    let block = rowbinary::block(
        &texts
            .iter()
            .map(|text| embed_row("stub", text))
            .collect::<Vec<_>>(),
    );

    let stdout = daemon.run_client("embed", &block).expect_success();
    let vectors = rowbinary::read_f32_arrays(&stdout, texts.len());

    for (index, text) in texts.iter().enumerate() {
        assert_eq!(vectors[index].len(), 384);
        let http: Vec<f32> = daemon
            .post_json("/v1/embeddings", json!({"model": "stub", "input": [text]}))
            .expect_status(200)
            .json()["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(vectors[index], http, "row {index} differs from HTTP");
    }
}

#[test]
fn one_process_serves_several_blocks() {
    let daemon = Daemon::stub();
    let mut input = Vec::new();
    let sizes = [1usize, 4, 2];
    for (block, size) in sizes.iter().enumerate() {
        let rows: Vec<Vec<u8>> = (0..*size)
            .map(|row| embed_row("stub", &format!("block {block} row {row}")))
            .collect();
        input.extend(rowbinary::block(&rows));
    }

    let stdout = daemon.run_client("embed", &input).expect_success();
    let total: usize = sizes.iter().sum();
    let vectors = rowbinary::read_f32_arrays(&stdout, total);
    assert_eq!(vectors.len(), total);
    assert!(vectors.iter().all(|vector| vector.len() == 384));
    assert_eq!(daemon.metric("model_bridge_embed_requests_total"), 3);
}

#[test]
fn every_block_is_answered_before_the_next_one_arrives() {
    // ClickHouse writes the next block only after reading the previous
    // answer. Buffering a reply until stdin closes would deadlock the query,
    // so the flush per block is part of the contract.
    let daemon = Daemon::stub();
    let mut child = Command::new(bin("bridge-client"))
        .arg("embed")
        .arg("--socket")
        .arg(daemon.socket())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bridge-client");

    // Nothing here may block forever: if the client withholds the first
    // answer, the watchdog turns the hang into a failed read.
    let pid = child.id();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(30));
        let _ = Command::new("kill").arg(pid.to_string()).status();
    });

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    for block in 0..3 {
        let rows = vec![embed_row("stub", &format!("block {block}"))];
        stdin
            .write_all(&rowbinary::block(&rows))
            .expect("write block");
        stdin.flush().expect("flush block");

        let mut reply = vec![0u8; row_size(384)];
        stdout
            .read_exact(&mut reply)
            .unwrap_or_else(|e| panic!("block {block} was not answered before the next one: {e}"));
        assert_eq!(rowbinary::read_f32_arrays(&reply, 1)[0].len(), 384);
    }

    drop(stdin);
    let status = child.wait().expect("wait for bridge-client");
    assert!(status.success(), "client exited with {status}");
}

#[test]
fn an_empty_block_produces_no_rows() {
    let daemon = Daemon::stub();
    let stdout = daemon.run_client("embed", b"0\n").expect_success();
    assert!(stdout.is_empty(), "an empty block produced {stdout:?}");
}

#[test]
fn end_of_input_without_a_block_is_a_clean_exit() {
    let daemon = Daemon::stub();
    let stdout = daemon.run_client("embed", b"").expect_success();
    assert!(stdout.is_empty());
}

#[test]
fn a_missing_daemon_fails_the_query_with_the_socket_path() {
    let socket = std::path::Path::new("/tmp/model-bridge-does-not-exist.sock");
    let run = functional_tests::run_bridge_client("embed", socket, b"0\n");
    let stderr = run.expect_failure();

    assert!(
        stderr.contains("bridge-client: connecting to the daemon at"),
        "unhelpful stderr: {stderr}"
    );
    assert!(
        stderr.contains("model-bridge-does-not-exist.sock"),
        "{stderr}"
    );
}

#[test]
fn an_unknown_model_fails_the_query_instead_of_returning_rows() {
    let daemon = Daemon::stub();
    let block = rowbinary::block(&[embed_row("ghost", "text")]);
    let run = daemon.run_client("embed", &block);

    assert!(run.stdout.is_empty(), "partial results were written");
    let stderr = run.expect_failure();
    assert!(stderr.contains("unknown model `ghost`"), "{stderr}");
    assert!(stderr.contains("available: stub"), "{stderr}");
}

#[test]
fn the_wrong_model_kind_fails_the_query() {
    let daemon = Daemon::stub();
    let mut row = rowbinary::string("stub");
    row.extend(rowbinary::string("query"));
    row.extend(rowbinary::string("document"));
    let stderr = daemon
        .run_client("rerank", &rowbinary::block(&[row]))
        .expect_failure();
    assert!(stderr.contains("not a rerank model"), "{stderr}");

    let mut row = rowbinary::string("stub");
    row.extend(rowbinary::f32_array(&[1.0, 2.0]));
    let stderr = daemon
        .run_client("evaluate", &rowbinary::block(&[row]))
        .expect_failure();
    assert!(stderr.contains("not a tabular model"), "{stderr}");
}

#[test]
fn bad_arguments_explain_the_usage() {
    let daemon = Daemon::stub();
    let stderr = daemon.run_client("translate", b"").expect_failure();
    assert!(stderr.contains("unknown mode `translate`"), "{stderr}");
    assert!(
        stderr.contains("expected `embed`, `rerank` or `evaluate`"),
        "{stderr}"
    );

    let bare = Command::new(bin("bridge-client"))
        .output()
        .expect("run bridge-client without arguments");
    assert!(!bare.status.success());
    assert!(
        String::from_utf8_lossy(&bare.stderr)
            .contains("usage: bridge-client <embed|rerank|evaluate> --socket PATH"),
        "{}",
        String::from_utf8_lossy(&bare.stderr)
    );
}

#[test]
fn a_malformed_chunk_header_fails_the_query() {
    let daemon = Daemon::stub();
    let stderr = daemon
        .run_client("embed", b"not-a-number\n")
        .expect_failure();
    assert!(stderr.contains("bad chunk header"), "{stderr}");
}

#[test]
fn a_large_block_is_answered_in_one_round_trip() {
    let daemon = Daemon::stub();
    const ROWS: usize = 5000;
    let rows: Vec<Vec<u8>> = (0..ROWS)
        .map(|row| embed_row("stub", &format!("row {row}")))
        .collect();

    let stdout = daemon
        .run_client("embed", &rowbinary::block(&rows))
        .expect_success();

    let vectors = rowbinary::read_f32_arrays(&stdout, ROWS);
    assert_eq!(vectors.len(), ROWS);
    assert!(vectors.iter().all(|vector| vector.len() == 384));
    assert_eq!(
        daemon.metric("model_bridge_embed_requests_total"),
        1,
        "a block must cost exactly one daemon round trip"
    );
    assert_eq!(
        daemon.metric("model_bridge_texts_embedded_total"),
        ROWS as u64
    );
}

#[test]
fn unusual_strings_survive_the_pipe() {
    let daemon = Daemon::stub();
    let texts = ["", "  ", "привет", "🐈", &"long ".repeat(2000)];
    let rows: Vec<Vec<u8>> = texts.iter().map(|text| embed_row("stub", text)).collect();

    let stdout = daemon
        .run_client("embed", &rowbinary::block(&rows))
        .expect_success();
    let vectors = rowbinary::read_f32_arrays(&stdout, texts.len());

    assert_eq!(vectors.len(), texts.len());
    assert!(vectors.iter().all(|vector| vector.len() == 384));
    assert_ne!(vectors[0], vectors[2]);
}

#[test]
fn invalid_utf8_is_replaced_rather_than_failing_the_block() {
    // A ClickHouse `String` is raw bytes. Failing a whole block over one bad
    // byte would be worse than embedding the replacement character.
    let daemon = Daemon::stub();
    let mut row = rowbinary::string("stub");
    row.extend(rowbinary::varuint(3));
    row.extend_from_slice(&[0xff, 0xfe, 0x41]);

    let stdout = daemon
        .run_client("embed", &rowbinary::block(&[row]))
        .expect_success();
    assert_eq!(rowbinary::read_f32_arrays(&stdout, 1)[0].len(), 384);
}
