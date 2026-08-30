//! End to end through a real ClickHouse. Everything else in this suite stands
//! in for the database; this file uses it, so the executable-UDF wiring, the
//! generated XML, block sizes and mutations are exercised the way an operator
//! meets them.
//!
//! Needs a `clickhouse` binary (PATH or `MODEL_BRIDGE_CLICKHOUSE`); without
//! one the tests skip unless `MODEL_BRIDGE_REQUIRE_TIERS=clickhouse`.

use std::path::PathBuf;
use std::process::Command;

use functional_tests::{bin, require_clickhouse, require_model, run_cli, Daemon, TempDir};
use serde_json::json;

const E5: &str = "multilingual-e5-small";
const RERANKER: &str = "bge-reranker-base";
const FRAUD: &str = "fraud-demo";

const TICKETS: [&str; 5] = [
    "I sent money a week ago and it still has not arrived",
    "My payment is stuck in processing for five days",
    "I cannot log in to the app after the update",
    "The fee was charged twice for one transfer",
    "The app crashes when I open the transaction history",
];

struct Cluster {
    scratch: TempDir,
    daemon: Daemon,
    clickhouse: PathBuf,
}

impl Cluster {
    /// Daemon plus a `clickhouse local` configured exactly the way the README
    /// tells an operator to configure a server: two config lines, nothing else.
    fn start(models: &[(&str, &str)]) -> Cluster {
        let clickhouse = match functional_tests::clickhouse_binary() {
            Some(path) => path,
            None => unreachable!("callers check with require_clickhouse!"),
        };
        let scratch = TempDir::new("clickhouse");
        let passports = scratch.child("models.d");
        std::fs::create_dir_all(&passports).unwrap();
        let root = functional_tests::models_dir().expect("models directory");

        for (name, kind) in models {
            let dir = root.join(name);
            let run = run_cli(
                scratch.path(),
                &[
                    "passport",
                    dir.to_str().unwrap(),
                    "--name",
                    name,
                    "--kind",
                    kind,
                    "--passports",
                    passports.to_str().unwrap(),
                ],
            );
            assert_eq!(run.code, Some(0), "passport for {name}: {}", run.stderr);
        }

        let daemon = Daemon::builder().models_dir(&passports).start();

        let out = scratch.child("clickhouse");
        let run = run_cli(
            scratch.path(),
            &[
                "gen-configs",
                "--client",
                bin("bridge-client").to_str().unwrap(),
                "--socket",
                daemon.socket().to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
            ],
        );
        assert_eq!(run.code, Some(0), "gen-configs: {}", run.stderr);

        std::fs::write(
            scratch.child("config.xml"),
            format!(
                "<clickhouse>\n\
                 \x20   <logger><level>warning</level><console>1</console></logger>\n\
                 \x20   <user_defined_executable_functions_config>{}/model_bridge_functions.xml</user_defined_executable_functions_config>\n\
                 \x20   <user_scripts_path>{}/scripts</user_scripts_path>\n\
                 \x20   <named_collections>\n\
                 \x20       <local_emb>\n\
                 \x20           <provider>openai</provider>\n\
                 \x20           <endpoint>http://127.0.0.1:{}/v1/embeddings</endpoint>\n\
                 \x20       </local_emb>\n\
                 \x20   </named_collections>\n\
                 </clickhouse>\n",
                out.display(),
                out.display(),
                daemon.port
            ),
        )
        .unwrap();

        Cluster {
            scratch,
            daemon,
            clickhouse,
        }
    }

    fn run(&self, sql: &str) -> std::process::Output {
        Command::new(&self.clickhouse)
            .arg("local")
            .arg("--config-file")
            .arg(self.scratch.child("config.xml"))
            .arg("--path")
            .arg(self.scratch.child("ch-data"))
            .arg("--query")
            .arg(sql)
            .output()
            .expect("run clickhouse local")
    }

    /// Runs SQL that must succeed and returns its output, trimmed.
    fn query(&self, sql: &str) -> String {
        let output = self.run(sql);
        assert!(
            output.status.success(),
            "query failed: {sql}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn query_fails(&self, sql: &str) -> String {
        let output = self.run(sql);
        assert!(
            !output.status.success(),
            "query was expected to fail: {sql}\nstdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        String::from_utf8_lossy(&output.stderr).to_string()
    }

    fn version(&self) -> u32 {
        self.query("SELECT splitByChar('.', version())[1]")
            .parse()
            .expect("major version")
    }
}

/// A `[1.0, 2.0]` literal ClickHouse can parse, from a vector the daemon
/// produced over HTTP.
fn array_literal(values: &[f32]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
    format!("[{}]", items.join(","))
}

fn http_vector(daemon: &Daemon, model: &str, text: &str) -> Vec<f32> {
    daemon
        .post_json("/v1/embeddings", json!({"model": model, "input": text}))
        .expect_status(200)
        .json()["data"][0]["embedding"]
        .as_array()
        .expect("embedding")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn the_generated_xml_registers_all_three_functions() {
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);

    let names = cluster.query(
        "SELECT name FROM system.functions \
         WHERE name IN ('localEmbed', 'localRerank', 'modelEvaluate') ORDER BY name",
    );
    assert_eq!(names, "localEmbed\nlocalRerank\nmodelEvaluate");
}

#[test]
fn local_embed_returns_the_daemon_vector() {
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);
    let expected = http_vector(&cluster.daemon, E5, "query: where did my transfer go");

    let answer = cluster.query(&format!(
        "SELECT length(localEmbed('{E5}', 'query: where did my transfer go')), \
                cosineDistance(localEmbed('{E5}', 'query: where did my transfer go'), {})",
        array_literal(&expected)
    ));
    let mut fields = answer.split('\t');
    assert_eq!(fields.next().unwrap(), "384");

    let distance: f64 = fields.next().unwrap().parse().expect("distance");
    assert!(
        distance.abs() < 1e-6,
        "SQL and HTTP produced different vectors (cosine distance {distance})"
    );
}

#[test]
fn local_rerank_orders_the_tickets() {
    require_clickhouse!();
    require_model!(RERANKER);
    let cluster = Cluster::start(&[(RERANKER, "rerank")]);

    let rows: Vec<String> = TICKETS
        .iter()
        .enumerate()
        .map(|(index, body)| format!("({}, '{}')", index + 1, body.replace('\'', "''")))
        .collect();
    let top = cluster.query(&format!(
        "CREATE TABLE tickets (id UInt32, body String) ENGINE = Memory; \
         INSERT INTO tickets VALUES {}; \
         SELECT id FROM tickets \
         ORDER BY localRerank('{RERANKER}', 'transfer has not reached the recipient', body) DESC \
         LIMIT 1",
        rows.join(",")
    ));

    assert_eq!(top, "1", "the direct answer must rank first in SQL too");
}

#[test]
fn model_evaluate_scores_a_table() {
    require_clickhouse!();
    require_model!(FRAUD);
    let cluster = Cluster::start(&[(FRAUD, "tabular")]);

    // The README's rows: the same amount at night and in the afternoon.
    let scores = cluster.query(&format!(
        "CREATE TABLE tx (id UInt32, amount Float64, hour Float64, is_new_device Float64, merchant_risk Float64) \
           ENGINE = Memory; \
         INSERT INTO tx VALUES (1, 4800, 2, 0, 0.1), (2, 4800, 14, 0, 0.1), (3, 9500, 3, 1, 0.9); \
         SELECT id, \
                modelEvaluate('{FRAUD}', [amount, hour, is_new_device, merchant_risk]::Array(Float32)) \
                  AS score \
         FROM tx ORDER BY score DESC"
    ));

    let order: Vec<&str> = scores
        .lines()
        .map(|line| line.split('\t').next().unwrap())
        .collect();
    assert_eq!(
        order,
        vec!["3", "1", "2"],
        "night and device risk must outrank the same amount in daylight:\n{scores}"
    );
}

#[test]
fn features_reach_the_daemon_only_once_they_are_float32() {
    // The UDF takes features at the model's own width, and ClickHouse will
    // not narrow a Float64 expression into it: its argument cast is an
    // accurate one, so the narrowing is the caller's to write.
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);

    let uncast = cluster.query_fails(&format!("SELECT modelEvaluate('{E5}', [0.1, 1.0])"));
    assert!(
        uncast.contains("CANNOT_CONVERT_TYPE"),
        "an uncast Float64 feature must stop at the cast:\n{uncast}"
    );

    // Cast, the same call reaches the daemon, which turns it down for being
    // an embedding model — proof that it arrived.
    let cast = cluster.query_fails(&format!(
        "SELECT modelEvaluate('{E5}', [0.1, 1.0]::Array(Float32))"
    ));
    assert!(
        cast.contains("is not a tabular model"),
        "the cast form must reach the daemon:\n{cast}"
    );
}

#[test]
fn a_mutation_can_backfill_a_column() {
    // The functions are declared deterministic precisely so this works.
    require_clickhouse!();
    require_model!(FRAUD);
    let cluster = Cluster::start(&[(FRAUD, "tabular")]);

    let filled = cluster.query(&format!(
        "CREATE TABLE tx (id UInt32, amount Float64, hour Float64, is_new_device Float64, \
           merchant_risk Float64, fraud_score Float32 DEFAULT 0) \
           ENGINE = MergeTree ORDER BY id; \
         INSERT INTO tx (id, amount, hour, is_new_device, merchant_risk) \
           SELECT number, 100 + number, number % 24, 0, 0.5 FROM numbers(50); \
         SET mutations_sync = 2; \
         ALTER TABLE tx UPDATE fraud_score = modelEvaluate( \
           '{FRAUD}', [amount, hour, is_new_device, merchant_risk]::Array(Float32)) WHERE 1; \
         SELECT count() FROM tx WHERE fraud_score != 0"
    ));

    assert_eq!(filled, "50", "the mutation did not score every row");
}

#[test]
fn a_whole_block_costs_one_daemon_round_trip() {
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);

    let before_requests = cluster.daemon.metric("model_bridge_embed_requests_total");
    let before_texts = cluster.daemon.metric("model_bridge_texts_embedded_total");

    // Summing the lengths keeps the column alive: ClickHouse prunes an
    // expression nothing consumes, and the UDF would never run.
    let dimensions = cluster.query(&format!(
        "SELECT sum(length(localEmbed('{E5}', concat('passage: row ', toString(number))))) \
         FROM numbers(200)"
    ));
    assert_eq!(dimensions, (200 * 384).to_string());

    let requests = cluster.daemon.metric("model_bridge_embed_requests_total") - before_requests;
    let texts = cluster.daemon.metric("model_bridge_texts_embedded_total") - before_texts;
    assert_eq!(texts, 200, "every row must reach the model exactly once");
    assert!(
        requests <= 4,
        "200 rows took {requests} daemon round trips; blocks are not being batched"
    );
}

#[test]
fn one_query_may_mix_models() {
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);

    let answer = cluster.query(&format!(
        "CREATE TABLE jobs (model String, text String) ENGINE = Memory; \
         INSERT INTO jobs VALUES ('{E5}', 'first'), ('stub', 'second'), ('{E5}', 'third'); \
         SELECT model, length(localEmbed(model, text)) FROM jobs ORDER BY text"
    ));

    assert_eq!(
        answer,
        format!("{E5}\t384\nstub\t384\n{E5}\t384"),
        "mixing models in one block changed the row order or the dimensions"
    );
}

#[test]
fn an_unknown_model_fails_the_query() {
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);

    let stderr = cluster.query_fails("SELECT localEmbed('ghost', 'text')");
    assert!(
        stderr.contains("unknown model `ghost`"),
        "the daemon's message must reach the user:\n{stderr}"
    );
}

#[test]
fn a_stopped_daemon_fails_the_query_instead_of_returning_rows() {
    require_clickhouse!();
    require_model!(E5);
    let mut cluster = Cluster::start(&[(E5, "embedding")]);
    cluster.daemon.stop();

    let stderr = cluster.query_fails(&format!("SELECT localEmbed('{E5}', 'text')"));
    assert!(
        stderr.contains("connecting to the daemon at"),
        "a dead daemon must fail the query loudly:\n{stderr}"
    );
}

#[test]
fn ai_embed_is_served_by_the_daemon_and_matches_local_embed() {
    require_clickhouse!();
    require_model!(E5);
    let cluster = Cluster::start(&[(E5, "embedding")]);
    if cluster.version() < 26 {
        functional_tests::skip(
            "clickhouse",
            "AI functions need ClickHouse 26.x; the UDF channel is covered separately",
        );
        return;
    }

    let identical = cluster.query(&format!(
        "SET allow_experimental_ai_functions = 1, \
             ai_function_embedding_default_credentials = 'local_emb'; \
         SELECT localEmbed('{E5}', 'passage: hello world') \
              = aiEmbed('passage: hello world', '{E5}')"
    ));

    assert_eq!(
        identical, "1",
        "the stock AI function and the UDF channel must return the same vector"
    );
}
