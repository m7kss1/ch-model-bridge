//! The admin CLI. `gen-configs` output is what ClickHouse parses at startup,
//! so its shape is a contract: a malformed functions file makes the server
//! retry it forever instead of starting.

use functional_tests::{bin, run_cli, TempDir};
use protocol::passport::{ModelKind, Passport};

fn model_dir(dir: &TempDir, name: &str) -> std::path::PathBuf {
    dir.model_dir(
        name,
        &[("model.onnx", b"graph bytes"), ("tokenizer.json", b"{}")],
    )
}

fn gen_configs(dir: &TempDir, extra: &[&str]) -> (String, std::path::PathBuf) {
    let out = dir.child("clickhouse");
    let socket = dir.child("bridge.sock");
    let mut args: Vec<String> = vec![
        "gen-configs".into(),
        "--client".into(),
        bin("bridge-client").to_string_lossy().into_owned(),
        "--socket".into(),
        socket.to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
    ];
    args.extend(extra.iter().map(|arg| arg.to_string()));
    let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();

    let run = run_cli(dir.path(), &borrowed);
    assert_eq!(run.code, Some(0), "gen-configs failed: {}", run.stderr);
    let xml = std::fs::read_to_string(out.join("model_bridge_functions.xml"))
        .expect("the functions file must be written");
    (xml, out)
}

#[test]
fn version_and_help_are_available() {
    let dir = TempDir::new("cli-help");
    for flag in ["--version", "--help"] {
        let run = run_cli(dir.path(), &[flag]);
        assert_eq!(
            run.code,
            Some(0),
            "model-bridge {flag} failed: {}",
            run.stderr
        );
        assert!(
            !run.stdout.is_empty(),
            "model-bridge {flag} printed nothing"
        );
    }
}

#[test]
fn a_tabular_passport_covers_the_graph_alone() {
    let dir = TempDir::new("cli-tabular");
    let model = model_dir(&dir, "scores");
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();

    let run = run_cli(
        dir.path(),
        &[
            "passport",
            model.to_str().unwrap(),
            "--name",
            "scores",
            "--kind",
            "tabular",
            "--passports",
            passports.to_str().unwrap(),
        ],
    );
    assert_eq!(run.code, Some(0), "{}", run.stderr);

    let passport = Passport::load(&passports.join("scores.toml")).unwrap();
    assert_eq!(passport.kind, ModelKind::Tabular);
    assert_eq!(
        passport.sha256.keys().collect::<Vec<_>>(),
        vec!["model.onnx"],
        "a tabular model has no tokenizer, so the passport must not claim one"
    );
    assert_eq!(passport.revision, 1, "the default revision");
    assert_eq!(passport.max_batch, 64, "the default batch size");
}

#[test]
fn a_passport_for_an_incomplete_directory_is_refused() {
    let dir = TempDir::new("cli-incomplete");
    let model = dir.model_dir("half", &[("model.onnx", b"graph bytes")]);
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();

    let run = run_cli(
        dir.path(),
        &[
            "passport",
            model.to_str().unwrap(),
            "--name",
            "half",
            "--kind",
            "embedding",
            "--passports",
            passports.to_str().unwrap(),
        ],
    );
    assert_ne!(run.code, Some(0), "a missing tokenizer must be an error");
    assert!(run.stderr.contains("tokenizer.json"), "{}", run.stderr);
    assert!(
        !passports.join("half.toml").exists(),
        "no passport may be written for a directory that cannot back it"
    );
}

#[test]
fn fetching_an_unknown_model_lists_the_catalog() {
    // Offline on purpose: the name is rejected before anything is downloaded.
    let dir = TempDir::new("cli-fetch");
    let run = run_cli(dir.path(), &["fetch", "gpt-5"]);

    assert_ne!(run.code, Some(0));
    assert!(
        run.stderr.contains("is not in the catalog"),
        "{}",
        run.stderr
    );
    assert!(run.stderr.contains("known models:"), "{}", run.stderr);
}

#[test]
fn gen_configs_declares_the_three_sql_functions() {
    let dir = TempDir::new("cli-gen");
    let (xml, _) = gen_configs(&dir, &[]);

    for (name, return_type) in [
        ("localEmbed", "Array(Float32)"),
        ("localRerank", "Float32"),
        ("modelEvaluate", "Float32"),
    ] {
        assert!(
            xml.contains(&format!("<name>{name}</name>")),
            "{name} is missing"
        );
        assert!(
            xml.contains(&format!("<return_type>{return_type}</return_type>")),
            "{name} must return {return_type}"
        );
    }

    assert_eq!(xml.matches("<type>executable_pool</type>").count(), 3);
    assert_eq!(xml.matches("<format>RowBinary</format>").count(), 3);
    assert_eq!(
        xml.matches("<send_chunk_header>1</send_chunk_header>")
            .count(),
        3,
        "the client reads blocks by row count, so the header is mandatory"
    );
    assert_eq!(
        xml.matches("<deterministic>1</deterministic>").count(),
        3,
        "without this, mutations on Replicated tables reject these functions"
    );

    // modelEvaluate takes Float64 because that is what ClickHouse float
    // expressions produce, and its accurate argument cast refuses to narrow
    // them; the client narrows instead.
    assert!(xml.contains("<argument><type>Array(Float64)</type><name>features</name></argument>"));
    assert!(xml.contains("<argument><type>String</type><name>document</name></argument>"));
}

#[test]
fn gen_configs_points_every_command_at_the_socket() {
    let dir = TempDir::new("cli-commands");
    let (xml, _) = gen_configs(&dir, &[]);
    let socket = dir.child("bridge.sock");

    for mode in ["embed", "rerank", "evaluate"] {
        assert!(
            xml.contains(&format!(
                "<command>bridge-client {mode} --socket {}</command>",
                socket.display()
            )),
            "the {mode} command does not point at the socket:\n{xml}"
        );
    }
    assert!(
        !xml.contains("<command>bridge-client embed --socket bridge.sock<"),
        "the socket path must be absolute: ClickHouse spawns the client from its own directory"
    );
}

#[test]
fn gen_configs_installs_a_runnable_client_in_the_scripts_directory() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new("cli-scripts");
    let (_, out) = gen_configs(&dir, &[]);
    let client = out.join("scripts/bridge-client");

    assert!(
        client.is_file(),
        "ClickHouse only spawns from user_scripts_path"
    );
    let mode = std::fs::metadata(&client).unwrap().permissions().mode();
    assert!(
        mode & 0o111 != 0,
        "the copied client is not executable: {mode:o}"
    );

    let run = std::process::Command::new(&client)
        .output()
        .expect("the copied client must run");
    assert!(String::from_utf8_lossy(&run.stderr).contains("usage: bridge-client"));
}

#[test]
fn gen_configs_bakes_in_the_tuning_flags() {
    let dir = TempDir::new("cli-tuning");
    let (xml, _) = gen_configs(
        &dir,
        &[
            "--pool-size",
            "4",
            "--command-read-timeout",
            "30000",
            "--max-command-execution-time",
            "45",
        ],
    );

    assert_eq!(xml.matches("<pool_size>4</pool_size>").count(), 3);
    assert_eq!(
        xml.matches("<command_read_timeout>30000</command_read_timeout>")
            .count(),
        3
    );
    assert_eq!(
        xml.matches("<max_command_execution_time>45</max_command_execution_time>")
            .count(),
        3
    );
}

#[test]
fn the_generated_xml_is_well_formed_and_comment_safe() {
    let dir = TempDir::new("cli-xml");
    let (xml, out) = gen_configs(&dir, &[]);

    // A comment containing `--` makes the file unparsable, and ClickHouse
    // answers an unparsable functions file by retrying it at startup forever.
    for comment in xml.split("<!--").skip(1) {
        let body = comment.split("-->").next().unwrap();
        assert!(
            !body.contains("--"),
            "an XML comment must not contain two consecutive dashes:\n{body}"
        );
    }

    let check = std::process::Command::new("python3")
        .arg("-c")
        .arg("import sys, xml.etree.ElementTree as ET; ET.parse(sys.argv[1])")
        .arg(out.join("model_bridge_functions.xml"))
        .output();
    match check {
        Ok(output) => assert!(
            output.status.success(),
            "the generated XML does not parse:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(_) => eprintln!("SKIP: no python3 to validate the XML with"),
    }
}

#[test]
fn regenerating_overwrites_instead_of_appending() {
    let dir = TempDir::new("cli-idempotent");
    let (first, _) = gen_configs(&dir, &[]);
    let (second, _) = gen_configs(&dir, &[]);
    assert_eq!(first, second, "gen-configs must be idempotent");

    let (retuned, _) = gen_configs(&dir, &["--pool-size", "2"]);
    assert_eq!(retuned.matches("<name>localEmbed</name>").count(), 1);
    assert!(retuned.contains("<pool_size>2</pool_size>"));
}

#[test]
fn the_default_out_dir_coexists_with_a_clickhouse_binary_in_cwd() {
    let dir = TempDir::new("cli-out-default");
    // The official installer drops the binary into the working directory as
    // `clickhouse` — exactly where gen-configs used to put its output.
    std::fs::write(dir.child("clickhouse"), b"the database binary").unwrap();

    let run = run_cli(
        dir.path(),
        &[
            "gen-configs",
            "--client",
            bin("bridge-client").to_str().unwrap(),
            "--socket",
            dir.child("bridge.sock").to_str().unwrap(),
        ],
    );
    assert_eq!(run.code, Some(0), "gen-configs failed: {}", run.stderr);
    assert!(
        dir.child("bridge-configs")
            .join("model_bridge_functions.xml")
            .is_file(),
        "the default output directory was not written"
    );
}

#[test]
fn a_file_in_the_way_of_out_is_reported_with_the_flag_to_move_it() {
    let dir = TempDir::new("cli-out-file");
    std::fs::write(dir.child("taken"), b"not a directory").unwrap();

    let run = run_cli(
        dir.path(),
        &[
            "gen-configs",
            "--client",
            bin("bridge-client").to_str().unwrap(),
            "--socket",
            dir.child("bridge.sock").to_str().unwrap(),
            "--out",
            dir.child("taken").to_str().unwrap(),
        ],
    );
    assert_ne!(run.code, Some(0), "a file in the way must fail the command");
    assert!(
        run.stderr.contains("a file is in the way") && run.stderr.contains("--out"),
        "the error must point at --out:\n{}",
        run.stderr
    );
}
