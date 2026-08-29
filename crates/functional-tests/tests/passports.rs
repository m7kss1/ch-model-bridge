//! Fail-close behaviour. A passport is the only way a model enters the
//! daemon, and the promise is that a file which does not match its checksum
//! stops the daemon instead of quietly serving different numbers.

use functional_tests::{corrupt_file, run_cli, Daemon, TempDir};
use protocol::passport::{ModelKind, Passport};

/// A model directory whose files are not real ONNX. Checksums are verified
/// before anything is parsed, so these are enough for the fail-close paths.
fn fake_model(dir: &TempDir, name: &str) -> std::path::PathBuf {
    dir.model_dir(
        name,
        &[
            ("model.onnx", b"not really an onnx graph"),
            ("tokenizer.json", b"{\"not\": \"a tokenizer\"}"),
        ],
    )
}

fn issue_passport(dir: &TempDir, model: &std::path::Path, name: &str, extra: &[&str]) -> String {
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();
    let mut args = vec![
        "passport",
        model.to_str().unwrap(),
        "--name",
        name,
        "--kind",
        "embedding",
        "--passports",
        passports.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    let run = run_cli(dir.path(), &args);
    assert_eq!(
        run.code,
        Some(0),
        "model-bridge passport failed: {}",
        run.stderr
    );
    passports.to_str().unwrap().to_string()
}

#[test]
fn a_passport_records_every_file_the_runtime_will_load() {
    let dir = TempDir::new("passport-contents");
    let model = fake_model(&dir, "demo");
    let passports = issue_passport(
        &dir,
        &model,
        "demo",
        &["--revision", "7", "--max-batch", "12"],
    );

    let path = std::path::Path::new(&passports).join("demo.toml");
    let passport = Passport::load(&path).expect("issued passport must parse");

    assert_eq!(passport.name, "demo");
    assert_eq!(passport.kind, ModelKind::Embedding);
    assert_eq!(passport.revision, 7);
    assert_eq!(passport.max_batch, 12);
    assert_eq!(passport.dir, model);
    assert_eq!(
        passport.sha256.keys().collect::<Vec<_>>(),
        vec!["model.onnx", "tokenizer.json"],
        "a text model is inseparable from its tokenizer"
    );
    passport
        .verify(&model)
        .expect("a freshly issued passport must verify");
}

#[test]
fn a_tampered_model_file_stops_the_daemon() {
    let dir = TempDir::new("tampered");
    let model = fake_model(&dir, "demo");
    let passports = issue_passport(&dir, &model, "demo", &[]);

    corrupt_file(&model.join("model.onnx"));

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(
        logs.contains("checksum mismatch"),
        "the daemon must say what did not match:\n{logs}"
    );
    assert!(
        logs.contains("differs from the one this passport was issued for"),
        "{logs}"
    );
}

#[test]
fn a_file_missing_from_the_model_directory_stops_the_daemon() {
    let dir = TempDir::new("missing-file");
    let model = fake_model(&dir, "demo");
    let passports = issue_passport(&dir, &model, "demo", &[]);

    std::fs::remove_file(model.join("tokenizer.json")).unwrap();

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("tokenizer.json"), "{logs}");
}

#[test]
fn a_passport_that_verifies_nothing_is_refused() {
    let dir = TempDir::new("empty-passport");
    let model = fake_model(&dir, "demo");
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();
    std::fs::write(
        passports.join("demo.toml"),
        format!(
            "name = \"demo\"\nkind = \"embedding\"\ndir = \"{}\"\nrevision = 1\n\n[sha256]\n",
            model.display()
        ),
    )
    .unwrap();

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("lists no files to verify"), "{logs}");
}

#[test]
fn an_unparsable_passport_is_refused() {
    let dir = TempDir::new("bad-toml");
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();
    std::fs::write(passports.join("demo.toml"), "kind = \"telepathy\"\n").unwrap();

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("invalid passport"), "{logs}");
}

#[test]
fn an_empty_passports_directory_stops_the_daemon() {
    let dir = TempDir::new("no-passports");
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("no passports found"), "{logs}");
}

#[test]
fn an_absent_passports_directory_leaves_the_stub_serving() {
    // The difference matters: "no models configured yet" is a normal state,
    // "a models directory with nothing in it" is a misconfiguration.
    let daemon = Daemon::stub();
    assert!(
        daemon.logs().contains("no passports directory, skipping"),
        "{}",
        daemon.logs()
    );
    assert_eq!(daemon.get("/v1/models").json()["data"][0]["name"], "stub");
}

#[test]
fn a_file_that_matches_its_checksum_but_is_not_a_model_stops_the_daemon() {
    // Fail-close is about serving nothing rather than serving nonsense: a
    // passport that verifies does not make a corrupt graph loadable.
    let dir = TempDir::new("not-a-model");
    let model = fake_model(&dir, "demo");
    let passports = issue_passport(&dir, &model, "demo", &[]);

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(logs.contains("loading `demo`"), "{logs}");
    assert!(
        !logs.contains("checksum mismatch"),
        "the checksums were fine; the graph was not:\n{logs}"
    );
}

#[test]
fn a_relative_directory_resolves_against_the_passport_file() {
    // Passport trees are meant to be movable, so `dir` may be relative.
    let dir = TempDir::new("relative-dir");
    let passports = dir.child("models.d");
    std::fs::create_dir_all(&passports).unwrap();
    let model = dir.model_dir(
        "models.d/demo",
        &[("model.onnx", b"graph"), ("tokenizer.json", b"{}")],
    );

    let sums = |file: &str| {
        protocol::passport::sha256_file(&model.join(file)).expect("checksum the fixture")
    };
    std::fs::write(
        passports.join("demo.toml"),
        format!(
            "name = \"demo\"\nkind = \"embedding\"\ndir = \"demo\"\nrevision = 1\n\n\
             [sha256]\n\"model.onnx\" = \"{}\"\n\"tokenizer.json\" = \"{}\"\n",
            sums("model.onnx"),
            sums("tokenizer.json")
        ),
    )
    .unwrap();

    let logs = Daemon::builder()
        .models_dir(&passports)
        .start_expect_failure();
    assert!(
        logs.contains("loading `demo`") && !logs.contains("checksum mismatch"),
        "the relative directory did not resolve against the passport:\n{logs}"
    );
}

#[test]
fn only_toml_files_are_treated_as_passports() {
    // Operators drop notes and backups next to passports; anything that is not
    // a `.toml` must be ignored rather than parsed.
    let dir = TempDir::new("only-toml");
    let model = fake_model(&dir, "demo");
    let passports = issue_passport(&dir, &model, "demo", &[]);
    let passports = std::path::Path::new(&passports);
    std::fs::write(passports.join("notes.txt"), "not a passport at all").unwrap();
    std::fs::write(passports.join("demo.toml.bak"), "garbage {{{").unwrap();

    let logs = Daemon::builder()
        .models_dir(passports)
        .start_expect_failure();
    assert!(
        logs.contains("loading `demo`"),
        "the daemon stopped on something that is not a passport:\n{logs}"
    );
}
