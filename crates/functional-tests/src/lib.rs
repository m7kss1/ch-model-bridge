//! Harness for the functional suite.
//!
//! Nothing here links the daemon. Tests spawn the built binaries and talk to
//! them exactly as ClickHouse does — RowBinary over a pipe into
//! `bridge-client`, frames over the unix socket, JSON over HTTP — so a green
//! run says the shipped artifacts behave, not that the code compiles.
//!
//! Binaries are looked up in `MODEL_BRIDGE_BIN_DIR` when set, which is how the
//! same suite is pointed at a release build or at binaries extracted from the
//! container image.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub use protocol::wire::{Request, Response};

/// How long a daemon may take to verify checksums and load its models before a
/// test gives up. Debug builds load a 118 MB encoder in seconds, not
/// milliseconds, and CI runners are slower still.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(240);

// ---------------------------------------------------------------- locations

/// Directory holding the binaries under test.
pub fn bin_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MODEL_BRIDGE_BIN_DIR") {
        return PathBuf::from(dir);
    }
    // .../target/<profile>/deps/<test executable>
    let mut path = std::env::current_exe().expect("current_exe");
    path.pop();
    if path.file_name().is_some_and(|name| name == "deps") {
        path.pop();
    }
    path
}

pub fn bin(name: &str) -> PathBuf {
    let path = bin_dir().join(name);
    assert!(
        path.is_file(),
        "`{name}` not found at {}.\n\
         Build the workspace first (cargo build --workspace), or point \
         MODEL_BRIDGE_BIN_DIR at a directory holding the binaries.",
        path.display()
    );
    path
}

/// Repository root, derived from this crate's manifest location.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repository root")
        .to_path_buf()
}

// ------------------------------------------------------------------ fixtures

/// Self-deleting scratch directory. Tests that need a model tree, a passport
/// or a socket build it here, so nothing leaks between runs.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "model-bridge-test-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create scratch directory");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn child(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Creates `name/` and fills it with the given files.
    pub fn model_dir(&self, name: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let dir = self.child(name);
        std::fs::create_dir_all(&dir).expect("create model directory");
        for (file, contents) in files {
            std::fs::write(dir.join(file), contents).expect("write model file");
        }
        dir
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Flips one byte of a file, the cheapest way to simulate a model that no
/// longer matches its passport.
pub fn corrupt_file(path: &Path) {
    let mut bytes = std::fs::read(path).expect("read file to corrupt");
    assert!(!bytes.is_empty(), "cannot corrupt an empty file");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(path, bytes).expect("write corrupted file");
}

// -------------------------------------------------------------------- daemon

#[derive(Clone)]
pub struct DaemonBuilder {
    args: Vec<String>,
    env: Vec<(String, String)>,
    models_dir: Option<PathBuf>,
    with_socket: bool,
    listen_port: Option<u16>,
}

impl Default for DaemonBuilder {
    fn default() -> Self {
        Self {
            args: Vec::new(),
            env: Vec::new(),
            models_dir: None,
            with_socket: true,
            listen_port: None,
        }
    }
}

impl DaemonBuilder {
    pub fn models_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn args<I: IntoIterator<Item = S>, S: Into<String>>(mut self, args: I) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env.push((key.to_string(), value.to_string()));
        self
    }

    pub fn without_socket(mut self) -> Self {
        self.with_socket = false;
        self
    }

    /// Binds this exact port instead of a fresh free one, for tests where the
    /// port is meant to collide.
    pub fn listen(mut self, port: u16) -> Self {
        self.listen_port = Some(port);
        self
    }

    fn command(&self, dir: &TempDir, port: u16, socket: &Path) -> Command {
        let mut command = Command::new(bin("bridged"));
        command
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}"))
            // Always explicit: the default is relative to the working
            // directory, which would make a test depend on where it ran.
            .arg("--models-dir")
            .arg(
                self.models_dir
                    .clone()
                    .unwrap_or_else(|| dir.child("no-passports-here")),
            );
        if self.with_socket {
            command.arg("--socket").arg(socket);
        }
        command.args(&self.args);
        command.env("RUST_LOG", "info");
        for (key, value) in &self.env {
            command.env(key, value);
        }
        command
    }

    /// Starts the daemon and waits until it answers on both channels.
    pub fn start(self) -> Daemon {
        let dir = TempDir::new("daemon");
        let port = self.listen_port.unwrap_or_else(free_port);
        let socket = dir.child("bridge.sock");
        let log = dir.child("bridged.log");
        let file = std::fs::File::create(&log).expect("create log file");

        let mut command = self.command(&dir, port, &socket);
        let child = command
            .stdout(Stdio::from(file.try_clone().expect("clone log handle")))
            .stderr(Stdio::from(file))
            .spawn()
            .expect("spawn bridged");

        let with_socket = self.with_socket;
        let mut daemon = Daemon {
            child,
            port,
            socket,
            log,
            dir,
            builder: self,
        };
        daemon.wait_ready(with_socket);
        daemon
    }

    /// Starts a daemon that is expected to refuse to serve, and returns what it
    /// logged. Used by the fail-close tests.
    pub fn start_expect_failure(self) -> String {
        let dir = TempDir::new("daemon-fail");
        let port = self.listen_port.unwrap_or_else(free_port);
        let socket = dir.child("bridge.sock");
        let log = dir.child("bridged.log");
        let file = std::fs::File::create(&log).expect("create log file");

        let mut command = self.command(&dir, port, &socket);
        let mut child = command
            .stdout(Stdio::from(file.try_clone().expect("clone log handle")))
            .stderr(Stdio::from(file))
            .spawn()
            .expect("spawn bridged");

        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            match child.try_wait().expect("wait for bridged") {
                Some(status) => {
                    let logs = std::fs::read_to_string(&log).unwrap_or_default();
                    assert!(
                        !status.success(),
                        "daemon was expected to refuse to start, but exited successfully.\n{logs}"
                    );
                    return logs;
                }
                None if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let logs = std::fs::read_to_string(&log).unwrap_or_default();
                    panic!(
                        "daemon was expected to refuse to start, but it is still running.\n{logs}"
                    );
                }
                None => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }
}

pub struct Daemon {
    child: Child,
    pub port: u16,
    pub socket: PathBuf,
    log: PathBuf,
    dir: TempDir,
    builder: DaemonBuilder,
}

impl Daemon {
    pub fn builder() -> DaemonBuilder {
        DaemonBuilder::default()
    }

    /// A daemon serving only the built-in `stub` embedder: enough to exercise
    /// every channel, every error path and the whole dispatcher without a
    /// single model file on disk.
    pub fn stub() -> Daemon {
        DaemonBuilder::default().start()
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub fn logs(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    pub fn get(&self, path: &str) -> HttpResponse {
        http(self.port, "GET", path, None)
    }

    pub fn post_json(&self, path: &str, body: serde_json::Value) -> HttpResponse {
        let payload = serde_json::to_vec(&body).expect("serialize request");
        http(
            self.port,
            "POST",
            path,
            Some(("application/json", &payload)),
        )
    }

    pub fn post_raw(&self, path: &str, content_type: &str, body: &[u8]) -> HttpResponse {
        http(self.port, "POST", path, Some((content_type, body)))
    }

    /// Prometheus counter value, or 0 when the counter has never moved.
    pub fn metric(&self, name: &str) -> u64 {
        let body = self.get("/metrics").expect_status(200).body;
        for line in body.lines() {
            if let Some(value) = line.strip_prefix(&format!("{name} ")) {
                return value.trim().parse().expect("counter value");
            }
        }
        panic!("counter `{name}` is missing from /metrics:\n{body}");
    }

    pub fn connect(&self) -> Socket {
        Socket::connect(&self.socket)
    }

    /// One request over the unix socket on a fresh connection.
    pub fn call(&self, request: &Request) -> Response {
        self.connect().call(request)
    }

    /// Runs `bridge-client` against this daemon with `stdin` on its pipe.
    pub fn run_client(&self, mode: &str, stdin: &[u8]) -> ClientRun {
        run_bridge_client(mode, &self.socket, stdin)
    }

    /// Stops the daemon so a test can watch what breaks without it.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Kills the daemon and brings a fresh one up on the same socket path,
    /// the way an operator restarts it to pick up changed models. The HTTP
    /// port is allocated anew unless the builder pinned one; the socket path
    /// is the part of the contract that survives a restart.
    pub fn restart(&mut self) {
        self.stop();
        self.port = self.builder.listen_port.unwrap_or_else(free_port);
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.log)
            .expect("reopen log file");
        let mut command = self.builder.command(&self.dir, self.port, &self.socket);
        self.child = command
            .stdout(Stdio::from(file.try_clone().expect("clone log handle")))
            .stderr(Stdio::from(file))
            .spawn()
            .expect("respawn bridged");
        self.wait_ready(self.builder.with_socket);
    }

    fn wait_ready(&mut self, with_socket: bool) {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait().expect("wait for bridged") {
                panic!(
                    "bridged exited with {status} during startup:\n{}",
                    self.logs()
                );
            }
            let http_ready = TcpStream::connect(("127.0.0.1", self.port)).is_ok();
            let socket_ready = !with_socket || self.socket.exists();
            if http_ready && socket_ready {
                // The listener is up; make sure it answers before returning.
                if self.get("/health").status == 200 {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "bridged did not become ready within {STARTUP_TIMEOUT:?}:\n{}",
                self.logs()
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

// ---------------------------------------------------------------------- http

pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    pub fn expect_status(self, expected: u16) -> Self {
        assert_eq!(
            self.status, expected,
            "unexpected status; body was:\n{}",
            self.body
        );
        self
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("response is not JSON ({e}):\n{}", self.body))
    }

    /// `error.message` from the OpenAI-compatible error envelope.
    pub fn error_message(&self) -> String {
        self.json()["error"]["message"]
            .as_str()
            .unwrap_or_else(|| panic!("no error.message in:\n{}", self.body))
            .to_string()
    }
}

/// Minimal HTTP/1.1 client. Deliberately not a crate: the tests are the
/// outside world, and `Connection: close` plus read-to-EOF is all the outside
/// world needs to be.
fn http(port: u16, method: &str, path: &str, body: Option<(&str, &[u8])>) -> HttpResponse {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to the daemon");
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .expect("set read timeout");

    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n");
    if let Some((content_type, payload)) = body {
        head.push_str(&format!(
            "Content-Type: {content_type}\r\nContent-Length: {}\r\n",
            payload.len()
        ));
    }
    head.push_str("\r\n");

    stream.write_all(head.as_bytes()).expect("write request");
    if let Some((_, payload)) = body {
        stream.write_all(payload).expect("write body");
    }
    stream.flush().expect("flush request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");

    let split = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("response without a header terminator");
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let body = String::from_utf8_lossy(&raw[split + 4..]).to_string();
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("malformed status line: {head}"));

    HttpResponse { status, body }
}

// ------------------------------------------------------------- binary channel

/// One connection to the daemon's unix socket, kept open so tests can send
/// several frames the way a pooled `bridge-client` process does.
pub struct Socket {
    stream: UnixStream,
}

impl Socket {
    pub fn connect(path: &Path) -> Self {
        let stream = UnixStream::connect(path)
            .unwrap_or_else(|e| panic!("connect to {}: {e}", path.display()));
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .expect("set read timeout");
        Self { stream }
    }

    pub fn send_frame(&mut self, payload: &[u8]) {
        self.stream
            .write_all(&(payload.len() as u32).to_le_bytes())
            .expect("write frame length");
        self.stream.write_all(payload).expect("write frame payload");
        self.stream.flush().expect("flush frame");
    }

    /// Sends a length prefix that does not match the payload, for the
    /// protocol's own error paths.
    pub fn send_raw(&mut self, length: u32, payload: &[u8]) {
        self.stream
            .write_all(&length.to_le_bytes())
            .expect("write frame length");
        self.stream.write_all(payload).expect("write frame payload");
        self.stream.flush().expect("flush frame");
    }

    /// Half-closes the connection, so a frame the daemon is still waiting for
    /// arrives as EOF instead of hanging the test.
    pub fn shutdown_write(&mut self) {
        self.stream
            .shutdown(std::net::Shutdown::Write)
            .expect("shutdown the write half");
    }

    pub fn read_frame(&mut self) -> std::io::Result<Vec<u8>> {
        let mut length = [0u8; 4];
        self.stream.read_exact(&mut length)?;
        let mut payload = vec![0u8; u32::from_le_bytes(length) as usize];
        self.stream.read_exact(&mut payload)?;
        Ok(payload)
    }

    pub fn call(&mut self, request: &Request) -> Response {
        self.send_frame(&protocol::wire::encode_request(request));
        self.read_response()
    }

    pub fn call_raw(&mut self, payload: &[u8]) -> Response {
        self.send_frame(payload);
        self.read_response()
    }

    fn read_response(&mut self) -> Response {
        let frame = self.read_frame().expect("read response frame");
        protocol::wire::decode_response(&frame).expect("decode response frame")
    }
}

/// Convenience for the common case of one request on a fresh connection.
pub fn wire_call(socket: &Path, request: &Request) -> Response {
    Socket::connect(socket).call(request)
}

/// The error text of a `Response::Error`, or a panic naming what came instead.
pub fn error_text(response: &Response) -> &str {
    match response {
        Response::Error(message) => message,
        other => panic!("expected an error response, got {other:?}"),
    }
}

// ---------------------------------------------------- the ClickHouse UDF pipe

/// RowBinary encoding, the format ClickHouse writes into an executable UDF's
/// pipe and reads back from it.
pub mod rowbinary {
    /// `String`: varuint length, then the bytes.
    pub fn string(value: &str) -> Vec<u8> {
        let mut out = varuint(value.len() as u64);
        out.extend_from_slice(value.as_bytes());
        out
    }

    /// `Array(Float32)`: varuint element count, then little-endian floats.
    pub fn f32_array(values: &[f32]) -> Vec<u8> {
        let mut out = varuint(values.len() as u64);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    pub fn varuint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    /// A block as ClickHouse sends it with `send_chunk_header = 1`: the row
    /// count on its own line, then the rows.
    pub fn block(rows: &[Vec<u8>]) -> Vec<u8> {
        let mut out = format!("{}\n", rows.len()).into_bytes();
        for row in rows {
            out.extend_from_slice(row);
        }
        out
    }

    /// Decodes `rows` values of `Array(Float32)`.
    pub fn read_f32_arrays(bytes: &[u8], rows: usize) -> Vec<Vec<f32>> {
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let (len, used) = read_varuint(&bytes[cursor..])
                .unwrap_or_else(|| panic!("row {row}: truncated array length"));
            cursor += used;
            let end = cursor + len as usize * 4;
            assert!(
                end <= bytes.len(),
                "row {row}: array of {len} floats runs past the end of the output"
            );
            out.push(
                bytes[cursor..end]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect(),
            );
            cursor = end;
        }
        assert_eq!(
            cursor,
            bytes.len(),
            "{} trailing bytes",
            bytes.len() - cursor
        );
        out
    }

    /// Decodes `rows` values of `Float32`.
    pub fn read_f32_scalars(bytes: &[u8], rows: usize) -> Vec<f32> {
        assert_eq!(
            bytes.len(),
            rows * 4,
            "expected {rows} float32 values, got {} bytes",
            bytes.len()
        );
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    fn read_varuint(bytes: &[u8]) -> Option<(u64, usize)> {
        let mut value = 0u64;
        let mut shift = 0;
        for (index, byte) in bytes.iter().enumerate() {
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some((value, index + 1));
            }
            shift += 7;
        }
        None
    }
}

pub struct ClientRun {
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: String,
}

impl ClientRun {
    pub fn expect_success(self) -> Vec<u8> {
        assert_eq!(
            self.code,
            Some(0),
            "bridge-client failed; stderr was:\n{}",
            self.stderr
        );
        self.stdout
    }

    pub fn expect_failure(self) -> String {
        assert_ne!(
            self.code,
            Some(0),
            "bridge-client was expected to fail but exited 0; stdout was {} bytes",
            self.stdout.len()
        );
        self.stderr
    }
}

pub fn run_bridge_client(mode: &str, socket: &Path, stdin: &[u8]) -> ClientRun {
    run_with_stdin(
        Command::new(bin("bridge-client"))
            .arg(mode)
            .arg("--socket")
            .arg(socket),
        stdin,
    )
}

/// Runs the admin CLI. Every subcommand resolves its default paths against the
/// working directory, so tests pass one explicitly.
pub fn run_cli(cwd: &Path, args: &[&str]) -> ClientRun {
    run_with_stdin(
        Command::new(bin("model-bridge"))
            .current_dir(cwd)
            .args(args),
        b"",
    )
}

fn run_with_stdin(command: &mut Command, stdin: &[u8]) -> ClientRun {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn process");
    if let Err(e) = child.stdin.take().expect("stdin pipe").write_all(stdin) {
        assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe, "write stdin: {e}");
    }
    let output = child.wait_with_output().expect("wait for process");
    ClientRun {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    }
}

// ----------------------------------------------------------------- test tiers

/// Directory holding real model directories, when the model-backed tier can
/// run. Defaults to `models/` in the repository, which is where
/// `model-bridge fetch` puts them.
pub fn models_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("MODEL_BRIDGE_MODELS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => repo_root().join("models"),
    };
    dir.is_dir().then_some(dir)
}

/// Reference scores written by `examples/train_fraud_model.py`, the oracle for
/// the tabular regression tests.
pub fn fraud_reference() -> Option<PathBuf> {
    let path = match std::env::var_os("MODEL_BRIDGE_FRAUD_REFERENCE") {
        Some(path) => PathBuf::from(path),
        None => repo_root().join("tmp/fraud-expected.json"),
    };
    path.is_file().then_some(path)
}

/// The `clickhouse` binary for the end-to-end tier.
pub fn clickhouse_binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("MODEL_BRIDGE_CLICKHOUSE") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let output = Command::new("sh")
        .arg("-c")
        .arg("command -v clickhouse")
        .output()
        .ok()?;
    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    (output.status.success() && path.is_file()).then_some(path)
}

/// Fails instead of skipping when the tier is listed in
/// `MODEL_BRIDGE_REQUIRE_TIERS`. CI sets it, so a missing model or a missing
/// ClickHouse binary can never turn into a silently green run.
pub fn skip(tier: &str, reason: &str) {
    let required = std::env::var("MODEL_BRIDGE_REQUIRE_TIERS").unwrap_or_default();
    assert!(
        !required.split(',').any(|name| name.trim() == tier),
        "tier `{tier}` is required by MODEL_BRIDGE_REQUIRE_TIERS but cannot run: {reason}"
    );
    eprintln!("SKIP [{tier}]: {reason}");
}

/// Returns the models directory, or skips the calling test.
#[macro_export]
macro_rules! require_models {
    () => {
        match $crate::models_dir() {
            Some(dir) => dir,
            None => {
                $crate::skip("models", "no models directory; run `model-bridge fetch`");
                return;
            }
        }
    };
}

/// Returns the path to a specific model directory, or skips the calling test.
#[macro_export]
macro_rules! require_model {
    ($name:expr) => {{
        let dir = $crate::require_models!().join($name);
        if !dir.join("model.onnx").is_file() {
            $crate::skip("models", &format!("{} is not downloaded", dir.display()));
            return;
        }
        dir
    }};
}

/// Returns the `clickhouse` binary, or skips the calling test.
#[macro_export]
macro_rules! require_clickhouse {
    () => {
        match $crate::clickhouse_binary() {
            Some(path) => path,
            None => {
                $crate::skip(
                    "clickhouse",
                    "no `clickhouse` binary on PATH or in MODEL_BRIDGE_CLICKHOUSE",
                );
                return;
            }
        }
    };
}
