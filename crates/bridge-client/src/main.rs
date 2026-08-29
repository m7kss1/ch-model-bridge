//! Thin adapter between the ClickHouse executable-UDF pipe protocol and the
//! daemon socket: `RowBinary` rows in on stdin, results out on stdout. Models
//! stay in the daemon, so pooled UDF processes do not multiply them in memory.
//!
//! ClickHouse is configured with `send_chunk_header = 1`: each block arrives
//! as a decimal row count terminated by `\n`, then that many rows. The reply
//! must contain exactly as many result rows, flushed before the next header
//! is read.
//!
//! The daemon connection is opened on first use and replaced whenever it
//! breaks. A daemon restart is the normal way to update models, and it must
//! not cost ClickHouse its pool of warmed-up UDF processes: a request caught
//! mid-flight is sent again over a fresh connection, which is safe because
//! inference has no side effects to repeat.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use protocol::wire::{self, Request, Response};

enum Mode {
    /// Rows of (model, text) -> one `Array(Float32)` per row.
    Embed,
    /// Rows of (model, query, document) -> one `Float32` per row.
    Rerank,
    /// Rows of (model, features `Array(Float64)`) -> one `Float32` per row.
    /// Features arrive as `Float64` because that is what ClickHouse float
    /// expressions produce; they are narrowed to the model's float32 here.
    Evaluate,
}

fn main() {
    if let Err(e) = run() {
        // stderr ends up in the ClickHouse log; a non-zero exit makes the
        // query fail instead of silently producing partial results.
        eprintln!("bridge-client: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let (mode, socket) = parse_args()?;
    // Not connected here: a daemon that is down while ClickHouse spawns the
    // pool must stall the first request, not kill the process on arrival.
    let mut daemon = Daemon::new(socket);

    let stdin = std::io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut output = BufWriter::new(stdout.lock());

    while let Some(rows) = read_chunk_header(&mut input)? {
        match mode {
            Mode::Embed => {
                let mut models = Vec::with_capacity(rows);
                let mut texts = Vec::with_capacity(rows);
                for _ in 0..rows {
                    models.push(read_string(&mut input)?);
                    texts.push(read_string(&mut input)?);
                }
                let results = embed_rows(&mut daemon, &models, texts)?;
                for vector in results {
                    write_varuint(&mut output, vector.len() as u64)?;
                    for value in vector {
                        output.write_all(&value.to_le_bytes())?;
                    }
                }
            }
            Mode::Rerank => {
                let mut models = Vec::with_capacity(rows);
                let mut pairs = Vec::with_capacity(rows);
                for _ in 0..rows {
                    models.push(read_string(&mut input)?);
                    let query = read_string(&mut input)?;
                    let document = read_string(&mut input)?;
                    pairs.push((query, document));
                }
                let scores = rerank_rows(&mut daemon, &models, pairs)?;
                for score in scores {
                    output.write_all(&score.to_le_bytes())?;
                }
            }
            Mode::Evaluate => {
                let mut models = Vec::with_capacity(rows);
                let mut features = Vec::with_capacity(rows);
                for _ in 0..rows {
                    models.push(read_string(&mut input)?);
                    features.push(read_f32_array(&mut input)?);
                }
                let scores = evaluate_rows(&mut daemon, &models, features)?;
                for score in scores {
                    output.write_all(&score.to_le_bytes())?;
                }
            }
        }
        output.flush()?;
    }
    Ok(())
}

/// The model name is a per-row argument in SQL, but it is almost always a
/// constant within a block; rows are grouped so each distinct model costs one
/// daemon round-trip.
fn group_by_model(models: &[String]) -> Vec<(String, Vec<usize>)> {
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, model) in models.iter().enumerate() {
        if !groups.contains_key(model.as_str()) {
            order.push(model.clone());
        }
        groups.entry(model).or_default().push(index);
    }
    order
        .into_iter()
        .map(|model| {
            let indices = groups.remove(model.as_str()).unwrap_or_default();
            (model, indices)
        })
        .collect()
}

fn embed_rows(
    daemon: &mut Daemon,
    models: &[String],
    texts: Vec<String>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut results: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
    for (model, indices) in group_by_model(models) {
        let group_texts: Vec<String> = indices.iter().map(|&i| texts[i].clone()).collect();
        let response = daemon.round_trip(&Request::Embed {
            model: model.clone(),
            texts: group_texts,
        })?;
        let Response::Embed { dim, vectors } = response else {
            bail!("daemon: {}", response_error(response));
        };
        let dim = dim as usize;
        if dim == 0 || vectors.len() != indices.len() * dim {
            bail!(
                "daemon returned {} floats for {} texts of dimension {dim}",
                vectors.len(),
                indices.len()
            );
        }
        for (slot, chunk) in indices.iter().zip(vectors.chunks_exact(dim)) {
            results[*slot] = chunk.to_vec();
        }
    }
    Ok(results)
}

fn rerank_rows(
    daemon: &mut Daemon,
    models: &[String],
    pairs: Vec<(String, String)>,
) -> anyhow::Result<Vec<f32>> {
    let mut results = vec![0f32; pairs.len()];
    for (model, indices) in group_by_model(models) {
        let group_pairs: Vec<(String, String)> =
            indices.iter().map(|&i| pairs[i].clone()).collect();
        let response = daemon.round_trip(&Request::Rerank {
            model: model.clone(),
            pairs: group_pairs,
        })?;
        let Response::Rerank { scores } = response else {
            bail!("daemon: {}", response_error(response));
        };
        if scores.len() != indices.len() {
            bail!(
                "daemon returned {} scores for {} pairs",
                scores.len(),
                indices.len()
            );
        }
        for (slot, score) in indices.iter().zip(scores) {
            results[*slot] = score;
        }
    }
    Ok(results)
}

fn evaluate_rows(
    daemon: &mut Daemon,
    models: &[String],
    features: Vec<Vec<f32>>,
) -> anyhow::Result<Vec<f32>> {
    let mut results = vec![0f32; features.len()];
    for (model, indices) in group_by_model(models) {
        // The daemon takes a rectangular matrix per request; ragged feature
        // arrays within one model are a caller bug worth naming precisely.
        let n_features = features[indices[0]].len();
        let mut values = Vec::with_capacity(indices.len() * n_features);
        for &index in &indices {
            if features[index].len() != n_features {
                bail!(
                    "row {index} has {} features while an earlier `{model}` row has {n_features}",
                    features[index].len()
                );
            }
            values.extend_from_slice(&features[index]);
        }
        let response = daemon.round_trip(&Request::Evaluate {
            model: model.clone(),
            n_features: n_features as u32,
            values,
        })?;
        let Response::Evaluate { scores } = response else {
            bail!("daemon: {}", response_error(response));
        };
        if scores.len() != indices.len() {
            bail!(
                "daemon returned {} scores for {} rows",
                scores.len(),
                indices.len()
            );
        }
        for (slot, score) in indices.iter().zip(scores) {
            results[*slot] = score;
        }
    }
    Ok(results)
}

fn response_error(response: Response) -> String {
    match response {
        Response::Error(message) => message,
        _ => "unexpected response kind".to_string(),
    }
}

/// How long a request keeps being retried once the daemon stops answering,
/// counted from the first failure. Long enough to ride out a restart that
/// serves prepared models, short enough that ClickHouse's own UDF timeouts,
/// ten seconds by default, stay the ones an operator reasons about.
const RETRY_WINDOW: Duration = Duration::from_secs(10);

/// The daemon connection, opened lazily and replaced after any transport
/// failure.
struct Daemon {
    socket: PathBuf,
    stream: Option<UnixStream>,
}

impl Daemon {
    fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            stream: None,
        }
    }

    /// One request frame out, one response frame back. When the transport
    /// fails — the daemon restarting is the expected reason — the request is
    /// resent on a fresh connection until `RETRY_WINDOW` past the first
    /// failure, so a restart briefly stalls queries instead of failing them
    /// and never leaves the pooled process wedged.
    fn round_trip(&mut self, request: &Request) -> anyhow::Result<Response> {
        let payload = wire::encode_request(request);
        let mut give_up: Option<Instant> = None;
        let mut delay = Duration::from_millis(50);
        loop {
            let error = match self.attempt(&payload) {
                Ok(frame) => {
                    return wire::decode_response(&frame)
                        .map_err(|e| anyhow!("bad response frame: {e}"));
                }
                Err(e) if !worth_retrying(&e) => {
                    return Err(e).with_context(|| {
                        format!("talking to the daemon at {}", self.socket.display())
                    });
                }
                Err(e) => e,
            };
            // Never reuse a stream that failed mid-frame: a late reply on it
            // could be mistaken for the answer to the resent request.
            self.stream = None;
            let give_up_at = *give_up.get_or_insert_with(|| Instant::now() + RETRY_WINDOW);
            let now = Instant::now();
            if now >= give_up_at {
                return Err(error).with_context(|| {
                    format!(
                        "daemon at {} still unreachable after retrying for {RETRY_WINDOW:?}",
                        self.socket.display()
                    )
                });
            }
            std::thread::sleep(delay.min(give_up_at - now));
            delay = (delay * 2).min(Duration::from_secs(1));
        }
    }

    fn attempt(&mut self, payload: &[u8]) -> io::Result<Vec<u8>> {
        if self.stream.is_none() {
            self.stream = Some(UnixStream::connect(&self.socket)?);
        }
        let stream = self.stream.as_mut().expect("connected above");
        stream.write_all(&(payload.len() as u32).to_le_bytes())?;
        stream.write_all(payload)?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf);
        if len > wire::MAX_FRAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("daemon sent a frame of {len} bytes, over the limit"),
            ));
        }
        let mut frame = vec![0u8; len as usize];
        stream.read_exact(&mut frame)?;
        Ok(frame)
    }
}

/// Transport failures that mean the daemon is gone or restarting: the socket
/// file missing or stale while the daemon rebinds, the connection refused,
/// reset or closed under a request. Anything else — a permission error, an
/// oversized frame — is a fault that retrying would only repeat.
fn worth_retrying(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound
            | io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
            | io::ErrorKind::UnexpectedEof
    )
}

fn parse_args() -> anyhow::Result<(Mode, PathBuf)> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, flag, socket] if flag == "--socket" => {
            let mode = match mode.as_str() {
                "embed" => Mode::Embed,
                "rerank" => Mode::Rerank,
                "evaluate" => Mode::Evaluate,
                other => bail!("unknown mode `{other}`; expected `embed`, `rerank` or `evaluate`"),
            };
            Ok((mode, PathBuf::from(socket)))
        }
        _ => bail!("usage: bridge-client <embed|rerank|evaluate> --socket PATH"),
    }
}

fn read_chunk_header(input: &mut impl BufRead) -> anyhow::Result<Option<usize>> {
    let mut line = String::new();
    let n = input.read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    let trimmed = line.trim_end_matches('\n');
    trimmed
        .parse::<usize>()
        .map(Some)
        .map_err(|_| anyhow!("bad chunk header {trimmed:?}"))
}

/// `RowBinary` strings are LEB128-length-prefixed raw bytes.
fn read_varuint(input: &mut impl Read) -> anyhow::Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let mut byte = [0u8; 1];
        input.read_exact(&mut byte)?;
        value |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("varint longer than 64 bits")
}

fn write_varuint(output: &mut impl Write, mut value: u64) -> anyhow::Result<()> {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            output.write_all(&[byte])?;
            return Ok(());
        }
        output.write_all(&[byte | 0x80])?;
    }
}

/// A `RowBinary` `Array(Float64)` is a LEB128 element count followed by the
/// elements as `f64` little-endian; values are narrowed to `f32` on read.
fn read_f32_array(input: &mut impl Read) -> anyhow::Result<Vec<f32>> {
    let len = read_varuint(input)? as usize;
    let mut values = Vec::with_capacity(len.min(1 << 20));
    for _ in 0..len {
        let mut bytes = [0u8; 8];
        input.read_exact(&mut bytes)?;
        values.push(f64::from_le_bytes(bytes) as f32);
    }
    Ok(values)
}

fn read_string(input: &mut impl Read) -> anyhow::Result<String> {
    let len = read_varuint(input)? as usize;
    let mut bytes = vec![0u8; len];
    input.read_exact(&mut bytes)?;
    // A ClickHouse `String` is raw bytes; the tokenizer needs UTF-8, so
    // invalid sequences are replaced rather than failing the whole block.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
