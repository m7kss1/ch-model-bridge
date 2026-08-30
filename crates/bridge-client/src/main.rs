//! Thin adapter between the ClickHouse executable-UDF pipe protocol and the
//! daemon socket: `RowBinary` rows in on stdin, results out on stdout. Models
//! stay in the daemon, so pooled UDF processes do not multiply them in memory.
//!
//! ClickHouse is configured with `send_chunk_header = 1`: each block arrives
//! as a decimal row count terminated by `\n`, then that many rows. The reply
//! must contain exactly as many result rows, flushed before the next header
//! is read.
//!
//! Every row the database scans crosses this process, so nothing here copies
//! a row it does not have to: rows are read once, handed to the encoder by
//! reference, and the frames are buffers the process reuses for its lifetime.
//!
//! The daemon connection is opened on first use and replaced whenever it
//! breaks. A daemon restart is the normal way to update models, and it must
//! not cost ClickHouse its pool of warmed-up UDF processes: a request caught
//! mid-flight is sent again over a fresh connection, which is safe because
//! inference has no side effects to repeat.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use protocol::wire::{self, Response};

enum Mode {
    /// Rows of (model, text) -> one `Array(Float32)` per row.
    Embed,
    /// Rows of (model, query, document) -> one `Float32` per row.
    Rerank,
    /// Rows of (model, features `Array(Float32)`) -> one `Float32` per row.
    /// Features arrive at the width the model runs on, so nothing is
    /// converted here. ClickHouse will not narrow a `Float64` expression on
    /// its own — its argument cast is an accurate one and refuses every value
    /// float32 cannot hold exactly, 0.1 included — so the caller casts.
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
            Mode::Embed => embed_block(&mut daemon, &mut input, &mut output, rows)?,
            Mode::Rerank => rerank_block(&mut daemon, &mut input, &mut output, rows)?,
            Mode::Evaluate => evaluate_block(&mut daemon, &mut input, &mut output, rows)?,
        }
        output.flush()?;
    }
    Ok(())
}

/// How long a request keeps being retried once the daemon stops answering,
/// counted from the first failure. Long enough to ride out a restart that
/// serves prepared models, short enough that ClickHouse's own UDF timeouts,
/// ten seconds by default, stay the ones an operator reasons about.
const RETRY_WINDOW: Duration = Duration::from_secs(10);

/// The daemon connection, opened lazily and replaced after any transport
/// failure, together with the two frame buffers every request reuses.
/// ClickHouse keeps a pooled client alive across many blocks, so the buffers
/// grow once to the size of the largest block and never allocate again.
struct Daemon {
    socket: PathBuf,
    stream: Option<UnixStream>,
    request: Vec<u8>,
    reply: Vec<u8>,
}

impl Daemon {
    fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            stream: None,
            request: Vec::new(),
            reply: Vec::new(),
        }
    }

    /// Sends whatever the caller encoded into `request` and decodes the answer.
    /// When the transport fails — the daemon restarting is the expected
    /// reason — the request is resent on a fresh connection until
    /// `RETRY_WINDOW` past the first failure, so a restart briefly stalls
    /// queries instead of failing them and never leaves the pooled process
    /// wedged.
    fn round_trip(&mut self) -> anyhow::Result<Response> {
        let mut give_up: Option<Instant> = None;
        let mut delay = Duration::from_millis(50);
        loop {
            let error = match self.attempt() {
                Ok(()) => {
                    return wire::decode_response(&self.reply)
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

    /// One frame out of `request`, one frame into `reply`.
    fn attempt(&mut self) -> io::Result<()> {
        if self.stream.is_none() {
            self.stream = Some(UnixStream::connect(&self.socket)?);
        }
        let stream = self.stream.as_mut().expect("connected above");
        stream.write_all(&(self.request.len() as u32).to_le_bytes())?;
        stream.write_all(&self.request)?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf);
        if len > wire::MAX_FRAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("daemon sent a frame of {len} bytes, over the limit"),
            ));
        }
        self.reply.resize(len as usize, 0);
        stream.read_exact(&mut self.reply)?;
        Ok(())
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

/// The model name is a per-row argument in SQL, but it is almost always a
/// constant within a block; rows are grouped so each distinct model costs one
/// daemon round-trip. The groups borrow the names — a block of one model
/// copies nothing at all.
fn group_by_model(models: &[String]) -> Vec<(&str, Vec<usize>)> {
    let mut order: Vec<&str> = Vec::new();
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, model) in models.iter().enumerate() {
        match groups.entry(model.as_str()) {
            Entry::Occupied(mut group) => group.get_mut().push(index),
            Entry::Vacant(slot) => {
                order.push(model.as_str());
                slot.insert(vec![index]);
            }
        }
    }
    order
        .into_iter()
        .map(|model| {
            let indices = groups.remove(model).unwrap_or_default();
            (model, indices)
        })
        .collect()
}

fn embed_block(
    daemon: &mut Daemon,
    input: &mut impl BufRead,
    output: &mut impl Write,
    rows: usize,
) -> anyhow::Result<()> {
    let mut models = Vec::with_capacity(rows);
    let mut texts = Vec::with_capacity(rows);
    for _ in 0..rows {
        models.push(read_string(input)?);
        texts.push(read_string(input)?);
    }

    // Answers arrive one flat buffer per model group. Rather than cutting them
    // into a vector per row, each row remembers the group its embedding landed
    // in and the offset there, and is written straight out of that buffer.
    let mut answers: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut placement = vec![(0usize, 0usize); rows];
    for (model, indices) in group_by_model(&models) {
        let group: Vec<&str> = indices.iter().map(|&index| texts[index].as_str()).collect();
        wire::encode_embed(model, &group, &mut daemon.request);
        let response = daemon.round_trip()?;
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
        for (position, &row) in indices.iter().enumerate() {
            placement[row] = (answers.len(), position * dim);
        }
        answers.push((dim, vectors));
    }

    for (answer, offset) in placement {
        let (dim, vectors) = &answers[answer];
        write_varuint(output, *dim as u64)?;
        for value in &vectors[offset..offset + *dim] {
            output.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn rerank_block(
    daemon: &mut Daemon,
    input: &mut impl BufRead,
    output: &mut impl Write,
    rows: usize,
) -> anyhow::Result<()> {
    let mut models = Vec::with_capacity(rows);
    let mut pairs = Vec::with_capacity(rows);
    for _ in 0..rows {
        models.push(read_string(input)?);
        let query = read_string(input)?;
        let document = read_string(input)?;
        pairs.push((query, document));
    }

    let mut scores = vec![0f32; rows];
    for (model, indices) in group_by_model(&models) {
        let group: Vec<(&str, &str)> = indices
            .iter()
            .map(|&index| {
                let (query, document) = &pairs[index];
                (query.as_str(), document.as_str())
            })
            .collect();
        wire::encode_rerank(model, &group, &mut daemon.request);
        let response = daemon.round_trip()?;
        let Response::Rerank { scores: answer } = response else {
            bail!("daemon: {}", response_error(response));
        };
        if answer.len() != indices.len() {
            bail!(
                "daemon returned {} scores for {} pairs",
                answer.len(),
                indices.len()
            );
        }
        for (&slot, score) in indices.iter().zip(answer) {
            scores[slot] = score;
        }
    }

    for score in scores {
        output.write_all(&score.to_le_bytes())?;
    }
    Ok(())
}

fn evaluate_block(
    daemon: &mut Daemon,
    input: &mut impl BufRead,
    output: &mut impl Write,
    rows: usize,
) -> anyhow::Result<()> {
    let mut models = Vec::with_capacity(rows);
    // Feature rows go into one buffer with `starts` marking where each begins,
    // so a row is a slice of it rather than a vector of its own.
    let mut values: Vec<f32> = Vec::new();
    let mut starts: Vec<usize> = Vec::with_capacity(rows + 1);
    for _ in 0..rows {
        models.push(read_string(input)?);
        starts.push(values.len());
        read_f32_array(input, &mut values)?;
    }
    starts.push(values.len());

    let mut scores = vec![0f32; rows];
    let mut group: Vec<f32> = Vec::new();
    for (model, indices) in group_by_model(&models) {
        // The daemon takes a rectangular matrix per request; ragged feature
        // arrays within one model are a caller bug worth naming precisely.
        let n_features = starts[indices[0] + 1] - starts[indices[0]];
        group.clear();
        group.reserve(indices.len() * n_features);
        for &index in &indices {
            let row = &values[starts[index]..starts[index + 1]];
            if row.len() != n_features {
                bail!(
                    "row {index} has {} features while an earlier `{model}` row has {n_features}",
                    row.len()
                );
            }
            group.extend_from_slice(row);
        }
        wire::encode_evaluate(model, n_features as u32, &group, &mut daemon.request);
        let response = daemon.round_trip()?;
        let Response::Evaluate { scores: answer } = response else {
            bail!("daemon: {}", response_error(response));
        };
        if answer.len() != indices.len() {
            bail!(
                "daemon returned {} scores for {} rows",
                answer.len(),
                indices.len()
            );
        }
        for (&slot, score) in indices.iter().zip(answer) {
            scores[slot] = score;
        }
    }

    for score in scores {
        output.write_all(&score.to_le_bytes())?;
    }
    Ok(())
}

fn response_error(response: Response) -> String {
    match response {
        Response::Error(message) => message,
        _ => "unexpected response kind".to_string(),
    }
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

/// A `RowBinary` `Array(Float32)` is a LEB128 element count followed by the
/// elements as `f32` little-endian; they are appended to `out`.
fn read_f32_array(input: &mut impl Read, out: &mut Vec<f32>) -> anyhow::Result<()> {
    let len = read_varuint(input)?;
    for _ in 0..len {
        let mut bytes = [0u8; 4];
        input.read_exact(&mut bytes)?;
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(())
}

fn read_string(input: &mut impl Read) -> anyhow::Result<String> {
    let len = read_varuint(input)? as usize;
    let mut bytes = vec![0u8; len];
    input.read_exact(&mut bytes)?;
    // A ClickHouse `String` is raw bytes; the tokenizer needs UTF-8, so
    // invalid sequences are replaced rather than failing the whole block.
    // Valid bytes — all of them, in practice — become the string in place.
    Ok(String::from_utf8(bytes)
        .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
}
