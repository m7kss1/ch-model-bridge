//! Shared contract between the daemon, the UDF thin client and the CLI.
//!
//! The wire protocol is deliberately dependency-free: the thin client links
//! only this crate and the standard library.

pub const PROTOCOL_VERSION: u8 = 1;

#[cfg(feature = "passport")]
pub mod passport;

/// Binary channel between `bridge-client` and the daemon socket. Every message
/// is one length-prefixed frame: `u32` little-endian payload size, then the
/// payload encoded by this module. One request frame yields exactly one
/// response frame.
pub mod wire {
    use super::PROTOCOL_VERSION;

    /// Refuse frames larger than this instead of allocating unboundedly on a
    /// corrupt length prefix.
    pub const MAX_FRAME: u32 = 256 << 20;

    const TASK_EMBED: u8 = 0;
    const TASK_RERANK: u8 = 1;
    const TASK_EVALUATE: u8 = 2;

    const STATUS_OK: u8 = 0;
    const STATUS_ERROR: u8 = 1;

    #[derive(Debug, PartialEq)]
    pub enum Request {
        Embed {
            model: String,
            texts: Vec<String>,
        },
        Rerank {
            model: String,
            pairs: Vec<(String, String)>,
        },
        /// Feature rows are flattened row-major; `n_features` recovers the row
        /// boundaries.
        Evaluate {
            model: String,
            n_features: u32,
            values: Vec<f32>,
        },
    }

    #[derive(Debug, PartialEq)]
    pub enum Response {
        /// Vectors are flattened row-major; `dim` recovers the row boundaries.
        Embed {
            dim: u32,
            vectors: Vec<f32>,
        },
        Rerank {
            scores: Vec<f32>,
        },
        Evaluate {
            scores: Vec<f32>,
        },
        Error(String),
    }

    /// Encodes an embed request into `out`, replacing its contents. Texts are
    /// borrowed and the buffer is the caller's: the thin client encodes
    /// straight from the rows it read, into a frame it reuses for every block.
    pub fn encode_embed(model: &str, texts: &[&str], out: &mut Vec<u8>) {
        let payload = texts.iter().map(|text| 4 + text.len()).sum();
        start(out, TASK_EMBED, model, texts.len() as u32, payload);
        for text in texts {
            put_str(out, text);
        }
    }

    /// Encodes a rerank request into `out`, replacing its contents.
    pub fn encode_rerank(model: &str, pairs: &[(&str, &str)], out: &mut Vec<u8>) {
        let payload = pairs
            .iter()
            .map(|(query, document)| 8 + query.len() + document.len())
            .sum();
        start(out, TASK_RERANK, model, pairs.len() as u32, payload);
        for (query, document) in pairs {
            put_str(out, query);
            put_str(out, document);
        }
    }

    /// Encodes an evaluate request into `out`, replacing its contents.
    /// `values` is the row-major feature matrix; `n_features` recovers the row
    /// boundaries.
    pub fn encode_evaluate(model: &str, n_features: u32, values: &[f32], out: &mut Vec<u8>) {
        let rows = (values.len() as u32).checked_div(n_features).unwrap_or(0);
        start(out, TASK_EVALUATE, model, rows, 4 + values.len() * 4);
        out.extend_from_slice(&n_features.to_le_bytes());
        put_f32s(out, values);
    }

    /// For callers that already hold an owned request — the daemon's own
    /// tests, mostly. Hot paths use the borrowed encoders above.
    pub fn encode_request(request: &Request) -> Vec<u8> {
        let mut out = Vec::new();
        match request {
            Request::Embed { model, texts } => {
                let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
                encode_embed(model, &texts, &mut out);
            }
            Request::Rerank { model, pairs } => {
                let pairs: Vec<(&str, &str)> = pairs
                    .iter()
                    .map(|(query, document)| (query.as_str(), document.as_str()))
                    .collect();
                encode_rerank(model, &pairs, &mut out);
            }
            Request::Evaluate {
                model,
                n_features,
                values,
            } => encode_evaluate(model, *n_features, values, &mut out),
        }
        out
    }

    /// The head every request shares — version, task kind, model name, item
    /// count — with room for `payload` more bytes taken in one go, so a reused
    /// buffer stops reallocating after the first block.
    fn start(out: &mut Vec<u8>, task: u8, model: &str, count: u32, payload: usize) {
        out.clear();
        out.reserve(2 + 4 + model.len() + 4 + payload);
        out.push(PROTOCOL_VERSION);
        out.push(task);
        put_str(out, model);
        out.extend_from_slice(&count.to_le_bytes());
    }

    pub fn decode_request(payload: &[u8]) -> Result<Request, String> {
        let mut cursor = Cursor::new(payload);
        let version = cursor.u8()?;
        if version != PROTOCOL_VERSION {
            return Err(format!(
                "protocol version mismatch: daemon speaks v{PROTOCOL_VERSION}, client sent v{version}"
            ));
        }
        let task = cursor.u8()?;
        let model = cursor.str()?;
        let count = cursor.u32()? as usize;
        match task {
            TASK_EMBED => {
                let mut texts = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    texts.push(cursor.str()?);
                }
                cursor.finish()?;
                Ok(Request::Embed { model, texts })
            }
            TASK_RERANK => {
                let mut pairs = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    let query = cursor.str()?;
                    let document = cursor.str()?;
                    pairs.push((query, document));
                }
                cursor.finish()?;
                Ok(Request::Rerank { model, pairs })
            }
            TASK_EVALUATE => {
                let n_features = cursor.u32()?;
                let total = count
                    .checked_mul(n_features as usize)
                    .ok_or("row count overflow")?;
                let values = cursor.f32s(total)?;
                cursor.finish()?;
                Ok(Request::Evaluate {
                    model,
                    n_features,
                    values,
                })
            }
            other => Err(format!("unknown task kind {other}")),
        }
    }

    pub fn encode_response(response: &Response) -> Vec<u8> {
        let mut out = Vec::new();
        match response {
            Response::Embed { dim, vectors } => {
                out.push(STATUS_OK);
                out.push(TASK_EMBED);
                out.extend(dim.to_le_bytes());
                out.extend((vectors.len() as u32).to_le_bytes());
                put_f32s(&mut out, vectors);
            }
            Response::Rerank { scores } => {
                out.push(STATUS_OK);
                out.push(TASK_RERANK);
                out.extend((scores.len() as u32).to_le_bytes());
                put_f32s(&mut out, scores);
            }
            Response::Evaluate { scores } => {
                out.push(STATUS_OK);
                out.push(TASK_EVALUATE);
                out.extend((scores.len() as u32).to_le_bytes());
                put_f32s(&mut out, scores);
            }
            Response::Error(message) => {
                out.push(STATUS_ERROR);
                let bytes = message.as_bytes();
                out.extend((bytes.len() as u32).to_le_bytes());
                out.extend(bytes);
            }
        }
        out
    }

    pub fn decode_response(payload: &[u8]) -> Result<Response, String> {
        let mut cursor = Cursor::new(payload);
        match cursor.u8()? {
            STATUS_OK => match cursor.u8()? {
                TASK_EMBED => {
                    let dim = cursor.u32()?;
                    let count = cursor.u32()? as usize;
                    let vectors = cursor.f32s(count)?;
                    cursor.finish()?;
                    Ok(Response::Embed { dim, vectors })
                }
                TASK_RERANK => {
                    let count = cursor.u32()? as usize;
                    let scores = cursor.f32s(count)?;
                    cursor.finish()?;
                    Ok(Response::Rerank { scores })
                }
                TASK_EVALUATE => {
                    let count = cursor.u32()? as usize;
                    let scores = cursor.f32s(count)?;
                    cursor.finish()?;
                    Ok(Response::Evaluate { scores })
                }
                other => Err(format!("unknown task kind {other} in response")),
            },
            STATUS_ERROR => {
                let message = cursor.str_u32()?;
                Ok(Response::Error(message))
            }
            other => Err(format!("unknown status byte {other}")),
        }
    }

    fn put_str(out: &mut Vec<u8>, value: &str) {
        out.extend((value.len() as u32).to_le_bytes());
        out.extend(value.as_bytes());
    }

    /// A block of embeddings runs to hundreds of megabytes; growing the buffer
    /// four bytes at a time would recopy all of it on the way.
    fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
        out.reserve(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    struct Cursor<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self { data, pos: 0 }
        }

        fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
            let end = self
                .pos
                .checked_add(n)
                .filter(|&end| end <= self.data.len())
                .ok_or_else(|| "frame truncated".to_string())?;
            let slice = &self.data[self.pos..end];
            self.pos = end;
            Ok(slice)
        }

        fn u8(&mut self) -> Result<u8, String> {
            Ok(self.take(1)?[0])
        }

        fn u32(&mut self) -> Result<u32, String> {
            Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
        }

        fn str(&mut self) -> Result<String, String> {
            self.str_u32()
        }

        fn str_u32(&mut self) -> Result<String, String> {
            let len = self.u32()? as usize;
            let bytes = self.take(len)?;
            String::from_utf8(bytes.to_vec()).map_err(|e| format!("invalid utf-8: {e}"))
        }

        fn f32s(&mut self, count: usize) -> Result<Vec<f32>, String> {
            let bytes = self.take(count.checked_mul(4).ok_or("frame size overflow")?)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect())
        }

        fn finish(&self) -> Result<(), String> {
            if self.pos == self.data.len() {
                Ok(())
            } else {
                Err(format!(
                    "{} trailing bytes in frame",
                    self.data.len() - self.pos
                ))
            }
        }
    }
}
