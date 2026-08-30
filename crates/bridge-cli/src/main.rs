//! Administrator CLI: fetches models, issues passports, generates ClickHouse
//! UDF configs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use clap::{Args, Parser, Subcommand, ValueEnum};
use protocol::passport::{sha256_file, ModelKind, Passport};

#[derive(Parser)]
#[command(
    name = "model-bridge",
    version,
    about = "Administrator utility for clickhouse-model-bridge"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model from the built-in catalog and issue its passport.
    Fetch {
        name: String,
        #[arg(long, default_value = "models")]
        models_root: PathBuf,
        #[arg(long, default_value = "models.d")]
        passports: PathBuf,
    },
    /// Issue a passport for a model directory already on disk — the air-gapped
    /// path, where files are brought in by any external means.
    Passport {
        dir: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long, value_enum)]
        kind: KindArg,
        #[arg(long, default_value_t = 1)]
        revision: u32,
        #[command(flatten)]
        tuning: ServingTuning,
        #[arg(long, default_value = "models.d")]
        passports: PathBuf,
    },
    /// Generate executable-UDF XML configs for ClickHouse.
    GenConfigs {
        /// Path to the `bridge-client` binary that ClickHouse will spawn.
        #[arg(long)]
        client: PathBuf,
        /// Unix socket the daemon listens on.
        #[arg(long)]
        socket: PathBuf,
        /// Output directory. Deliberately not `clickhouse`: that name is
        /// taken by the binary itself in the likeliest working directory.
        #[arg(long, default_value = "bridge-configs")]
        out: PathBuf,
        #[command(flatten)]
        tuning: UdfTuning,
    },
}

/// Serving knobs recorded in the passport. Every one is optional: an absent
/// value is not written, and the daemon resolves the default at load time —
/// `max_batch` by model kind, `sessions` to one, `max_tokens` from
/// `tokenizer.json` or the 512 family default.
#[derive(Args)]
struct ServingTuning {
    /// Rows per ONNX run. Defaults by kind: 64 for embedding and rerank,
    /// 65536 (a whole ClickHouse block) for tabular.
    #[arg(long)]
    max_batch: Option<usize>,
    /// Parallel ONNX sessions the daemon keeps for this model. The daemon
    /// caps the value to the host's core count at load time.
    #[arg(long)]
    sessions: Option<usize>,
    /// Truncation limit in tokens for text models with a context longer than
    /// the 512-token default, such as 8k-context encoders.
    #[arg(long)]
    max_tokens: Option<usize>,
}

impl ServingTuning {
    fn unspecified() -> Self {
        Self {
            max_batch: None,
            sessions: None,
            max_tokens: None,
        }
    }
}

/// Tuning written into every generated function, named after the ClickHouse
/// config keys it feeds. Overrides belong here and not in the generated XML,
/// which the next `gen-configs` run overwrites.
#[derive(Args)]
struct UdfTuning {
    /// `bridge-client` processes ClickHouse may spawn per function. Slots
    /// gate concurrent queries; inference concurrency lives in the daemon.
    #[arg(long, default_value_t = 16)]
    pool_size: u64,
    /// How long ClickHouse waits for a block's results, in milliseconds. The
    /// client replies only after the daemon finishes the whole batch, so this
    /// bounds one block's inference time.
    #[arg(long, default_value_t = 120_000)]
    command_read_timeout: u64,
    /// How long a query waits for a free pool process, in seconds.
    #[arg(long, default_value_t = 120)]
    max_command_execution_time: u64,
}

#[derive(Clone, Copy, ValueEnum)]
enum KindArg {
    Embedding,
    Rerank,
    Tabular,
}

impl From<KindArg> for ModelKind {
    fn from(kind: KindArg) -> Self {
        match kind {
            KindArg::Embedding => ModelKind::Embedding,
            KindArg::Rerank => ModelKind::Rerank,
            KindArg::Tabular => ModelKind::Tabular,
        }
    }
}

struct CatalogFile {
    name: &'static str,
    url: &'static str,
    /// Pinned upstream checksum. `None` means trust-on-first-use: the checksum
    /// is computed at fetch time and recorded in the passport.
    sha256: Option<&'static str>,
}

struct CatalogEntry {
    name: &'static str,
    kind: ModelKind,
    files: &'static [CatalogFile],
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        name: "multilingual-e5-small",
        kind: ModelKind::Embedding,
        files: &[
            CatalogFile {
                name: "model.onnx",
                url: "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/onnx/model_quantized.onnx",
                sha256: Some("f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193"),
            },
            CatalogFile {
                name: "tokenizer.json",
                url: "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/tokenizer.json",
                sha256: Some("0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
            },
        ],
    },
    CatalogEntry {
        name: "bge-reranker-base",
        kind: ModelKind::Rerank,
        files: &[
            CatalogFile {
                name: "model.onnx",
                url: "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx/model_quantized.onnx",
                sha256: Some("dd98f3e67837d23210a6b7550c08cced4f61845b940ac45be3565840a10f3244"),
            },
            CatalogFile {
                name: "tokenizer.json",
                url: "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/tokenizer.json",
                sha256: Some("48564c5c7d3fa64d85d95e65414a542385f88b0f128fd8d4163fd7a57f2be05c"),
            },
        ],
    },
];

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Fetch {
            name,
            models_root,
            passports,
        } => fetch(&name, &models_root, &passports),
        Command::Passport {
            dir,
            name,
            kind,
            revision,
            tuning,
            passports,
        } => issue_passport(&dir, &name, kind.into(), revision, &tuning, &passports),
        Command::GenConfigs {
            client,
            socket,
            out,
            tuning,
        } => gen_configs(&client, &socket, &out, &tuning),
    }
}

fn fetch(name: &str, models_root: &Path, passports: &Path) -> anyhow::Result<()> {
    let entry = CATALOG
        .iter()
        .find(|entry| entry.name == name)
        .ok_or_else(|| {
            let known: Vec<_> = CATALOG.iter().map(|entry| entry.name).collect();
            anyhow!(
                "`{name}` is not in the catalog; known models: {}",
                known.join(", ")
            )
        })?;

    let dir = models_root.join(name);
    std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;

    let mut sums = BTreeMap::new();
    for file in entry.files {
        let target = dir.join(file.name);
        if target.exists() {
            let sum = sha256_file(&target)?;
            match file.sha256 {
                Some(pinned) if pinned != sum => {
                    eprintln!(
                        "{}: checksum differs from the catalog, re-downloading",
                        target.display()
                    );
                }
                _ => {
                    eprintln!("{}: already present", target.display());
                    sums.insert(file.name.to_string(), sum);
                    continue;
                }
            }
        }
        download(file.url, &target)?;
        let sum = sha256_file(&target)?;
        if let Some(pinned) = file.sha256 {
            if pinned != sum {
                std::fs::remove_file(&target).ok();
                bail!(
                    "{}: checksum mismatch (expected {pinned}, got {sum})",
                    file.url
                );
            }
        } else {
            eprintln!(
                "{}: recorded sha256 {sum} (trust-on-first-use)",
                target.display()
            );
        }
        sums.insert(file.name.to_string(), sum);
    }

    write_passport(
        entry.name,
        entry.kind,
        &dir,
        1,
        &ServingTuning::unspecified(),
        sums,
        passports,
    )
}

fn download(url: &str, target: &Path) -> anyhow::Result<()> {
    eprintln!("downloading {url}");
    let response = ureq::get(url).call().map_err(|e| anyhow!("{url}: {e}"))?;
    let mut reader = response.into_body().into_reader();

    // Download into a `.part` file so an interrupted transfer never leaves a
    // truncated file under the final name.
    let part = target.with_extension("part");
    let mut out = std::fs::File::create(&part).with_context(|| part.display().to_string())?;
    std::io::copy(&mut reader, &mut out).with_context(|| format!("downloading {url}"))?;
    std::fs::rename(&part, target)?;
    Ok(())
}

fn issue_passport(
    dir: &Path,
    name: &str,
    kind: ModelKind,
    revision: u32,
    tuning: &ServingTuning,
    passports: &Path,
) -> anyhow::Result<()> {
    // Text models are inseparable from their tokenizer, so the passport
    // covers both files; a tabular model is the graph alone.
    let files: &[&str] = match kind {
        ModelKind::Tabular => &["model.onnx"],
        _ => &["model.onnx", "tokenizer.json"],
    };
    let mut sums = BTreeMap::new();
    for file in files {
        let path = dir.join(file);
        if !path.is_file() {
            bail!(
                "{}: not found; a model directory needs {}",
                path.display(),
                files.join(" and ")
            );
        }
        sums.insert(file.to_string(), sha256_file(&path)?);
    }
    write_passport(name, kind, dir, revision, tuning, sums, passports)
}

fn write_passport(
    name: &str,
    kind: ModelKind,
    dir: &Path,
    revision: u32,
    tuning: &ServingTuning,
    sha256: BTreeMap<String, String>,
    passports: &Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(passports).with_context(|| passports.display().to_string())?;
    let passport = Passport {
        name: name.to_string(),
        kind,
        dir: dir
            .canonicalize()
            .with_context(|| dir.display().to_string())?,
        revision,
        max_batch: tuning.max_batch,
        sessions: tuning.sessions,
        max_tokens: tuning.max_tokens,
        sha256,
    };
    let path = passports.join(format!("{name}.toml"));
    passport.save(&path)?;
    println!("passport: {}", path.display());
    Ok(())
}

/// One generated UDF: SQL name, ClickHouse return type, its `(type, name)`
/// arguments, and the `bridge-client` mode that serves it.
type FunctionSpec = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static str,
);

/// Emits everything ClickHouse needs to expose `localEmbed` and `localRerank`
/// as executable UDFs: the functions XML and a scripts directory with the
/// thin client, which must live under `user_scripts_path` because ClickHouse
/// only spawns commands from there.
fn gen_configs(client: &Path, socket: &Path, out: &Path, tuning: &UdfTuning) -> anyhow::Result<()> {
    // Said outright rather than left to create_dir_all's "Not a directory":
    // the file in the way is typically the `clickhouse` binary, which the
    // official installer drops into the working directory.
    if out.exists() && !out.is_dir() {
        bail!(
            "{}: a file is in the way of the output directory; pick another with --out",
            out.display()
        );
    }
    let scripts = out.join("scripts");
    std::fs::create_dir_all(&scripts).with_context(|| scripts.display().to_string())?;

    let client_target = scripts.join("bridge-client");
    std::fs::copy(client, &client_target).with_context(|| {
        format!(
            "copying {} to {}",
            client.display(),
            client_target.display()
        )
    })?;

    let socket = socket
        .canonicalize()
        .unwrap_or_else(|_| socket.to_path_buf());

    let functions: &[FunctionSpec] = &[
        (
            "localEmbed",
            "Array(Float32)",
            &[("String", "model"), ("String", "text")],
            "embed",
        ),
        (
            "localRerank",
            "Float32",
            &[
                ("String", "model"),
                ("String", "query"),
                ("String", "document"),
            ],
            "rerank",
        ),
        (
            "modelEvaluate",
            "Float32",
            &[("String", "model"), ("Array(Float64)", "features")],
            "evaluate",
        ),
    ];

    let mut xml = String::from(
        "<!-- Generated by `model-bridge gen-configs`; the next run overwrites this\n\
         file, so retune through that command's flags instead of editing here. -->\n\
         <functions>\n",
    );
    for (name, return_type, arguments, mode) in functions {
        xml.push_str("    <function>\n        <type>executable_pool</type>\n");
        xml.push_str(&format!("        <name>{name}</name>\n"));
        xml.push_str(&format!(
            "        <return_type>{return_type}</return_type>\n"
        ));
        for (arg_type, arg_name) in *arguments {
            xml.push_str(&format!(
                "        <argument><type>{arg_type}</type><name>{arg_name}</name></argument>\n"
            ));
        }
        // `deterministic` holds because passports pin model revisions, and it
        // is what admits these functions into `ALTER TABLE ... UPDATE`
        // backfills on `Replicated*` tables.
        xml.push_str(&format!(
            "        <format>RowBinary</format>\n\
             \x20       <command>bridge-client {mode} --socket {}</command>\n\
             \x20       <send_chunk_header>1</send_chunk_header>\n\
             \x20       <pool_size>{}</pool_size>\n\
             \x20       <command_read_timeout>{}</command_read_timeout>\n\
             \x20       <max_command_execution_time>{}</max_command_execution_time>\n\
             \x20       <deterministic>1</deterministic>\n\
             \x20   </function>\n",
            socket.display(),
            tuning.pool_size,
            tuning.command_read_timeout,
            tuning.max_command_execution_time,
        ));
    }
    xml.push_str("</functions>\n");

    let xml_path = out.join("model_bridge_functions.xml");
    std::fs::write(&xml_path, xml).with_context(|| xml_path.display().to_string())?;

    let out_abs = out.canonicalize().unwrap_or_else(|_| out.to_path_buf());
    println!("wrote {}", xml_path.display());
    println!("wrote {}", client_target.display());
    println!("\nadd to the ClickHouse server config:");
    println!(
        "  <user_defined_executable_functions_config>{}/model_bridge_functions.xml</user_defined_executable_functions_config>",
        out_abs.display()
    );
    println!(
        "  <user_scripts_path>{}/scripts</user_scripts_path>",
        out_abs.display()
    );
    Ok(())
}
