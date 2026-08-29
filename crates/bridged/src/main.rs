//! clickhouse-model-bridge daemon: serves embedding and rerank models over an
//! OpenAI-compatible HTTP endpoint, so a stock ClickHouse reaches them through
//! `aiEmbed` with a named collection pointing at `--listen`.

mod dispatcher;
mod engine;
mod http;
mod metrics;
mod onnx;
mod registry;
mod uds;

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "bridged", version, about)]
struct Args {
    /// Address of the OpenAI-compatible HTTP endpoint.
    #[arg(long, default_value = "127.0.0.1:9017")]
    listen: String,

    /// Directory with model passports (`*.toml`); every model is
    /// checksum-verified before loading. Skipped when the directory is absent.
    #[arg(long, default_value = "models.d")]
    models_dir: std::path::PathBuf,

    /// Unverified embedding model, as NAME=DIR where DIR contains `model.onnx`
    /// and `tokenizer.json`. Development shortcut that bypasses passports; may
    /// be repeated.
    #[arg(long = "model", value_name = "NAME=DIR")]
    models: Vec<String>,

    /// Dimensionality of the deterministic development stub, registered under
    /// the model name `stub`.
    #[arg(long, default_value_t = 384)]
    stub_dim: usize,

    /// Embedding cache capacity per model, in entries. Repeated texts and
    /// re-runs are answered from the cache without touching the model.
    #[arg(long, default_value_t = 8192)]
    cache_entries: usize,

    /// Unix socket for the binary channel used by `bridge-client` UDFs.
    /// Disabled when not set.
    #[arg(long)]
    socket: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let metrics = std::sync::Arc::new(metrics::Metrics::default());
    let mut registry = registry::Registry::new(Arc::clone(&metrics), args.cache_entries);
    registry.register_embedder(
        "stub",
        "stub",
        0,
        64,
        Arc::new(engine::StubEngine::new(args.stub_dim)),
    );

    if args.models_dir.is_dir() {
        registry.load_models_dir(&args.models_dir)?;
    } else {
        tracing::info!(dir = %args.models_dir.display(), "no passports directory, skipping");
    }

    for spec in &args.models {
        let (name, dir) = spec
            .split_once('=')
            .with_context(|| format!("--model expects NAME=DIR, got `{spec}`"))?;
        tracing::warn!(
            model = name,
            "loading without a passport: files are not checksum-verified"
        );
        let embedder = onnx::OnnxEmbedder::load(std::path::Path::new(dir))
            .with_context(|| format!("loading model `{name}` from {dir}"))?;
        registry.register_embedder(name, "onnx", 0, 64, Arc::new(embedder));
    }

    let state = http::AppState {
        registry: Arc::new(registry),
        metrics,
    };

    if let Some(socket) = args.socket {
        let registry = Arc::clone(&state.registry);
        tokio::spawn(async move {
            if let Err(e) = uds::serve(socket, registry).await {
                tracing::error!("binary channel failed: {e:#}");
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, models = ?state.registry.names(), "serving");

    axum::serve(listener, http::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("stopped");
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
