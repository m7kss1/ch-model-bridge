# syntax=docker/dockerfile:1

# Build and runtime share a Debian release, and both are trixie for one
# reason: the prebuilt ONNX Runtime that `ort` downloads is compiled against
# glibc 2.38+ and links `__isoc23_strtoll`, so bookworm (2.36) fails at link
# time. Bases are pinned by digest and kept current by Dependabot; the tag in
# the comment is what the digest resolved from.
FROM rust:1.98-trixie@sha256:271849e998ffce5776454bbf98c5dc21baafc854ff8e566197908d3aca9a81e8 AS builder

ENV CARGO_TERM_COLOR=never \
    CARGO_INCREMENTAL=0 \
    CARGO_NET_RETRY=5

WORKDIR /src

# Dependencies first, compiled against stub sources. The result is a layer that
# survives every change that does not touch a manifest or the lockfile, which
# is what keeps image builds from recompiling ONNX Runtime, tokenizers and
# tokio each time. A --mount=type=cache would be faster locally and useless in
# CI, where cache mounts start empty and only layers are restored.
COPY Cargo.toml Cargo.lock ./
COPY crates/protocol/Cargo.toml crates/protocol/
COPY crates/bridged/Cargo.toml crates/bridged/
COPY crates/bridge-client/Cargo.toml crates/bridge-client/
COPY crates/bridge-cli/Cargo.toml crates/bridge-cli/
COPY crates/functional-tests/Cargo.toml crates/functional-tests/
RUN mkdir -p crates/protocol/src crates/bridged/src crates/bridge-client/src \
             crates/bridge-cli/src crates/functional-tests/src \
 && : > crates/protocol/src/lib.rs \
 && : > crates/functional-tests/src/lib.rs \
 && for crate in bridged bridge-client bridge-cli; do \
        echo 'fn main() {}' > "crates/$crate/src/main.rs"; \
    done \
 && cargo build --release --locked --bin bridged --bin bridge-client --bin model-bridge

# The real sources. Dropping the stub fingerprints forces the workspace crates
# to be rebuilt while every third-party crate stays cached in the layer above.
# Only the three shipped binaries are built; the functional-test harness is a
# workspace member and stays out of the image.
COPY crates crates
RUN rm -rf target/release/.fingerprint/protocol-* \
           target/release/.fingerprint/bridged-* \
           target/release/.fingerprint/bridge-cli-* \
           target/release/.fingerprint/bridge-client-* \
 && cargo build --release --locked --bin bridged --bin bridge-client --bin model-bridge \
 && mkdir -p /out \
 && cp target/release/bridged target/release/bridge-client target/release/model-bridge /out/ \
 && strip /out/bridged /out/bridge-client /out/model-bridge

# ------------------------------------------------------------------- runtime
FROM debian:trixie-slim@sha256:d7e12182ce18b85b93007c1dedf31f2d29e01ccf3182cc4017c709b6259bc132 AS runtime

# libstdc++6: ONNX Runtime is statically linked into the daemon, but its C++
#   standard library is not.
# ca-certificates: `model-bridge fetch` downloads models over HTTPS.
# curl: the HEALTHCHECK below, and hand debugging of the endpoints.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# Fixed uid/gid: the socket directory is shared with ClickHouse, and a stable
# owner is what makes that mount's permissions predictable across hosts. Only
# the parent directory is created: the daemon refuses to start on an existing
# but empty passports directory, while an absent one means `no models yet` and
# leaves the built-in stub embedder serving.
RUN groupadd --system --gid 10001 bridge \
 && useradd --system --uid 10001 --gid bridge --home-dir /var/lib/model-bridge \
            --shell /usr/sbin/nologin bridge \
 && mkdir -p /var/lib/model-bridge /run/model-bridge \
 && chown -R bridge:bridge /var/lib/model-bridge /run/model-bridge

COPY --from=builder /out/ /usr/local/bin/

LABEL org.opencontainers.image.title="clickhouse-model-bridge" \
      org.opencontainers.image.description="Local ONNX inference for ClickHouse SQL: embeddings, reranking and tabular scoring over executable UDFs and an OpenAI-compatible endpoint" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/m7kss1/ch-model-bridge"

USER bridge:bridge
WORKDIR /var/lib/model-bridge
ENV RUST_LOG=info
EXPOSE 9017

# Graceful shutdown is wired to SIGINT. As PID 1 the daemon would ignore the
# default SIGTERM and wait out the ten-second kill timer on every stop.
STOPSIGNAL SIGINT

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9017/health || exit 1

# No ENTRYPOINT: the image ships three binaries, so `docker run IMAGE
# model-bridge fetch ...` and `docker run IMAGE bridged --help` both work.
# Binding 0.0.0.0 inside the container is safe because publishing is the
# operator's choice — see the README for `-p 127.0.0.1:9017:9017`.
CMD ["bridged", \
     "--listen", "0.0.0.0:9017", \
     "--models-dir", "/var/lib/model-bridge/models.d", \
     "--socket", "/run/model-bridge/bridge.sock"]
