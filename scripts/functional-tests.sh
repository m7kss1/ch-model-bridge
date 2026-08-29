#!/usr/bin/env bash
# Runs the functional suite the same way locally and in CI.
#
#   scripts/functional-tests.sh                       # core tier, no prerequisites
#   scripts/functional-tests.sh --tier models         # needs downloaded models
#   scripts/functional-tests.sh --tier all --fetch    # downloads what it needs first
#
# Tiers:
#   core        every channel and every error path, served by the built-in stub
#   models      regressions against the real ONNX models, including the numbers
#               the README publishes
#   clickhouse  end to end through a real `clickhouse local`
#
# The tests find the binaries through MODEL_BRIDGE_BIN_DIR, so the same suite
# can be pointed at a release build or at binaries taken from the image.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/.cache"
# Bump the version together with CLICKHOUSE_VERSION in ci.yml — and the two
# checksums with it.
PINNED_CLICKHOUSE_VERSION="26.7.5.10"
PINNED_CLICKHOUSE_SHA512_AMD64="b0dfc37f96db0a9b87925c5543959759ebaaa8007c32a2e2a2e7ae07a3b4cd44c69785a52344f033e6d75705eafc304744db96e3f67a8229b614f4d15ffada9c"
PINNED_CLICKHOUSE_SHA512_ARM64="6d49abeb5b8fe3f8a15718b5657707819bae3eba77c7b59e4bcbf89358a1fb3b4d70a48b34b74f05e20734dbd2ad946a777656f51749212ce44a1ca3f410c4a7"
CLICKHOUSE_VERSION="${CLICKHOUSE_VERSION:-$PINNED_CLICKHOUSE_VERSION}"

tier=core
profile=release
fetch=0
train=0
filter=()

while [ $# -gt 0 ]; do
    case "$1" in
        --tier) tier="$2"; shift 2 ;;
        --debug) profile=debug; shift ;;
        --fetch) fetch=1; shift ;;
        --train-tabular) train=1; shift ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) filter+=("$1"); shift ;;
    esac
done

case "$tier" in
    core|models|clickhouse|all) ;;
    *) echo "unknown tier: $tier (core, models, clickhouse, all)" >&2; exit 2 ;;
esac

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- build -------------------------------------------------------------------
# An externally supplied MODEL_BRIDGE_BIN_DIR wins: that is how CI runs the
# suite against the artifacts it is about to ship, and how anyone can run it
# against binaries taken out of the container image.
if [ -n "${MODEL_BRIDGE_BIN_DIR:-}" ]; then
    say "testing the binaries in $MODEL_BRIDGE_BIN_DIR"
else
    say "building the binaries under test ($profile)"
    if [ "$profile" = release ]; then
        cargo build --release --locked --workspace
    else
        cargo build --locked --workspace
    fi
    export MODEL_BRIDGE_BIN_DIR="$ROOT/target/$profile"
fi

# --- prerequisites -----------------------------------------------------------
needs_models=0
needs_clickhouse=0
case "$tier" in models|all) needs_models=1 ;; esac
case "$tier" in clickhouse|all) needs_clickhouse=1; needs_models=1 ;; esac

if [ "$fetch" = 1 ] && [ "$needs_models" = 1 ]; then
    say "downloading catalog models into $ROOT/models"
    # Passports go to a scratch directory: the tests issue their own, and any
    # passports you keep in models.d/ must stay untouched.
    for model in multilingual-e5-small bge-reranker-base; do
        "$MODEL_BRIDGE_BIN_DIR/model-bridge" fetch "$model" \
            --models-root "$ROOT/models" --passports "$CACHE/passports"
    done
fi

if [ "$train" = 1 ]; then
    say "training the tabular demo model"
    python3 "$ROOT/examples/train_fraud_model.py"
fi

if [ "$fetch" = 1 ] && [ "$needs_clickhouse" = 1 ] && [ -z "${MODEL_BRIDGE_CLICKHOUSE:-}" ]; then
    binary="$CACHE/clickhouse-$CLICKHOUSE_VERSION/clickhouse"
    if [ ! -x "$binary" ]; then
        say "downloading ClickHouse $CLICKHOUSE_VERSION"
        case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; *) arch=amd64 ;; esac
        url="https://packages.clickhouse.com/tgz/stable/clickhouse-common-static-$CLICKHOUSE_VERSION-$arch.tgz"
        mkdir -p "$CACHE/clickhouse-$CLICKHOUSE_VERSION"
        tarball="$CACHE/clickhouse-$CLICKHOUSE_VERSION/tarball.tgz"
        curl -fsSL -o "$tarball" "$url"
        # The tarball feeds a test tier, so it is pinned like everything else.
        # Overriding CLICKHOUSE_VERSION opts out of the pin, loudly.
        if [ "$CLICKHOUSE_VERSION" = "$PINNED_CLICKHOUSE_VERSION" ]; then
            case "$arch" in
                arm64) sum="$PINNED_CLICKHOUSE_SHA512_ARM64" ;;
                *) sum="$PINNED_CLICKHOUSE_SHA512_AMD64" ;;
            esac
            echo "$sum  $tarball" | sha512sum --check --quiet -
        else
            echo "note: no pinned checksum for ClickHouse $CLICKHOUSE_VERSION; downloaded unverified" >&2
        fi
        tar -xzf "$tarball" -C "$CACHE/clickhouse-$CLICKHOUSE_VERSION" --strip-components=3 \
            "clickhouse-common-static-$CLICKHOUSE_VERSION/usr/bin/clickhouse"
        rm -f "$tarball"
    fi
    export MODEL_BRIDGE_CLICKHOUSE="$binary"
fi

# A tier that cannot find its prerequisites must fail, not quietly skip.
required=()
if [ "$needs_models" = 1 ]; then required+=(models tabular-reference); fi
if [ "$needs_clickhouse" = 1 ]; then required+=(clickhouse); fi
if [ ${#required[@]} -gt 0 ]; then
    MODEL_BRIDGE_REQUIRE_TIERS=$(IFS=,; echo "${required[*]}")
    export MODEL_BRIDGE_REQUIRE_TIERS
fi

# --- run ---------------------------------------------------------------------
targets=()
case "$tier" in
    core) targets=(http_api binary_channel udf_client cli passports) ;;
    models) targets=(models) ;;
    clickhouse) targets=(clickhouse) ;;
    all) targets=(http_api binary_channel udf_client cli passports models clickhouse) ;;
esac

args=(test -p functional-tests --locked)
for target in "${targets[@]}"; do
    args+=(--test "$target")
done

say "running the $tier tier"
cargo "${args[@]}" -- "${filter[@]}"
