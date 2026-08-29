#!/usr/bin/env bash
# Proves a built image actually serves. Covers both channels — HTTP and the
# unix socket the ClickHouse UDFs use — plus the container contract: health,
# non-root user, graceful shutdown. No model files needed: the daemon's
# built-in `stub` embedder answers on both channels.
#
#   scripts/smoke-test.sh [IMAGE]
set -Eeuo pipefail

IMAGE="${1:-ch-model-bridge:dev}"
CONTAINER="model-bridge-smoke-$$"
SOCKET="/run/model-bridge/bridge.sock"
STUB_DIM=384

fail() { echo "smoke: FAIL: $*" >&2; exit 1; }
ok()   { echo "smoke: ok    $*"; }

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "smoke: testing $IMAGE"
docker run --detach --name "$CONTAINER" --publish 127.0.0.1::9017 "$IMAGE" >/dev/null
published=$(docker port "$CONTAINER" 9017/tcp 2>/dev/null | head -1) || true
[ -n "$published" ] || fail "no published port, the container did not start:
$(docker logs "$CONTAINER" 2>&1 | tail -20)"
base="http://$published"

# --- readiness ---------------------------------------------------------------
for _ in $(seq 30); do
    curl -fsS "$base/health" >/dev/null 2>&1 && break
    docker inspect --format '{{.State.Running}}' "$CONTAINER" | grep -qx true \
        || fail "container exited during startup:
$(docker logs "$CONTAINER" 2>&1 | tail -20)"
    sleep 1
done
curl -fsS "$base/health" | grep -qx ok || fail "/health did not answer ok"
ok "/health on the published port"

# The HEALTHCHECK baked into the image is part of the contract; docker only
# runs the first probe an interval after start, hence the generous budget.
for _ in $(seq 60); do
    health=$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER")
    [ "$health" = healthy ] && break
    [ "$health" = unhealthy ] && fail "container reported unhealthy"
    sleep 1
done
[ "$health" = healthy ] || fail "container never became healthy"
ok "container HEALTHCHECK reports healthy"

# --- the image contract ------------------------------------------------------
[ "$(docker exec "$CONTAINER" id -u)" = 10001 ] || fail "daemon is not running as uid 10001"
ok "daemon runs as a non-root fixed uid"

curl -fsS "$base/v1/models" | grep -q '"stub"' || fail "/v1/models does not list the stub model"
ok "/v1/models lists the loaded models"

# --- channel B: OpenAI-compatible HTTP ---------------------------------------
http_json=$(curl -fsS -X POST "$base/v1/embeddings" \
    -H 'content-type: application/json' \
    -d '{"model":"stub","input":["hello","world"]}')
echo "$http_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)['data']
assert len(data) == 2, f'expected 2 embeddings, got {len(data)}'
assert len(data[0]['embedding']) == $STUB_DIM, f\"expected $STUB_DIM dims, got {len(data[0]['embedding'])}\"
" || fail "/v1/embeddings returned an unexpected shape"
ok "/v1/embeddings returns 2 x $STUB_DIM floats"

# --- channel A: the UDF binary channel ---------------------------------------
# `bridge-client` speaks the ClickHouse executable-UDF pipe protocol: a decimal
# row count, then RowBinary rows of (model, text); out come RowBinary
# Array(Float32) rows. Feeding it by hand is what ClickHouse would do.
udf_out=$(mktemp)
python3 -c "
import sys
def rowbinary_str(value):
    out, n = bytearray(), len(value)
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            break
    return bytes(out) + value
sys.stdout.buffer.write(b'2\n')
for text in (b'hello', b'world'):
    sys.stdout.buffer.write(rowbinary_str(b'stub') + rowbinary_str(text))
" | docker exec -i "$CONTAINER" bridge-client embed --socket "$SOCKET" > "$udf_out" \
    || fail "bridge-client failed against the daemon socket"

python3 -c "
import json, struct, sys
raw = open('$udf_out', 'rb').read()
http = json.loads(sys.stdin.read())['data']
pos, rows = 0, []
for _ in range(2):
    length, shift = 0, 0
    while True:                       # RowBinary varuint array length
        byte = raw[pos]; pos += 1
        length |= (byte & 0x7F) << shift; shift += 7
        if not byte & 0x80:
            break
    assert length == $STUB_DIM, f'row has {length} dims, expected $STUB_DIM'
    rows.append(struct.unpack_from('<%df' % length, raw, pos))
    pos += 4 * length
assert pos == len(raw), f'{len(raw) - pos} trailing bytes in the UDF reply'
for i, (udf, item) in enumerate(zip(rows, http)):
    assert list(udf) == [struct.unpack('<f', struct.pack('<f', v))[0] for v in item['embedding']], \
        f'row {i}: the UDF channel and the HTTP channel disagree'
" <<<"$http_json" || fail "the UDF channel output is malformed or differs from HTTP"
rm -f "$udf_out"
ok "bridge-client over the socket: 2 x $STUB_DIM floats, identical to HTTP"

# --- observability -----------------------------------------------------------
metrics=$(curl -fsS "$base/metrics")
requests=$(awk '/^model_bridge_embed_requests_total /{print $2}' <<<"$metrics")
hits=$(awk '/^model_bridge_cache_hits_total /{print $2}' <<<"$metrics")
[ "${requests:-0}" -ge 2 ] || fail "/metrics counted $requests embed requests, expected at least 2"
[ "${hits:-0}" -ge 1 ] || fail "/metrics counted no cache hits after repeating the same texts"
ok "/metrics counted $requests embed requests and $hits cache hits"

# --- shutdown ----------------------------------------------------------------
# STOPSIGNAL must reach the daemon's handler. Without it, PID 1 ignores the
# default SIGTERM and every stop costs the full kill timeout.
started=$(date +%s)
docker stop --timeout 15 "$CONTAINER" >/dev/null
elapsed=$(( $(date +%s) - started ))
[ "$elapsed" -lt 15 ] || fail "container ignored the stop signal and was killed after ${elapsed}s"
[ "$(docker inspect --format '{{.State.ExitCode}}' "$CONTAINER")" = 0 ] \
    || fail "container exited non-zero on stop"
docker logs "$CONTAINER" 2>&1 | grep -q stopped || fail "no graceful shutdown in the logs"
ok "graceful shutdown in ${elapsed}s with exit code 0"

echo "smoke: all checks passed"
