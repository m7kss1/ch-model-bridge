#!/usr/bin/env python3
"""Trains a toy fraud-scoring model on synthetic transactions and exports it
to ONNX for `modelEvaluate`.

The `tabular` model kind expects a graph with one float-tensor input of shape
[N, n_features] and one float output of shape [N] or [N, 1]. A regressor
trained on 0/1 labels fits that contract directly and its score approximates
the fraud probability; an sklearn classifier would export two outputs (label
and probabilities) and be rejected at load time.

Feature order is part of the model contract: amount, hour_of_day,
is_new_device, merchant_risk. SQL callers must pass features in this order,
cast to Array(Float32): modelEvaluate('fraud-demo', [amount, hour_of_day,
is_new_device, merchant_risk]::Array(Float32)).
"""

import json
import pathlib

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "fraud-demo"
EXPECTED = ROOT / "tmp" / "fraud-expected.json"

rng = np.random.default_rng(42)
n = 40_000

amount = np.minimum(rng.lognormal(mean=5.5, sigma=1.2, size=n), 10_000.0)
hour = rng.integers(0, 24, size=n).astype(np.float64)
is_new_device = (rng.random(n) < 0.2).astype(np.float64)
merchant_risk = rng.random(n) ** 2

# Ground truth is interactional on purpose: a large amount alone is not fraud,
# a large amount at night is. The demo then shows the model learned the
# interaction, not a single-feature threshold.
p = np.full(n, 0.02)
night = hour < 6
p += 0.75 * ((amount > 3000) & night)
p += 0.60 * ((is_new_device == 1) & (merchant_risk > 0.7) & (amount > 1000))
p = np.clip(p, 0.0, 0.97)
y = (rng.random(n) < p).astype(np.float64)

X = np.column_stack([amount, hour, is_new_device, merchant_risk])
model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
model.fit(X, y)

print(f"trained on {n} rows, fraud rate {y.mean():.3f}")
print(f"mean score on fraud rows:  {model.predict(X[y == 1]).mean():.3f}")
print(f"mean score on clean rows:  {model.predict(X[y == 0]).mean():.3f}")

onnx_model = convert_sklearn(
    model, initial_types=[("features", FloatTensorType([None, X.shape[1]]))]
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "model.onnx").write_bytes(onnx_model.SerializeToString())
print(f"wrote {MODEL_DIR / 'model.onnx'} ({len(onnx_model.SerializeToString())} bytes)")

# Reference outputs for verifying the daemon. Inputs are cast to float32
# first so sklearn sees the same values ONNX Runtime will; tree thresholds
# are still float64 here vs float32 in ONNX, hence a tolerance, not equality.
named = [
    ("night + large amount", [4800.0, 2.0, 0.0, 0.1]),
    ("same amount, daytime", [4800.0, 14.0, 0.0, 0.1]),
    ("night, small amount", [120.0, 2.0, 0.0, 0.1]),
    ("new device + risky merchant", [1500.0, 15.0, 1.0, 0.9]),
    ("known device, risky merchant", [1500.0, 15.0, 0.0, 0.9]),
]
random_rows = np.column_stack(
    [
        np.minimum(rng.lognormal(5.5, 1.2, 200), 10_000.0),
        rng.integers(0, 24, 200).astype(np.float64),
        (rng.random(200) < 0.2).astype(np.float64),
        rng.random(200) ** 2,
    ]
).astype(np.float32)

named_rows = np.array([row for _, row in named], dtype=np.float32)
EXPECTED.parent.mkdir(exist_ok=True)
EXPECTED.write_text(
    json.dumps(
        {
            "named": [
                {"label": label, "features": row, "expected": score}
                for (label, row), score in zip(
                    named, model.predict(named_rows.astype(np.float64)).tolist()
                )
            ],
            "rows": random_rows.tolist(),
            "expected": model.predict(random_rows.astype(np.float64)).tolist(),
        },
        indent=1,
    )
)
print(f"wrote {EXPECTED}")

for (label, _), score in zip(named, model.predict(named_rows.astype(np.float64))):
    print(f"  {label:32s} -> {score:6.3f}")
