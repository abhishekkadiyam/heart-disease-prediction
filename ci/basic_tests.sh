#!/usr/bin/env bash
python - <<PY
import pandas as pd, sys
try:
    df = pd.read_csv("data/sample_heart_sample.csv")
    print("Sample data read OK. shape=", df.shape)
except Exception as e:
    print("DATA READ ERROR", e); sys.exit(1)
print("Basic smoke test OK")
PY
