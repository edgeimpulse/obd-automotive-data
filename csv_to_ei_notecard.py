#!/usr/bin/env python3
from edge_impulse_linux.runner import ImpulseRunner
import csv, argparse, numpy as np

# --- args ---
ap = argparse.ArgumentParser(description="CSV → Edge Impulse inference")
ap.add_argument("--model", required=True, help="Path to .eim")
ap.add_argument("--csv", required=True, help="Input CSV path")
ap.add_argument("--axes", required=True, help="Comma-separated header names")
ap.add_argument("--hz", type=float, default=2.0)
ap.add_argument("--window_ms", type=int, default=1000)
ap.add_argument("--step_ms", type=int, default=1000)
ap.add_argument("--time_col", default="time (ms.)")
args = ap.parse_args()

axes = [a.strip() for a in args.axes.split(",")]

def rows_from_csv(path, time_col, axes, hz):
    dt = int(round(1000.0 / max(hz, 1e-6)))
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            # timestamp (ms); fall back to index if missing
            try:
                t = int(float(row.get(time_col, i * dt)))
            except Exception:
                t = i * dt
            vals = [float(row.get(ax, 0) or 0) for ax in axes]
            out.append((t, vals))
    out.sort(key=lambda x: x[0])
    return out

rows = rows_from_csv(args.csv, args.time_col, axes, args.hz)

# --- windowing: time-major flatten ---
dt = int(round(1000.0 / max(args.hz, 1e-6)))
samps = max(1, int(round(args.window_ms / dt)))
step  = max(1, int(round(args.step_ms   / dt)))
vals  = [v for _, v in rows]
times = [t for t, _ in rows]

X, Ts = [], []
for s in range(0, len(vals) - samps + 1, step):
    seg = np.array(vals[s:s+samps], dtype=float).reshape(-1)   # [ax0@t0, ax1@t0, ..., axN@tK]
    X.append(seg.tolist())
    Ts.append(times[s])

# --- inference ---
runner = ImpulseRunner(args.model)
info = runner.init()
labels = info.get("classifiers", [{}])[0].get("labels", [])
print("[INFO] labels:", labels or "(none)")
try:
    for t, feats in zip(Ts, X):
        res = runner.classify(feats)
        scores = res.get("result", {}).get("classification", {})
        if scores:
            k, v = max(scores.items(), key=lambda kv: kv[1])
            print(f"[{t/1000:.3f}s] {k}={v:.3f}")
finally:
    runner.stop()

