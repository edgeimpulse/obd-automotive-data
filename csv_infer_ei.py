#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CSV → Edge Impulse inference (generic)

- Reads a time-series CSV with a time column and one column per axis
  (e.g., "RPM [RPM], PEDAL INPUT [%], MAF [g/s], NOx [ppm]").
- Slices into sliding windows that match your model’s training settings.
- Runs inference using an Edge Impulse .eim model (via edge_impulse_linux).
- Prints top-1 (or top-k) per window.
- (Optional) Writes results to a JSONL file (--out-jsonl).

Usage:

  # Basic playback at 2 Hz with 1 s window / 1 s step
  python3 csv_infer_ei.py \
    --model ./obd_fault.eim \
    --csv ./obd_healthy.csv \
    --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
    --hz 2 --window-ms 1000 --step-ms 1000

  # Top-3 and also store results to predictions.jsonl
  python3 csv_infer_ei.py \
    --model ./model.eim \
    --csv ./data.csv \
    --axes "ax,ay,az" --hz 50 --window-ms 1000 --step-ms 200 \
    --print-topk 3 --out-jsonl predictions.jsonl

Dependencies:
  python3 -m pip install edge_impulse_linux numpy
"""

from __future__ import annotations
import argparse, csv, json, sys
from typing import List, Sequence, Tuple
import numpy as np
from edge_impulse_linux.runner import ImpulseRunner


# ------------------------------ CSV helpers ------------------------------- #

def read_rows(path: str, time_col: str, axes: Sequence[str], hz: float) -> List[Tuple[int, List[float]]]:
    """Return [(t_ms, [axis0, axis1, ...]), ...], sorted by time."""
    rows: List[Tuple[int, List[float]]] = []
    dt = 1000.0 / max(hz, 1e-6)
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            # time
            try:
                t_ms = int(float(row.get(time_col, i * dt)))
            except Exception:
                t_ms = int(i * dt)
            # axes
            vals: List[float] = []
            for ax in axes:
                try:
                    v = row.get(ax, "")
                    vals.append(float(v) if v not in ("", None) else 0.0)
                except Exception:
                    vals.append(0.0)
            rows.append((t_ms, vals))
    rows.sort(key=lambda x: x[0])
    return rows


def make_windows(
    rows: Sequence[Tuple[int, Sequence[float]]],
    hz: float,
    window_ms: int,
    step_ms: int,
    n_axes: int,
) -> Tuple[List[List[float]], List[int], int, int]:
    """
    Return (X, Ts, frame_size, samples_per_window)
      X  = [flat_feature_vector_per_window]
      Ts = [window_start_time_ms]
      frame_size = n_axes * samples_per_window
    """
    dt = int(round(1000.0 / max(hz, 1e-6)))
    n_per_win  = max(1, int(round(window_ms / max(dt, 1))))
    n_per_step = max(1, int(round(step_ms  / max(dt, 1))))

    times = [t for t, _ in rows]
    vals  = [v for _, v in rows]

    X: List[List[float]] = []
    Ts: List[int] = []

    for start in range(0, len(vals) - n_per_win + 1, n_per_step):
        seg = vals[start : start + n_per_win]
        X.append(np.array(seg, dtype=float).reshape(-1).tolist())
        Ts.append(times[start])

    frame_size = n_per_win * n_axes
    return X, Ts, frame_size, n_per_win


# ---------------------------------- Main ---------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser("CSV → Edge Impulse inference (generic)")
    ap.add_argument("--model", required=True, help="Path to Edge Impulse .eim model")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--axes", required=True,
                    help="Comma-separated axis names in CSV (order must match model training)")
    ap.add_argument("--time-col", default="time (ms.)", help="Time column in ms (default: 'time (ms.)')")

    ap.add_argument("--hz", type=float, default=2.0, help="Sampling rate assumed for CSV (Hz)")
    ap.add_argument("--window-ms", type=int, default=1000, help="Window length (ms)")
    ap.add_argument("--step-ms", type=int, default=1000, help="Step/stride (ms)")

    ap.add_argument("--expected-size", type=int, default=0,
                    help="Pad/truncate feature vectors to this length if set (advanced)")
    ap.add_argument("--print-topk", type=int, default=1, help="How many top classes to print per window")
    ap.add_argument("--out-jsonl", default="", help="If set, append JSON lines with results to this file")
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    rows = read_rows(args.csv, args.time_col, axes, args.hz)
    if not rows:
        print("ERROR: CSV contained no rows or columns not found.", file=sys.stderr)
        return 2

    X, Ts, frame, n_per_win = make_windows(rows, args.hz, args.window_ms, args.step_ms, len(axes))
    print(f"[INFO] {len(rows)} samples @ {args.hz:.3f} Hz; window {args.window_ms} ms "
          f"({n_per_win} samples), step {args.step_ms} ms. Windows={len(X)}, frame={frame}")

    # Init model
    runner = ImpulseRunner(args.model)
    info = runner.init()
    labels = info.get("classifiers", [{}])[0].get("labels", [])
    if labels:
        print("[INFO] labels:", labels)
    else:
        print("[WARN] model has no classifier labels (regression/anomaly or wrong .eim)")

    # Adjust feature length if requested
    target = args.expected_size or frame
    if target != frame:
        print(f"[WARN] frame={frame} != expected_size={target} → pad/truncate")
        def fix_len(z):
            z = list(z)
            return z + [0.0] * (target - len(z)) if len(z) < target else z[:target]
        X = [fix_len(z) for z in X]

    # Optional JSONL logging
    out = open(args.out_jsonl, "a") if args.out_jsonl else None

    try:
        for t_ms, feats in zip(Ts, X):
            res = runner.classify(list(feats))
            result = res.get("result", {})
            scores = result.get("classification", {})

            if scores:
                top_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: args.print_topk]
                msg = " ".join([f"{k}={v:.3f}" for k, v in top_items])
                print(f"[{t_ms/1000.0:7.3f}s] {msg}")
                if out:
                    top_label, top_prob = top_items[0]
                    out.write(json.dumps({
                        "t_ms": int(t_ms),
                        "top": top_label,
                        "prob": float(top_prob),
                        "scores": {k: float(v) for k, v in scores.items()},
                    }) + "\n")
            else:
                # Regression/anomaly or unknown schema
                print(f"[{t_ms/1000.0:7.3f}s] result={result}")
                if out:
                    out.write(json.dumps({"t_ms": int(t_ms), "result": result}) + "\n")
    finally:
        try:
            runner.stop()
        except Exception:
            pass
        if out:
            out.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

