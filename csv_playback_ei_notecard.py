#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CSV > Edge Impulse inference → (optional) Blues Notecard uplink

- Reads time-series CSV with columns matching your model axes
  (e.g., "RPM [RPM], PEDAL INPUT [%], MAF [g/s], NOx [ppm]").
- Slices into sliding windows (hz / window / step).
- Runs inference using an Edge Impulse .eim model via edge_impulse_linux.
- Prints top classification per window.
- Optionally posts each inference to a Blues Notecard over its USB CDC port
  using simple line-delimited JSON (no extra deps).

Usage (examples):

  # Plain playback at 2 Hz with 1s window/step
  python3 csv_playback_ei_notecard.py \
    --model ./obd_fault.eim \
    --csv ./obd_run_healthy.csv \
    --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
    --hz 2 --window-ms 1000 --step-ms 1000

  # Playback + Notecard uplink (replace product UID & port)
  python3 csv_playback_ei_notecard.py \
    --model ./obd_fault.eim \
    --csv ./obd_run_healthy.csv \
    --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
    --hz 2 --window-ms 1000 --step-ms 1000 \
    --notecard-port /dev/ttyACM0 \
    --product "org.yourco:your-product" \
    --sync

Requirements:
  pip install edge_impulse_linux numpy pandas pyserial
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import serial
from edge_impulse_linux.runner import ImpulseRunner


# ------------------------------ CSV helpers ------------------------------- #

def _sniff_dialect(fp) -> csv.Dialect:
    """Try to sniff CSV dialect, default to excel if sniff fails."""
    sample = fp.read(4096)
    fp.seek(0)
    try:
        return csv.Sniffer().sniff(sample)
    except Exception:
        return csv.excel


def read_csv_rows(
    csv_path: str,
    time_col: str,
    axes: Sequence[str],
    hz: float,
) -> List[Tuple[int, List[float]]]:
    """
    Read CSV rows as (t_ms, [axis0, axis1, ...]).
    Falls back to index-based time if time column is missing/unparseable.
    Missing values are treated as 0.0 to keep the frame numeric.
    """
    rows: List[Tuple[int, List[float]]] = []
    with open(csv_path, "r", newline="") as f:
        dialect = _sniff_dialect(f)
        reader = csv.DictReader(f, dialect=dialect)
        dt_ms = 1000.0 / max(hz, 1e-6)
        for i, row in enumerate(reader):
            # time
            try:
                t_ms = int(float(row.get(time_col, i * dt_ms)))
            except Exception:
                t_ms = int(i * dt_ms)

            # axes
            values: List[float] = []
            for ax in axes:
                try:
                    v = row.get(ax, "")
                    values.append(float(v) if v not in ("", None) else 0.0)
                except Exception:
                    values.append(0.0)

            rows.append((t_ms, values))

    rows.sort(key=lambda x: x[0])
    return rows


def make_windows(
    rows: Sequence[Tuple[int, Sequence[float]]],
    hz: float,
    window_ms: int,
    step_ms: int,
    n_axes: int,
) -> Tuple[np.ndarray, List[int], int, int]:
    """
    Slice rows into sliding windows and flatten time-major:
      [ax0@t0, ax1@t0, ..., axN@t0, ax0@t1, ...]
    Returns (X, window_start_times_ms, frame_size, samples_per_window)
    """
    dt_ms = int(round(1000.0 / max(hz, 1e-6)))
    n_per_win = max(1, int(round(window_ms / max(dt_ms, 1))))
    n_per_step = max(1, int(round(step_ms / max(dt_ms, 1))))

    times = [t for t, _ in rows]
    vals = [v for _, v in rows]

    X: List[List[float]] = []
    Ts: List[int] = []

    for start in range(0, len(vals) - n_per_win + 1, n_per_step):
        seg = vals[start : start + n_per_win]
        X.append(np.array(seg, dtype=float).reshape(-1).tolist())
        Ts.append(times[start])

    frame_size = n_per_win * n_axes
    return np.array(X, dtype=float), Ts, frame_size, n_per_win


# ---------------------------- Notecard helpers ---------------------------- #

def nc_open(port: str, baud: int = 115200) -> serial.Serial:
    """Open the Notecard's CDC ACM port."""
    return serial.Serial(port, baudrate=baud, timeout=2)


def nc_txrx(ser: serial.Serial, req: Dict) -> Dict:
    """Send a JSON req and read one JSON line back (best-effort)."""
    ser.write((json.dumps(req) + "\n").encode())
    ser.flush()
    t0 = time.time()
    buf = b""
    while time.time() - t0 < 3.0:
        chunk = ser.read(256)
        if chunk:
            buf += chunk
            if b"\n" in buf:
                break
    try:
        return json.loads(buf.decode(errors="ignore").strip() or "{}")
    except Exception:
        return {}


def nc_init(ser: serial.Serial, product: str, continuous: bool = True) -> None:
    nc_txrx(ser, {
        "req": "hub.set",
        "product": product,
        "mode": "continuous" if continuous else "periodic"
    })
    # Optional: wake on server events
    nc_txrx(ser, {"req": "card.attn", "mode": "server"})


# ---------------------------------- Main ---------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="CSV → Edge Impulse inference → (optional) Blues Notecard uplink"
    )
    ap.add_argument("--model", required=True, help="Path to Edge Impulse .eim model")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument(
        "--axes",
        default="RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]",
        help="Comma-separated axis names in CSV (order must match model)",
    )
    ap.add_argument("--time-col", default="time (ms.)", help="Time column (ms)")
    ap.add_argument("--label-col", default="", help="Optional ground-truth label column to include in payloads")

    ap.add_argument("--hz", type=float, default=2.0, help="Sampling rate assumed for CSV")
    ap.add_argument("--window-ms", type=int, default=1000, help="Window length (ms)")
    ap.add_argument("--step-ms", type=int, default=1000, help="Step/stride (ms)")
    ap.add_argument(
        "--expected-size",
        type=int,
        default=0,
        help="Force feature vector length (pad/truncate) if set",
    )
    ap.add_argument(
        "--print-topk",
        type=int,
        default=1,
        help="How many top classes to print per window",
    )

    # Notecard (optional)
    ap.add_argument(
        "--notecard-port",
        default="",
        help="Notecard CDC port (e.g., /dev/ttyACM0 or /dev/serial/by-id/...)",
    )
    ap.add_argument(
        "--product",
        default="",
        help="Notehub ProductUID (org.company:project) — required if --notecard-port set",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="Call hub.sync after each note (immediate upload; higher data use)",
    )

    args = ap.parse_args()

    axes: List[str] = [a.strip() for a in args.axes.split(",") if a.strip()]
    rows = read_csv_rows(args.csv, args.time_col, axes, args.hz)
    if not rows:
        print("ERROR: CSV contained no rows. Check delimiter and column names.", file=sys.stderr)
        return 2

    X, Ts, frame, n_per_win = make_windows(rows, args.hz, args.window_ms, args.step_ms, len(axes))
    print(f"[INFO] {len(rows)} samples @ {args.hz:.3f} Hz; window {args.window-ms if hasattr(args,'window-ms') else args.window_ms} ms "
          f"({n_per_win} samples), step {args.step_ms} ms.")
    print(f"[INFO] Prepared {len(X)} windows; frame_size/window = {frame} (axes={len(axes)}).")

    # Init model
    runner = ImpulseRunner(args.model)
    info = runner.init()
    labels = info.get("classifiers", [{}])[0].get("labels", [])
    if labels:
        print("[INFO] Model labels:", labels)
    else:
        print("[WARN] Model reports no classifier labels (regression/anomaly or wrong .eim).")

    # Adjust frame size if user asked
    target_size = args.expected_size or frame
    if target_size != frame:
        print(f"[WARN] Computed frame={frame} but expected_size={target_size}; padding/truncating.")
        def fix_len(a: Sequence[float]) -> List[float]:
            a = list(a)
            if len(a) < target_size:
                return a + [0.0] * (target_size - len(a))
            return a[:target_size]
        X = np.array([fix_len(x) for x in X], dtype=float)

    # Notecard?
    nc: serial.Serial | None = None
    if args.notecard_port:
        if not args.product:
            print("ERROR: --product is required when using --notecard-port", file=sys.stderr)
            return 2
        try:
            nc = nc_open(args.notecard_port)
            nc_init(nc, args.product, continuous=True)
            print(f"[INFO] Notecard ready on {args.notecard_port}, product={args.product}")
        except Exception as e:
            print(f"[ERROR] Notecard init failed: {e}", file=sys.stderr)
            nc = None

    # (Optional) map window start index back to a label if label_col present
    label_by_time: Dict[int, str] = {}
    if args.label_col:
        # Build quick lookup of first non-empty label at each timestamp
        try:
            with open(args.csv, "r", newline="") as f:
                dialect = _sniff_dialect(f)
                reader = csv.DictReader(f, dialect=dialect)
                dt_ms = 1000.0 / max(args.hz, 1e-6)
                for i, row in enumerate(reader):
                    try:
                        t_ms = int(float(row.get(args.time_col, i * dt_ms)))
                    except Exception:
                        t_ms = int(i * dt_ms)
                    lbl = (row.get(args.label_col) or "").strip()
                    if lbl and t_ms not in label_by_time:
                        label_by_time[t_ms] = lbl
        except Exception:
            pass

    # Classify
    try:
        for tms, feats in zip(Ts, X):
            feats_list = feats.tolist() if isinstance(feats, np.ndarray) else list(feats)
            res = runner.classify(feats_list)
            result = res.get("result", {})
            scores = result.get("classification", {})

            if scores:
                top_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: args.print_topk]
                msg = " ".join([f"{k}={v:.3f}" for k, v in top_items])
                print(f"[{tms/1000.0:7.3f}s] {msg}")
                top_label, top_prob = top_items[0]

                if nc is not None:
                    body = {
                        "ts": int(time.time()),
                        "pred": top_label,
                        "prob": float(top_prob),
                    }
                    # Include ground truth if available at this timestamp (best-effort)
                    if tms in label_by_time:
                        body["label"] = label_by_time[tms]
                    nc_txrx(nc, {"req": "note.add", "file": "obd.qo", "body": body})
                    if args.sync:
                        nc_txrx(nc, {"req": "hub.sync"})
            else:
                # Regression/anomaly or unknown schema
                print(f"[{tms/1000.0:7.3f}s] result={result}")
                if nc is not None:
                    body = {"ts": int(time.time()), "result": result}
                    if tms in label_by_time:
                        body["label"] = label_by_time[tms]
                    nc_txrx(nc, {"req": "note.add", "file": "obd.qo", "body": body})
                    if args.sync:
                        nc_txrx(nc, {"req": "hub.sync"})
    finally:
        try:
            runner.stop()
        except Exception:
            pass
        if nc is not None:
            try:
                nc.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

