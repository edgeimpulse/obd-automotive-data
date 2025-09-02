#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CSV -> Edge Impulse inference -> (optional) Blues Notecard uplink

- Reads time-series CSV with headers matching your model axes
  (e.g., "RPM [RPM], PEDAL INPUT [%], MAF [g/s], NOx [ppm]").
- Windows & flattens samples in time-major order for Edge Impulse .eim models.
- Runs on-device inference via edge_impulse_linux.
- Optionally sends events to Blues Notehub via Notecard:
    * USB/Serial (CDC)  OR  Grove/I2C.

Flood control:
  --alert-class airleak_nox   only send when top class == this label
  --alert-threshold 0.85      minimum probability to send
  --alert-cooldown 60         seconds between alerts for the SAME state
  --decimate 10               send every Nth window (0=disabled)
  --sync-every 5              call hub.sync every N notes (0=never)
  --mode periodic --outbound-mins 5  device-side rate control

Examples:

# 1) Inference only (2 Hz, 1 s window, 0.5 s step)
python3 csv_to_ei_notecard_fixed.py \
  --model ./obd_fault.eim \
  --csv ./obd_run_unhealthy.csv \
  --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
  --time-col "time (ms.)" \
  --hz 2 --window-ms 1000 --step-ms 500

# 2) Serial Notecard (USB), periodic outbound, batch sync every 10 notes
python3 csv_to_ei_notecard_fixed.py \
  --model ./obd_fault.eim \
  --csv ./obd_run_unhealthy.csv \
  --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
  --time-col "time (ms.)" \
  --hz 2 --window-ms 1000 --step-ms 500 \
  --notecard-transport serial \
  --notecard-port /dev/ttyACM0 \
  --product "com.edgeimpulse.eoin:obd_automotive_data" \
  --mode periodic --outbound-mins 5 \
  --include-scores --include-location \
  --sync-every 10

# 3) Grove/I2C Notecard on Pi (bus 1), same behavior as above
python3 csv_to_ei_notecard_fixed.py \
  --model ./obd_fault.eim \
  --csv ./obd_run_unhealthy.csv \
  --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
  --time-col "time (ms.)" \
  --hz 2 --window-ms 1000 --step-ms 500 \
  --notecard-transport i2c \
  --i2c-bus /dev/i2c-1 --i2c-addr 0x17 \
  --product "com.edgeimpulse.eoin:obd_automotive_data" \
  --mode periodic --outbound-mins 5 \
  --include-scores --include-location \
  --sync-every 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
from edge_impulse_linux.runner import ImpulseRunner

# Optional deps (loaded lazily when needed)
try:
    import serial  # pyserial for USB/Serial Notecard
except Exception:
    serial = None  # type: ignore


# ------------------------------ CSV helpers ------------------------------- #
def _sniff_dialect(fp) -> csv.Dialect:
    sample = fp.read(4096)
    fp.seek(0)
    try:
        return csv.Sniffer().sniff(sample)
    except Exception:
        return csv.excel


def read_csv_rows(csv_path: str, time_col: str, axes: Sequence[str], hz: float) -> List[Tuple[int, List[float]]]:
    """Return rows as (t_ms, [axis0..]). Falls back to index-based time; blanks -> 0.0."""
    rows: List[Tuple[int, List[float]]] = []
    with open(csv_path, "r", newline="") as f:
        dialect = _sniff_dialect(f)
        reader = csv.DictReader(f, dialect=dialect)
        dt_ms = 1000.0 / max(hz, 1e-6)
        for i, row in enumerate(reader):
            try:
                t_ms = int(float(row.get(time_col, i * dt_ms)))
            except Exception:
                t_ms = int(i * dt_ms)
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


def make_windows(rows: Sequence[Tuple[int, Sequence[float]]], hz: float, window_ms: int, step_ms: int, n_axes: int):
    """Return (X, window_start_times_ms, frame_size, samples_per_window)."""
    dt_ms = int(round(1000.0 / max(hz, 1e-6)))
    n_per_win = max(1, int(round(window_ms / max(dt_ms, 1))))
    n_per_step = max(1, int(round(step_ms / max(dt_ms, 1))))

    times = [t for t, _ in rows]
    vals = [v for _, v in rows]

    X: List[List[float]] = []
    Ts: List[int] = []
    for start in range(0, len(vals) - n_per_win + 1, n_per_step):
        seg = vals[start:start + n_per_win]
        # time-major flatten: [ax0@t0, ax1@t0, ..., axN@t0, ax0@t1, ...]
        X.append(np.array(seg, dtype=float).reshape(-1).tolist())
        Ts.append(times[start])

    frame_size = n_per_win * n_axes
    return np.array(X, dtype=float), Ts, frame_size, n_per_win


# ---------------------------- Notecard wrappers --------------------------- #
class NoteCardBase:
    def add_note(self, file: str, body: Dict, sync: bool = False) -> Dict: ...
    def get_location(self) -> Dict: ...
    def close(self): ...


class NoteCardSerial(NoteCardBase):
    def __init__(self, port: str, product: str, mode: str = "continuous",
                 outbound_mins: int = 0, baud: int = 115200, timeout: float = 2.0):
        if serial is None:
            raise RuntimeError("pyserial not installed. Try: pip install pyserial")
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        cfg = {"req": "hub.set", "product": product, "mode": mode}
        if mode == "periodic" and outbound_mins > 0:
            cfg["outbound"] = outbound_mins
        self._txrx(cfg)
        self._txrx({"req": "card.attn", "mode": "server"})  # optional

    def _txrx(self, req: Dict, timeout_s: float = 3.0) -> Dict:
        self.ser.write((json.dumps(req) + "\n").encode())
        self.ser.flush()
        t0 = time.time()
        buf = b""
        while time.time() - t0 < timeout_s:
            chunk = self.ser.read(256)
            if chunk:
                buf += chunk
                if b"\n" in buf:
                    break
        try:
            return json.loads(buf.decode(errors="ignore").strip() or "{}")
        except Exception:
            return {}

    def add_note(self, file: str, body: Dict, sync: bool = False) -> Dict:
        resp = self._txrx({"req": "note.add", "file": file, "body": body})
        if sync:
            self._txrx({"req": "hub.sync"})
        return resp

    def get_location(self) -> Dict:
        return self._txrx({"req": "card.location"}) or {}

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class NoteCardI2C(NoteCardBase):
    def __init__(self, product: str, mode: str = "periodic", outbound_mins: int = 5,
                 bus: str = "/dev/i2c-1", i2c_addr: int = 0x17, timeout_ms: int = 1000):
        try:
            import notecard
            from periphery import I2C
        except Exception as e:
            raise RuntimeError("I2C mode requires: pip install notecard periphery") from e
        self._notecard = notecard
        self._i2c = I2C(bus)
        self._card = notecard.OpenI2C(self._i2c, 0, i2c_addr, timeout_ms)
        self._tx({"req": "hub.set", "product": product, "mode": mode, "outbound": outbound_mins})

    def _tx(self, req: Dict) -> Dict:
        try:
            return self._notecard.Transaction(self._card, req) or {}
        except Exception:
            return {}

    def add_note(self, file: str, body: Dict, sync: bool = False) -> Dict:
        resp = self._tx({"req": "note.add", "file": file, "body": body})
        if sync:
            self._tx({"req": "hub.sync"})
        return resp

    def get_location(self) -> Dict:
        return self._tx({"req": "card.location"}) or {}

    def close(self):
        try:
            self._i2c.close()
        except Exception:
            pass


# ---------------------------------- Main ---------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="CSV -> Edge Impulse inference -> (optional) Blues Notecard uplink")

    ap.add_argument("--model", required=True, help="Path to Edge Impulse .eim model")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--axes", required=True, help="Comma-separated axis names (order must match model)")
    ap.add_argument("--time-col", "--time_col", dest="time_col", default="time (ms.)", help="Time column (ms)")

    ap.add_argument("--hz", type=float, default=2.0, help="Sampling rate assumed for CSV")
    ap.add_argument("--window-ms", "--window_ms", dest="window_ms", type=int, default=1000, help="Window length (ms)")
    ap.add_argument("--step-ms", "--step_ms", dest="step_ms", type=int, default=1000, help="Step/stride (ms)")
    ap.add_argument("--expected-size", "--expected_size", dest="expected_size", type=int, default=0,
                    help="Force feature length (pad/truncate) if set")
    ap.add_argument("--print-topk", "--print_topk", dest="print_topk", type=int, default=1,
                    help="How many top classes to print per window")
    ap.add_argument("--label-col", "--label_col", dest="label_col", default="", help="Optional ground truth label column")

    # Notecard (optional)
    ap.add_argument("--notecard-transport", choices=["serial", "i2c"], default="serial",
                    help="Transport to Notecard when sending (default serial).")
    ap.add_argument("--notecard-port", "--notecard_port", dest="notecard_port", default="",
                    help="Serial CDC port (e.g., /dev/ttyACM0) if using --notecard-transport serial")
    ap.add_argument("--i2c-bus", "--i2c_bus", dest="i2c_bus", default="/dev/i2c-1",
                    help="I2C bus path for --notecard-transport i2c (default /dev/i2c-1)")
    ap.add_argument("--i2c-addr", "--i2c_addr", dest="i2c_addr", type=lambda x: int(x, 0), default=0x17,
                    help="I2C address (hex ok, default 0x17)")
    ap.add_argument("--product", default="", help="Notehub ProductUID (org.company:project)")
    ap.add_argument("--mode", choices=["continuous", "periodic"], default="continuous", help="Notecard hub mode")
    ap.add_argument("--outbound-mins", "--outbound_mins", dest="outbound_mins", type=int, default=0,
                    help="If --mode periodic, upload every M minutes")
    ap.add_argument("--file", default="obd.qo", help="Note file name")
    ap.add_argument("--include-location", action="store_true", help="Attach best-effort location to each note")
    ap.add_argument("--include-scores", action="store_true", help="Attach full class score map to each note")
    ap.add_argument("--dry-run", action="store_true", help="Print would-be notes instead of sending")

    # Flood-control
    ap.add_argument("--alert-class", default="", help="Only send when top class == this label (else gate)")
    ap.add_argument("--alert-threshold", type=float, default=0.8, help="Min prob to send an alert")
    ap.add_argument("--alert-cooldown", type=float, default=60.0, help="Min seconds between alerts for SAME state")
    ap.add_argument("--decimate", type=int, default=0, help="Send every Nth window (0=disabled)")
    ap.add_argument("--sync-every", "--sync_every", dest="sync_every", type=int, default=0,
                    help="Call hub.sync every N notes (0=never)")

    args = ap.parse_args()

    axes: List[str] = [a.strip() for a in args.axes.split(",") if a.strip()]
    rows = read_csv_rows(args.csv, args.time_col, axes, args.hz)
    if not rows:
        print("ERROR: CSV contained no rows. Check delimiter and column names.", file=sys.stderr)
        return 2

    X, Ts, frame, n_per_win = make_windows(rows, args.hz, args.window_ms, args.step_ms, len(axes))
    print(f"[INFO] {len(rows)} samples @ {args.hz:.3f} Hz; window {args.window_ms} ms ({n_per_win} samples), step {args.step_ms} ms.")
    print(f"[INFO] Prepared {len(X)} windows; frame_size/window = {frame} (axes={len(axes)}).")

    # Load model
    runner = ImpulseRunner(args.model)
    info = runner.init()
    labels = info.get("classifiers", [{}])[0].get("labels", [])
    print("[INFO] Model labels:", labels or "(none reported)")

    # Resize frame if needed
    target_size = args.expected_size or frame
    if target_size != frame:
        print(f"[WARN] Computed frame={frame} but expected_size={target_size}; padding/truncating.")
        def fix_len(a):
            a = list(a)
            if len(a) < target_size: return a + [0.0] * (target_size - len(a))
            return a[:target_size]
        X = np.array([fix_len(x) for x in X], dtype=float)

    # Build label lookup if provided
    label_by_time: Dict[int, str] = {}
    if args.label_col:
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

    # Notecard selection
    nc: Optional[NoteCardBase] = None
    if (args.notecard_transport == "serial" and args.notecard_port) or (args.notecard_transport == "i2c"):
        if not args.product:
            print("ERROR: --product is required when using Notecard uplink", file=sys.stderr)
            return 2
        try:
            if args.notecard_transport == "serial":
                nc = NoteCardSerial(args.notecard_port, args.product, mode=args.mode, outbound_mins=args.outbound_mins)
                print(f"[INFO] Notecard(serial) on {args.notecard_port}, product={args.product}, mode={args.mode}, outbound={args.outbound_mins}m")
            else:
                nc = NoteCardI2C(product=args.product, mode=args.mode, outbound_mins=args.outbound_mins,
                                 bus=args.i2c_bus, i2c_addr=args.i2c_addr)
                print(f"[INFO] Notecard(I2C) on {args.i2c_bus}@0x{args.i2c_addr:02X}, product={args.product}, mode={args.mode}, outbound={args.outbound_mins}m")
        except Exception as e:
            print(f"[ERROR] Notecard init failed: {e}", file=sys.stderr)
            nc = None

    # Gating state
    note_count = 0
    last_sent_by_state: Dict[str, float] = {}
    prev_state: str = ""

    try:
        for tms, feats in zip(Ts, X):
            feats_list = feats.tolist() if isinstance(feats, np.ndarray) else list(feats)
            res = runner.classify(feats_list)
            result = res.get("result", {})
            scores = result.get("classification", {})

            if scores:
                top_label, top_prob = max(scores.items(), key=lambda kv: kv[1])
                print(f"[{tms/1000.0:7.3f}s] {top_label}={top_prob:.3f}")
                send = False
                now = time.time()

                if args.alert_class:
                    if top_label == args.alert_class and top_prob >= args.alert_threshold:
                        last = last_sent_by_state.get(top_label, 0.0)
                        if (now - last) >= args.alert_cooldown and prev_state != top_label:
                            send = True
                            last_sent_by_state[top_label] = now
                            prev_state = top_label
                    else:
                        if prev_state != "" and top_label != prev_state:
                            prev_state = top_label
                else:
                    if args.decimate > 1:
                        if ((note_count + 1) % args.decimate) == 0:
                            send = True
                    else:
                        send = True

                if (nc is not None) and send:
                    body: Dict = {
                        "ts": int(now),
                        "window_start_ms": int(tms),
                        "pred": top_label,
                        "prob": float(top_prob),
                    }
                    if args.include_scores:
                        body["scores"] = {k: float(v) for k, v in scores.items()}
                    if args.include_location:
                        try:
                            loc = nc.get_location()
                            if loc:
                                body["where"] = loc
                        except Exception:
                            pass
                    if tms in label_by_time:
                        body["label"] = label_by_time[tms]

                    if args.dry_run:
                        print("[DRY-RUN note.add]", json.dumps(body))
                    else:
                        nc.add_note(args.file, body, sync=False)
                        note_count += 1
                        if args.sync_every and (note_count % args.sync_every) == 0:
                            nc.add_note(args.file, {"meta": "sync-marker", "count": note_count}, sync=True)

            else:
                # regression/anomaly or unknown schema
                print(f"[{tms/1000.0:7.3f}s] result={result}")
                if nc is not None and (args.decimate == 0 or ((note_count + 1) % args.decimate) == 0):
                    body = {"ts": int(time.time()), "window_start_ms": int(tms), "result": result}
                    if args.dry_run:
                        print("[DRY-RUN note.add]", json.dumps(body))
                    else:
                        nc.add_note(args.file, body, sync=False)
                        note_count += 1
                        if args.sync_every and (note_count % args.sync_every) == 0:
                            nc.add_note(args.file, {"meta": "sync-marker", "count": note_count}, sync=True)

    finally:
        try:
            runner.stop()
        except Exception:
            pass
        if nc is not None:
            nc.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

