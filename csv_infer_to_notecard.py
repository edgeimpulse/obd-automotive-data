#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CSV → Edge Impulse inference → (optional) Blues Notecard upload (serial JSON)

Minimal: prints top-1 per window. If --notecard-port is set, posts {"pred","prob"} to Notehub.

Requirements:
  pip install edge_impulse_linux numpy pyserial
"""

from __future__ import annotations
import argparse, csv, json, sys, time
from typing import Dict, List, Sequence, Tuple
import numpy as np
import serial
from edge_impulse_linux.runner import ImpulseRunner

# ---------- Notecard (raw serial JSON) ----------
def nc_open(port: str, baud: int = 115200) -> serial.Serial:
    return serial.Serial(port, baudrate=baud, timeout=2)

def nc_txrx(ser: serial.Serial, req: Dict) -> Dict:
    ser.write((json.dumps(req)+"\n").encode()); ser.flush()
    t0=time.time(); buf=b""
    while time.time()-t0 < 3.0:
        chunk=ser.read(256)
        if chunk: buf+=chunk
        if b"\n" in buf: break
    try: return json.loads(buf.decode(errors="ignore").strip() or "{}")
    except: return {}

def nc_init(ser: serial.Serial, product: str) -> None:
    nc_txrx(ser, {"req":"hub.set","product":product,"mode":"continuous"})
    nc_txrx(ser, {"req":"card.attn","mode":"server"})

# ---------- CSV & windowing ----------
def read_rows(path: str, time_col: str, axes: Sequence[str], hz: float) -> List[Tuple[int,List[float]]]:
    rows=[]; dt=1000.0/max(hz,1e-6)
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for i,row in enumerate(r):
            try: t=int(float(row.get(time_col, i*dt)))
            except: t=int(i*dt)
            vals=[]
            for ax in axes:
                try: vals.append(float(row.get(ax,"") or 0.0))
                except: vals.append(0.0)
            rows.append((t, vals))
    rows.sort(key=lambda x:x[0]); return rows

def make_windows(rows, hz, win_ms, step_ms, n_axes):
    dt=int(round(1000.0/max(hz,1e-6)))
    nwin=max(1,int(round(win_ms/max(dt,1))))
    nstep=max(1,int(round(step_ms/max(dt,1))))
    V=[v for _,v in rows]; T=[t for t,_ in rows]
    X=[]; Ts=[]
    for start in range(0, len(V)-nwin+1, nstep):
        seg=V[start:start+nwin]
        X.append(np.array(seg,dtype=float).reshape(-1).tolist())
        Ts.append(T[start])
    return X, Ts, nwin*n_axes, nwin

# ---------- Main ----------
def main() -> int:
    ap=argparse.ArgumentParser("CSV → EI inference → (optional) Notecard")
    ap.add_argument("--model", required=True, help=".eim file")
    ap.add_argument("--csv", required=True, help="CSV with axes")
    ap.add_argument("--axes", default="RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]",
                    help="Comma-separated axis names in CSV (order must match model)")
    ap.add_argument("--time-col", default="time (ms.)")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--expected-size", type=int, default=0, help="Pad/truncate to this length if set")
    ap.add_argument("--notecard-port", default="", help="/dev/ttyACM0 or /dev/serial/by-id/...")
    ap.add_argument("--product", default="", help="Notehub ProductUID (org.company:project)")
    ap.add_argument("--sync", action="store_true", help="hub.sync after each note")
    args=ap.parse_args()

    axes=[s.strip() for s in args.axes.split(",") if s.strip()]
    rows=read_rows(args.csv, args.time_col, axes, args.hz)
    if not rows:
        print("ERROR: CSV empty or columns not found.", file=sys.stderr); return 2

    X, Ts, frame, nwin = make_windows(rows, args.hz, args.window_ms, args.step_ms, len(axes))
    print(f"[INFO] {len(rows)} samples @ {args.hz:.3f} Hz; window {args.window_ms} ms "
          f"({nwin} samples), step {args.step_ms} ms. Windows={len(X)}, frame={frame}")

    runner=ImpulseRunner(args.model); info=runner.init()
    labels=info.get("classifiers",[{}])[0].get("labels",[])
    if labels: print("[INFO] labels:", labels)
    else:     print("[WARN] model has no labels (regression/anomaly or wrong build)")

    # adjust size if requested
    target=args.expected_size or frame
    if target!=frame:
        print(f"[WARN] frame={frame} != expected_size={target} → pad/truncate")
        def fix(z):
            z=list(z)
            return z + [0.0]*(target-len(z)) if len(z)<target else z[:target]
        X=[fix(z) for z in X]

    nc=None
    if args.notecard_port:
        if not args.product:
            print("ERROR: --product is required with --notecard-port", file=sys.stderr); return 2
        try:
            nc=nc_open(args.notecard_port); nc_init(nc, args.product)
            print(f"[INFO] Notecard ready on {args.notecard_port}, product={args.product}")
        except Exception as e:
            print(f"[ERROR] Notecard init failed: {e}", file=sys.stderr); nc=None

    try:
        for t, feats in zip(Ts, X):
            res=runner.classify(list(feats))
            scores=res.get("result",{}).get("classification",{})
            if scores:
                top=max(scores,key=scores.get); p=float(scores[top])
                print(f"[{t/1000.0:6.3f}s] {top} {p:.3f}")
                if nc:
                    nc_txrx(nc, {"req":"note.add","file":"obd.qo","body":{"ts":int(time.time()),"pred":top,"prob":p}})
                    if args.sync: nc_txrx(nc, {"req":"hub.sync"})
            else:
                print(f"[{t/1000.0:6.3f}s] result={res.get('result',{})}")
                if nc:
                    nc_txrx(nc, {"req":"note.add","file":"obd.qo","body":{"ts":int(time.time()),"result":res.get("result",{})}})
                    if args.sync: nc_txrx(nc, {"req":"hub.sync"})
    finally:
        try: runner.stop()
        except: pass
        if nc:
            try: nc.close()
            except: pass

    return 0

if __name__=="__main__":
    sys.exit(main())

