#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
ELM327 (USB or Bluetooth RFCOMM) → CSV for Edge Impulse

Columns:
  time (ms.), fault_label, RPM [RPM], PEDAL INPUT [%], MAF [g/s], NOx [ppm]

Notes
- Works with USB (/dev/ttyUSB*, /dev/serial/by-id/...) and BT after binding RFCOMM to /dev/rfcomm0.
- Defaults to AUTO protocol (ATSP0). Override with --proto 6 for CAN 11/500 etc.
- NOx is manufacturer-specific on many cars. If you know a PID, pass it via --nox-pid "01 5E" (example).
"""

from __future__ import annotations
import argparse, csv, re, time
from typing import List, Optional
import serial

# ---------- ELM helpers ----------
def _open(port: str, baud: int, timeout: float = 1.5) -> serial.Serial:
    s = serial.Serial(port, baudrate=baud, timeout=timeout)
    s.setDTR(True); s.setRTS(True)
    s.reset_input_buffer(); s.reset_output_buffer()
    return s

def _read_to_prompt(s: serial.Serial, timeout_s: float = 1.5) -> str:
    t0 = time.monotonic(); buf = ""
    while time.monotonic() - t0 < timeout_s:
        ch = s.read(256).decode(errors="ignore")
        if ch:
            buf += ch
            if ">" in buf: break
        else:
            time.sleep(0.02)
    return buf

def _txrx(s: serial.Serial, cmd: str, sleep: float = 0.12, tout: float = 1.5) -> str:
    s.reset_input_buffer()
    s.write((cmd + "\r").encode())
    time.sleep(sleep)
    return _read_to_prompt(s, tout)

def _lines(resp: str) -> List[str]:
    txt = resp.replace("\r","\n").replace(">","")
    return [ln.strip() for ln in txt.split("\n") if ln.strip()]

def _hx(line: str) -> List[int]:
    return [int(x,16) for x in line.split() if re.fullmatch(r"[0-9A-Fa-f]{2}", x)]

def elm_init(s: serial.Serial, proto: str) -> None:
    for c in ("ATZ","ATE0","ATL0","ATS0","ATH0"): _txrx(s, c, 0.15, 2.0)
    _txrx(s, f"ATSP{proto}", 0.15, 2.0)  # "0"=AUTO, "6"=ISO 15765-4 CAN 11/500, etc.
    _txrx(s, "ATAT1", 0.15, 2.0)         # adaptive timing
    _txrx(s, "ATST0A", 0.15, 2.0)        # timeout tweak

def qpid(s: serial.Serial, cmd: str) -> Optional[List[int]]:
    """
    Query a PID string like "01 0C". Returns response bytes or None.
    Accepts typical '41 <pid> ...' replies. (No ISO-TP multi-frame handling here.)
    """
    r = _txrx(s, cmd, 0.10, 1.2)
    for ln in _lines(r):
        U = ln.upper()
        if "NO DATA" in U or "UNABLE" in U: return None
        if U.startswith("SEARCHING"): continue
        bs = _hx(ln)
        if len(bs) >= 2:
            req_mode = int(cmd.split()[0], 16)
            req_pid  = int(cmd.split()[-1], 16)
            # mode 01 -> reply 41; mode 22 -> reply 62, etc.
            if req_mode + 0x40 == bs[0]:
                # if PID present, check it; some replies omit PID echo
                if len(bs) >= 2 and bs[1] in (req_pid,):
                    return bs
                # else accept anyway (best effort)
                return bs
    return None

# ---------- Decoders (Mode 01) ----------
def rpm_from_41_0C(bs: List[int]) -> float:
    return ((bs[2]<<8)|bs[3])/4.0 if len(bs) >= 4 else 0.0

def thr_from_41_11(bs: List[int]) -> float:
    return (bs[2]*100.0)/255.0 if len(bs) >= 3 else 0.0

def maf_from_41_10(bs: List[int]) -> float:
    return ((bs[2]<<8)|bs[3])/100.0 if len(bs) >= 4 else 0.0

def two_byte_value(bs: List[int]) -> float:
    """Generic 16-bit big-endian value -> float (ppm)."""
    return float(((bs[2]<<8)|bs[3])) if len(bs) >= 4 else 0.0

# ---------- Main ----------
def main() -> None:
    p = argparse.ArgumentParser("ELM327 → CSV (USB or Bluetooth/RFCOMM)")
    p.add_argument("--port", required=True, help="/dev/ttyUSB0, /dev/rfcomm0, or /dev/serial/by-id/...")
    p.add_argument("--baud", type=int, default=38400, help="38400 (common), 115200, or 9600")
    p.add_argument("--proto", default="0", help='ELM protocol: "0" AUTO (default), "6" CAN 11/500, etc.')
    p.add_argument("--hz", type=float, default=2.0, help="Sample rate (Hz)")
    p.add_argument("--label", default="unlabeled", help="fault_label column value")
    p.add_argument("--outfile", default="obd_run.csv")
    p.add_argument("--duration-s", type=float, default=0, help="Stop after N seconds (0 = until Ctrl+C)")
    p.add_argument("--nox-pid", default="", help='Optional NOx PID, e.g. "01 5E" (else zeros are logged)')
    a = p.parse_args()

    s = _open(a.port, a.baud)
    elm_init(s, a.proto)

    dt = 1.0 / max(a.hz, 0.1)
    t0 = time.monotonic()
    end = (t0 + a.duration_s) if a.duration_s > 0 else None

    with open(a.outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time (ms.)","fault_label","RPM [RPM]","PEDAL INPUT [%]","MAF [g/s]","NOx [ppm]"])
        print(f"[INFO] Logging {a.hz} Hz to {a.outfile} — Ctrl+C to stop.")
        try:
            while True:
                if end and time.monotonic() >= end: break
                tms = int((time.monotonic() - t0) * 1000)

                r = qpid(s, "01 0C"); rpm = rpm_from_41_0C(r or [])
                r = qpid(s, "01 11"); thr = thr_from_41_11(r or [])
                r = qpid(s, "01 10"); maf = maf_from_41_10(r or [])

                nox = 0.0
                if a.nox_pid:
                    r = qpid(s, a.nox_pid); nox = two_byte_value(r or [])

                w.writerow([tms, a.label, f"{rpm:.1f}", f"{thr:.1f}", f"{maf:.2f}", f"{nox:.1f}"])
                f.flush()
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
        finally:
            try: s.close()
            except: pass

if __name__ == "__main__":
    main()

