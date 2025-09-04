#!/usr/bin/env python3
"""
ELM327 USB > CSV logger (Edge Impulse friendly).

- Default 2 Hz timing with stable period scheduling.
- Robust ELM init (ATZ, ATE0, ATL0, ATS0, ATHx, ATSPx, ATSTxx, ATDPN).
- Logs: time (ms.), fault_label, RPM, PEDAL INPUT, MAF, SPEED, COOLANT, MAP.
- Optionally writes zeros when PIDs missing (engine off or not supported).

Tested with typical ELM327 USB adapters on Linux/Raspberry Pi.
"""
from __future__ import annotations
import argparse, csv, sys, time, signal, os, re, typing as t
import serial  # pyserial

PROMPT = b'>'
CR = b'\r'
HEX2 = re.compile(r'([0-9A-Fa-f]{2})')

def read_until_prompt(ser: serial.Serial, timeout: float = 2.0) -> str:
    t0 = time.time()
    buf = b""
    while time.time() - t0 < timeout:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if PROMPT in buf:
                break
        else:
            time.sleep(0.01)
    return buf.decode(errors="ignore")

def send_cmd(ser: serial.Serial, cmd: str, timeout: float = 2.0, pause: float = 0.02) -> list[str]:
    ser.reset_input_buffer()
    ser.write(cmd.encode("ascii") + CR)
    ser.flush()
    time.sleep(pause)
    raw = read_until_prompt(ser, timeout=timeout)
    # Normalize and clean lines
    lines = [l.strip() for l in raw.replace('\r','\n').split('\n')]
    clean: list[str] = []
    for l in lines:
        if not l or l == '>' or l.upper().startswith('SEARCHING'):
            continue
        clean.append(l)
    return clean

def init_elm(ser: serial.Serial, protocol: str = "6", timeout_hex: str = "0A", show_headers: bool = False) -> None:
    # timeout_hex: ATSTxx (xx hex; 0A≈100ms, 14≈200ms, 1E≈300ms)
    seq = [
        "ATZ", "ATE0", "ATL0", "ATS0",
        ("ATH1" if show_headers else "ATH0"),
        f"ATSP{protocol}", f"ATST{timeout_hex}", "ATDPN"
    ]
    for c in seq:
        send_cmd(ser, c, timeout=2.0)
        time.sleep(0.05)

def parse_pid(lines: list[str], pid_hex: str) -> t.Optional[list[int]]:
    """Find bytes after '41 <pid>' (skip NO DATA / 7F negatives)."""
    for ln in lines:
        up = ln.upper()
        if "NO DATA" in up or up.startswith("7F"):
            continue
        bs = HEX2.findall(ln)
        for i in range(len(bs) - 2):
            if bs[i].lower() == "41" and bs[i + 1].lower() == pid_hex:
                return [int(x, 16) for x in bs[i + 2:]]
    return None

def q_pid_bytes(ser: serial.Serial, mode_pid: str, pid_hex: str, need: int = 1, tries: int = 3, tout: float = 1.2) -> t.Optional[list[int]]:
    for _ in range(tries):
        lines = send_cmd(ser, mode_pid, timeout=tout)
        bs = parse_pid(lines, pid_hex)
        if bs and len(bs) >= need:
            return bs
        time.sleep(0.05)
    return None

# Decoders (return float or None)
def q_rpm(ser: serial.Serial) -> t.Optional[float]:
    bs = q_pid_bytes(ser, "010C", "0c", need=2)
    return None if not bs else ((bs[0] << 8) | bs[1]) / 4.0

def q_maf(ser: serial.Serial) -> t.Optional[float]:
    bs = q_pid_bytes(ser, "0110", "10", need=2)
    return None if not bs else ((bs[0] << 8) | bs[1]) / 100.0

def q_throttle(ser: serial.Serial) -> t.Optional[float]:
    bs = q_pid_bytes(ser, "0111", "11", need=1)
    return None if not bs else (bs[0] * 100.0) / 255.0

def q_speed(ser: serial.Serial) -> t.Optional[float]:
    bs = q_pid_bytes(ser, "010D", "0d", need=1)
    return None if not bs else float(bs[0])  # km/h

def q_coolant(ser: serial.Serial) -> t.Optional[float]:
    bs = q_pid_bytes(ser, "0105", "05", need=1)
    return None if not bs else float(bs[0] - 40)  # °C

def q_map(ser: serial.Serial) -> t.Optional[float]:
    bs = q_pid_bytes(ser, "010B", "0b", need=1)
    return None if not bs else float(bs[0])  # kPa

stop = False
def _stop(*_: object) -> None:
    global stop
    stop = True

def main() -> None:
    ap = argparse.ArgumentParser(description="ELM327 USB -> CSV (Edge Impulse friendly)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=38400)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--label", default="healthy")
    ap.add_argument("--out", default="obd_log.csv")
    ap.add_argument("--force-protocol", default="6", help="ELM ATSPx (6=ISO15765-4 CAN 11/500). Use 0 for auto.")
    ap.add_argument("--timeout-hex", default="0A", help="ATSTxx hex timeout (0A≈100ms, 14≈200ms, 1E≈300ms)")
    ap.add_argument("--headers", action="store_true", help="ATH1 during init (debug)")
    ap.add_argument("--zeros-when-missing", action="store_true",
                    help="Write 0.0 when PID missing/None (else blank).")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.25)
    except Exception as e:
        print(f"ERROR opening {args.port}@{args.baud}: {e}", file=sys.stderr)
        sys.exit(1)

    init_elm(ser, protocol=args.force_protocol, timeout_hex=args.timeout_hex, show_headers=args.headers)

    fields = ["time (ms.)","fault_label",
              "RPM [RPM]","PEDAL INPUT [%]","MAF [g/s]",
              "SPEED [km/h]","COOLANT [C]","MAP [kPa]"]

    write_header = not os.path.exists(args.out)
    t0 = time.time()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()

        print(f"Logging > {args.out} at {args.hz} Hz. Ctrl+C to stop.")
        period = 1.0 / max(0.1, args.hz)

        while not stop:
            ts = time.time()
            row: dict[str, str | int] = {
                "time (ms.)": int((ts - t0) * 1000),
                "fault_label": args.label
            }

            rpm = q_rpm(ser)
            thr = q_throttle(ser)
            maf = q_maf(ser)
            spd = q_speed(ser)
            ect = q_coolant(ser)
            mapk = q_map(ser)

            def fmt(val: t.Optional[float]) -> str:
                if val is None:
                    return "0.000" if args.zeros_when_missing else ""
                return f"{val:.3f}"

            row["RPM [RPM]"]       = fmt(rpm)
            row["PEDAL INPUT [%]"] = fmt(thr)
            row["MAF [g/s]"]       = fmt(maf)
            row["SPEED [km/h]"]    = fmt(spd)
            row["COOLANT [C]"]     = fmt(ect)
            row["MAP [kPa]"]       = fmt(mapk)

            w.writerow(row)  # type: ignore[arg-type]
            f.flush()

            sleep_left = period - (time.time() - ts)
            if sleep_left > 0:
                time.sleep(sleep_left)

    try:
        ser.close()
    except Exception:
        pass
    print("\nStopped. CSV saved.")

if __name__ == "__main__":
    main()
