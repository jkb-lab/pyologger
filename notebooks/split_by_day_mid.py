#!/usr/bin/env python3
import csv, gzip, io, os, argparse
from pathlib import Path
from datetime import datetime

# === CONFIG ===
IN_PATH = Path("/Volumes/WORK-SSD/Datasets/Unpublished/pale-adult-lion_vid-accel_africa_TW/00_Data-from-Mike/16660 Astrid/ACC_Astrid2.csv")
OUT_DIR = IN_PATH.parent
OUT_PREFIX = IN_PATH.stem + "_"
ENCODING = "utf-8"
DELIM = ","
GZIP_OUTPUT = False
DATETIME_COL = "Date ISO String"
MILLIS_COL   = "UTC Milliseconds since 1970"

def parse_dt(row):
    s = (row.get(DATETIME_COL) or "").strip()
    if s:
        t = s.replace("Z", "").split("+")[0]
        try: return datetime.fromisoformat(t)
        except Exception: pass
    m = (row.get(MILLIS_COL) or "").strip()
    if m.isdigit():
        try: return datetime.utcfromtimestamp(int(m)/1000.0)
        except Exception: pass
    return None

def open_out(day_str, header):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if GZIP_OUTPUT else ".csv"
    p = OUT_DIR / f"{OUT_PREFIX}{day_str}{suffix}"
    f = gzip.open(p, "wt", newline="", encoding=ENCODING) if GZIP_OUTPUT else p.open("w", newline="", encoding=ENCODING)
    w = csv.writer(f, delimiter=DELIM)
    w.writerow(header)
    return p, f, w

def find_real_header(f):
    """Skip metadata lines like 'Collar: ...' and return (header, offset_bytes)."""
    while True:
        pos = f.tell()
        line = f.readline()
        if not line: raise RuntimeError("EOF before header found.")
        if line.lower().startswith("collar:"):  # skip metadata
            continue
        header = [h.strip() for h in line.rstrip("\r\n").split(DELIM)]
        return header, f.tell()

def find_safe_start(path: Path, start_percent: float):
    """Seek to approximately start_percent of file and move to next newline."""
    size = os.path.getsize(path)
    start = int(size * start_percent)
    with path.open("rb") as f:
        f.seek(start)
        f.readline()  # skip partial line
        pos = f.tell()
    return pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-percent", type=float, default=0.0,
                    help="Start reading roughly at this fraction of file (0.0–1.0).")
    args = ap.parse_args()

    start_byte = 0
    if args.start_percent:
        start_byte = find_safe_start(IN_PATH, args.start_percent)
        print(f"Starting around {args.start_percent*100:.1f}% of file (byte {start_byte:,})")

    with IN_PATH.open("r", encoding=ENCODING, newline="") as f:
        # if starting mid-file, skip metadata/header detection
        if args.start_percent == 0.0:
            header, data_offset = find_real_header(f)
        else:
            # find header separately
            header, data_offset = find_real_header(open(IN_PATH, "r", encoding=ENCODING))
            f.seek(start_byte)
        r = csv.DictReader(f, fieldnames=header, delimiter=DELIM)

        out_handle = None
        out_writer = None
        current_day = None

        try:
            for row in r:
                ts = parse_dt(row)
                if ts is None:
                    continue
                day = ts.strftime("%Y-%m-%d")
                if current_day != day:
                    if out_handle: out_handle.close()
                    _, out_handle, out_writer = open_out(day, header)
                    current_day = day
                out_writer.writerow([row.get(col, "") for col in header])
        finally:
            if out_handle:
                out_handle.close()
    print("Done.")

if __name__ == "__main__":
    main()
