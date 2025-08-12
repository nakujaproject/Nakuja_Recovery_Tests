#!/usr/bin/env python3
"""
Deployment Events Report Generator

Examines telemetry CSV files to determine if drogue and main parachutes were deployed.
Logic hierarchy:
1. If explicit deployment flag columns exist (case-insensitive):
   - drogue_deployed, drogue, drogue_flag, drogue_event
   - main_deployed, main, main_flag, main_event
   Any value == 1 (or '1', or True) marks deployment rows.
2. Else if state column (string) contains phase names:
   - First appearance of DROGUE_DESCENT => drogue deployment time
   - First appearance of MAIN_DESCENT  => main deployment time
3. Else if only numeric state codes:
   - Detect apogee (max altitude). After apogee, record first state change as drogue, next distinct new state as main.
   (Heuristic, flagged as inferred)

For each detected deployment, output a row snapshot with key fields and also a full-row dump (CSV style) in the report.

Outputs:
 - For each input file: outputs/drone test/<file-stem>/reports/deployment_report.txt
 - Additionally, a consolidated summary: outputs/drone test/deployment_summary_all.txt

Usage:
  python deployment_report.py --files telemetry_logs/telemetry_*.csv
  python deployment_report.py --directory telemetry_logs

"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import textwrap

DROGUE_FLAG_CANDIDATES = ["drogue_deployed","drogue","drogue_flag","drogue_event"]
MAIN_FLAG_CANDIDATES   = ["main_deployed","main","main_flag","main_event"]
STRING_DROGUE_STATES = {"DROGUE_DESCENT","DROGUE","DROGUE_DEPLOY","CHUTE1","DROGUE-DEPLOY"}
STRING_MAIN_STATES   = {"MAIN_DESCENT","MAIN","MAIN_DEPLOY","CHUTE2","MAIN-DEPLOY"}

SAFE_ALT_COLS = ["kalman_altitude","agl_altitude","gps_altitude"]


def _p(msg:str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii','ignore').decode())


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'iso_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['iso_timestamp'], errors='coerce')
    # Build relative time seconds
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df['t'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    else:
        df['t'] = np.arange(len(df))*0.1
    # Numeric conversion for candidate columns
    for c in set(SAFE_ALT_COLS + ['velocity','kalman_vertical_velocity','temperature','state'] + DROGUE_FLAG_CANDIDATES + MAIN_FLAG_CANDIDATES):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='ignore')  # keep strings for state if textual
    return df


def choose_alt_col(df: pd.DataFrame) -> str | None:
    for c in SAFE_ALT_COLS:
        if c in df.columns and df[c].notna().sum()>0:
            return c
    return None


def find_flag_row(df: pd.DataFrame, candidates) -> int | None:
    for name in candidates:
        if name in df.columns:
            col = df[name]
            # treat 1 / True / '1' as deployed
            mask = (col == 1) | (col.astype(str).str.lower().isin(['1','true','yes']))
            idxs = np.where(mask.values)[0]
            if idxs.size:
                return int(idxs[0])
    return None


def find_string_state_row(df: pd.DataFrame, state_set) -> int | None:
    if 'state' not in df.columns:
        return None
    # state might be numeric or string; coerce to string
    ser = df['state'].astype(str)
    mask = ser.str.upper().isin(state_set)
    idxs = np.where(mask.values)[0]
    if idxs.size:
        return int(idxs[0])
    return None


def heuristic_numeric_states(df: pd.DataFrame, apogee_idx: int) -> tuple[int|None,int|None,bool]:
    """If states are numeric only, infer drogue & main as first two new post-apogee states distinct from apogee state.
    Returns (drogue_idx, main_idx, inferred_flag)"""
    if 'state' not in df.columns:
        return None,None, False
    states = df['state']
    # If contains any non-digit, skip heuristic
    if any(not str(s).isdigit() for s in states.dropna().unique()):
        return None,None, False
    apogee_state = states.iloc[apogee_idx] if 0 <= apogee_idx < len(states) else None
    seen = {apogee_state}
    drogue_idx = None
    main_idx = None
    for i in range(apogee_idx+1, len(states)):
        s = states.iloc[i]
        if s not in seen:
            if drogue_idx is None:
                drogue_idx = i
                seen.add(s)
            elif main_idx is None and s not in seen:
                main_idx = i
                seen.add(s)
                break
    return drogue_idx, main_idx, True


def extract_snapshot(df: pd.DataFrame, idx: int) -> dict:
    row = df.iloc[idx]
    snapshot = {
        'index': idx,
        't_s': float(row.get('t', np.nan)),
    }
    if 'timestamp' in df.columns:
        snapshot['timestamp'] = row.get('timestamp')
    for c in SAFE_ALT_COLS:
        if c in df.columns:
            snapshot[c] = row.get(c)
    for c in ['velocity','kalman_vertical_velocity','temperature','pressure','battery_voltage','wifi_rssi','lora_rssi','state']:
        if c in df.columns:
            snapshot[c] = row.get(c)
    # any flag columns present
    for c in DROGUE_FLAG_CANDIDATES + MAIN_FLAG_CANDIDATES:
        if c in df.columns:
            snapshot[c] = row.get(c)
    return snapshot


def format_snapshot(snapshot: dict) -> str:
    lines = []
    for k,v in snapshot.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def analyze_file(csv_path: Path, out_root: Path) -> dict:
    df = load_csv(csv_path)
    alt_col = choose_alt_col(df)
    apogee_idx = None
    if alt_col:
        try:
            apogee_idx = int(df[alt_col].idxmax())
        except Exception:
            apogee_idx = None

    # Strategy 1: explicit flag columns
    drogue_idx = find_flag_row(df, DROGUE_FLAG_CANDIDATES)
    main_idx   = find_flag_row(df, MAIN_FLAG_CANDIDATES)
    inferred = False

    # Strategy 2: string states
    if drogue_idx is None:
        drogue_idx = find_string_state_row(df, STRING_DROGUE_STATES)
    if main_idx is None:
        main_idx = find_string_state_row(df, STRING_MAIN_STATES)

    # Strategy 3: numeric heuristic
    if drogue_idx is None and apogee_idx is not None:
        h_drogue, h_main, inf = heuristic_numeric_states(df, apogee_idx)
        drogue_idx = drogue_idx or h_drogue
        main_idx = main_idx or h_main
        inferred = inf

    report_dir = out_root / csv_path.stem / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / 'deployment_report.txt'

    lines = []
    lines.append(f"Deployment Report: {csv_path.name}")
    lines.append("="*70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Records: {len(df)}")
    lines.append(f"Altitude column used for apogee: {alt_col if alt_col else 'N/A'}")
    if apogee_idx is not None:
        lines.append(f"Apogee index: {apogee_idx}, time: {df['t'].iloc[apogee_idx]:.2f}s")
        if alt_col:
            lines.append(f"Apogee altitude: {df[alt_col].iloc[apogee_idx]}")
    else:
        lines.append("Apogee not determined")

    if inferred:
        lines.append("Heuristic numeric state inference was used (verify manually).")

    def add_event(label, idx):
        if idx is None:
            lines.append(f"{label}: NOT DETECTED")
        else:
            snap = extract_snapshot(df, idx)
            lines.append(f"{label}: index={idx}, t={snap.get('t_s','?'):.2f}s")
            if alt_col and alt_col in snap:
                lines.append(f"  Altitude ({alt_col}): {snap[alt_col]}")
            if 'state' in snap:
                lines.append(f"  State: {snap['state']}")
            if 'temperature' in snap:
                lines.append(f"  Temperature: {snap['temperature']}")
            lines.append("  Full snapshot:")
            lines.append(textwrap.indent(format_snapshot(snap), prefix="    "))

    add_event('Drogue Deployment', drogue_idx)
    add_event('Main Deployment', main_idx)

    # Save per-file report
    report_file.write_text("\n".join(lines)+"\n", encoding='utf-8')

    return {
        'file': csv_path.name,
        'drogue_idx': drogue_idx,
        'main_idx': main_idx,
        'apogee_idx': apogee_idx,
        'drogue_time': None if drogue_idx is None else df['t'].iloc[drogue_idx],
        'main_time': None if main_idx is None else df['t'].iloc[main_idx],
        'apogee_time': None if apogee_idx is None else df['t'].iloc[apogee_idx],
        'drogue_alt': None if drogue_idx is None or not alt_col else df[alt_col].iloc[drogue_idx],
        'main_alt': None if main_idx is None or not alt_col else df[alt_col].iloc[main_idx],
        'apogee_alt': None if apogee_idx is None or not alt_col else df[alt_col].iloc[apogee_idx],
        'heuristic': inferred
    }


def write_summary(summaries, out_root: Path):
    if not summaries:
        return
    lines = ["Consolidated Deployment Summary", "="*70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    hdr = ["file","drogue_time_s","drogue_alt","main_time_s","main_alt","apogee_time_s","apogee_alt","heuristic"]
    lines.append(",".join(hdr))
    for s in summaries:
        row = [
            s['file'],
            f"{s['drogue_time']:.2f}" if s['drogue_time'] is not None else "",
            f"{s['drogue_alt']:.2f}" if s['drogue_alt'] is not None else "",
            f"{s['main_time']:.2f}" if s['main_time'] is not None else "",
            f"{s['main_alt']:.2f}" if s['main_alt'] is not None else "",
            f"{s['apogee_time']:.2f}" if s['apogee_time'] is not None else "",
            f"{s['apogee_alt']:.2f}" if s['apogee_alt'] is not None else "",
            "Y" if s['heuristic'] else "N"
        ]
        lines.append(",".join(row))
    (out_root / 'deployment_summary_all.txt').write_text("\n".join(lines)+"\n", encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Generate drogue/main deployment reports from telemetry CSV files')
    parser.add_argument('--files', nargs='+', help='Specific CSV files')
    parser.add_argument('--directory','-d', help='Directory containing telemetry_*.csv files')
    parser.add_argument('--out', default='outputs/drone test', help='Output root (default: outputs/drone test)')
    args = parser.parse_args()

    files = []
    if args.files:
        files = [Path(f) for f in args.files]
    elif args.directory:
        p = Path(args.directory)
        files = sorted(p.glob('telemetry_*.csv'))
    else:
        # default: all in telemetry_logs
        p = Path('telemetry_logs')
        files = sorted(p.glob('telemetry_*.csv'))

    if not files:
        _p('No CSV files found for analysis.')
        return

    out_root = Path(args.out)
    summaries = []
    for f in files:
        _p(f'Processing {f.name} ...')
        try:
            s = analyze_file(f, out_root)
            summaries.append(s)
        except Exception as e:
            _p(f'Error processing {f}: {e}')
    write_summary(summaries, out_root)
    _p('Deployment analysis complete.')

if __name__ == '__main__':
    main()
