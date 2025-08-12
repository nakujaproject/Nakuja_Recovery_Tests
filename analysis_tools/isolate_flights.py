#!/usr/bin/env python3
"""
Flight Segment Isolator (Drone Lift Tests)

Purpose:
  Scan telemetry CSV logs and extract ONLY the active flight segments where altitude
  rises from a near-ground baseline to a drone-assisted apogee (~100 m) then descends.

Detection Logic:
  1. Choose best altitude column (kalman_altitude > agl_altitude > gps_altitude).
  2. Smooth altitude with a rolling median + rolling mean to reduce noise.
  3. Detect peaks using scipy.signal.find_peaks (if available) or fallback manual method.
  4. Keep peaks within configurable window (min_peak <= peak <= max_peak).
  5. For each peak locate segment start (scan backward until altitude near baseline)
     and segment end (scan forward until altitude returns near baseline).
  6. Validate segment by minimum duration, minimum ascent, and monotonic shape ratio.
  7. Save each isolated segment to outputs/flight_segments/<file-stem>_seg<k>.csv
     plus a plot with annotations and metrics.
  8. Produce consolidated summary CSV + human readable report.

CLI:
  python isolate_flights.py --directory telemetry_logs --min-peak 80 --max-peak 130 --target 100
  python isolate_flights.py --files telemetry_logs/telemetry_2025*.csv

Outputs:
  outputs/flight_segments/segments_summary.csv
  outputs/flight_segments/segments_report.txt
  outputs/flight_segments/<source_stem>_seg<k>.csv
  outputs/flight_segments/<source_stem>_seg<k>.png

Assumptions:
  - Sampling rate is roughly constant; if no timestamp present we assume 10 Hz.
  - Baseline altitude is median of first 5% (capped [0,2000] samples) of the file.

Safe for Windows console (no emojis).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import math
import warnings
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

plt.style.use('seaborn-v0_8')

ALT_PRIORITY = ["kalman_altitude", "agl_altitude", "gps_altitude"]


def _p(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii','ignore').decode())


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'iso_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['iso_timestamp'], errors='coerce')
        if df['timestamp'].notna().any():
            df['t'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    if 't' not in df.columns:
        # assume 10 Hz
        df['t'] = np.arange(len(df)) * 0.1
    for c in ALT_PRIORITY + ['velocity','kalman_vertical_velocity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def choose_alt_col(df: pd.DataFrame):
    for c in ALT_PRIORITY:
        if c in df.columns and df[c].notna().sum() > 5:
            return c
    return None


def smooth_altitude(alt: pd.Series, sample_dt: float) -> pd.Series:
    # Rolling median then mean; window ~ 0.5 s
    win = max(3, int(round(0.5 / max(sample_dt, 1e-3))))
    med = alt.rolling(window=win, center=True, min_periods=1).median()
    sm = med.rolling(window=win, center=True, min_periods=1).mean()
    return sm


def detect_peaks(alt_s: pd.Series, min_peak: float, max_peak: float, sample_dt: float):
    y = alt_s.values
    if len(y) < 10:
        return []
    distance = max(5, int(round(1.0 / max(sample_dt, 1e-3))))  # at least 1s apart
    if HAVE_SCIPY:
        peaks, props = find_peaks(y, height=min_peak, distance=distance, prominence=3.0)
        peaks = [int(p) for p in peaks if y[p] <= max_peak]
    else:
        # simple manual: local max and threshold
        peaks = []
        for i in range(3, len(y)-3):
            if y[i] >= min_peak and y[i] <= max_peak and y[i] == max(y[i-3:i+4]):
                peaks.append(i)
        # enforce distance
        filtered = []
        for p in peaks:
            if not filtered or (p - filtered[-1]) >= distance:
                filtered.append(p)
        peaks = filtered
    return peaks


def find_segment_bounds(alt_s: pd.Series, peak_idx: int, baseline: float, sample_dt: float, margin: float = 5.0):
    y = alt_s.values
    n = len(y)
    # backward
    start = peak_idx
    thresh = baseline + margin
    for i in range(peak_idx, -1, -1):
        if y[i] <= thresh:
            start = i
            break
    # forward
    end = peak_idx
    for j in range(peak_idx, n):
        if y[j] <= thresh:
            end = j
            break
    # expand a tiny margin
    return max(0, start-2), min(n-1, end+2)


def segment_metrics(df: pd.DataFrame, alt_col: str, start: int, end: int):
    seg = df.iloc[start:end+1]
    t = seg['t'].values
    alt = seg[alt_col].values
    if len(t) < 2:
        return None
    dur = t[-1] - t[0]
    peak_alt = np.nanmax(alt)
    peak_idx_rel = np.nanargmax(alt)
    time_to_peak = t[peak_idx_rel] - t[0]
    ascent_rate = (peak_alt - alt[0]) / max(time_to_peak, 1e-6)
    descent_rate = (peak_alt - alt[-1]) / max(t[-1]-t[peak_idx_rel], 1e-6)
    monotonic_up_frac = np.mean(np.diff(alt[:peak_idx_rel+1]) >= -0.2)
    monotonic_down_frac = np.mean(np.diff(alt[peak_idx_rel:]) <= 0.2)
    return {
        'start_index': start,
        'end_index': end,
        'duration_s': dur,
        'peak_alt': peak_alt,
        'time_to_peak_s': time_to_peak,
        'ascent_rate_mps': ascent_rate,
        'descent_rate_mps': descent_rate,
        'monotonic_up_frac': monotonic_up_frac,
        'monotonic_down_frac': monotonic_down_frac
    }


def validate_segment(m: dict, min_duration: float, min_ascent_rate: float, target_low: float, target_high: float):
    if m['peak_alt'] < target_low or m['peak_alt'] > target_high:
        return False
    if m['duration_s'] < min_duration:
        return False
    if m['ascent_rate_mps'] < min_ascent_rate:
        return False
    if m['monotonic_up_frac'] < 0.7:
        return False
    if m['monotonic_down_frac'] < 0.5:
        return False
    return True


def plot_segment(df: pd.DataFrame, alt_col: str, start: int, end: int, metrics: dict, out_path: Path):
    seg = df.iloc[start:end+1]
    t0 = df['t'].iloc[start]
    t = seg['t'] - t0
    alt = seg[alt_col]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, alt, label='Altitude', lw=2)
    ax.set_xlabel('Segment Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f"Segment peak {metrics['peak_alt']:.1f} m, dur {metrics['duration_s']:.1f}s")
    ax.grid(True, alpha=0.3)
    # annotate peak
    peak_rel_idx = np.nanargmax(alt.values)
    ax.axvline(t.iloc[peak_rel_idx], color='k', ls='--', alpha=0.5)
    ax.text(t.iloc[peak_rel_idx], metrics['peak_alt'], f"Peak {metrics['peak_alt']:.1f} m", ha='center', va='bottom')
    # ascent/descent rates
    ax.text(0.02, 0.95, f"Ascent {metrics['ascent_rate_mps']:.2f} m/s\nDescent {metrics['descent_rate_mps']:.2f} m/s", transform=ax.transAxes, va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def process_file(path: Path, args, summary_rows: list):
    df = load_csv(path)
    alt_col = choose_alt_col(df)
    if not alt_col:
        _p(f"{path.name}: No altitude column available")
        return
    alt = df[alt_col].astype(float)
    if alt.notna().sum() < 20:
        _p(f"{path.name}: Insufficient altitude data")
        return
    # sample dt
    t = df['t'].values
    if len(t) > 1:
        sample_dt = float(np.median(np.diff(t)))
    else:
        sample_dt = 0.1
    alt_s = smooth_altitude(alt, sample_dt)
    # baseline from first 5% or first 200 samples
    first_n = max(10, min(int(len(alt_s) * 0.05), 200))
    baseline = np.nanmedian(alt_s.iloc[:first_n])
    peaks = detect_peaks(alt_s, args.min_peak, args.max_peak, sample_dt)
    used_ranges = []
    seg_counter = 0
    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True, parents=True)

    for p in peaks:
        start, end = find_segment_bounds(alt_s, p, baseline, sample_dt, margin=args.baseline_margin)
        # overlap check
        if any(not (end < a or start > b) for a,b in used_ranges):
            continue
        m = segment_metrics(df, alt_col, start, end)
        if not m:
            continue
        if validate_segment(m, args.min_duration, args.min_ascent_rate, args.target_low, args.target_high):
            seg_counter += 1
            used_ranges.append((start,end))
            seg_df = df.iloc[start:end+1].copy()
            seg_name_base = f"{path.stem}_seg{seg_counter}"
            seg_csv = out_root / f"{seg_name_base}.csv"
            seg_plot = out_root / f"{seg_name_base}.png"
            seg_df.to_csv(seg_csv, index=False)
            plot_segment(df, alt_col, start, end, m, seg_plot)
            # record summary
            summary_rows.append({
                'file': path.name,
                'segment_id': seg_counter,
                'alt_col': alt_col,
                **m
            })
    if seg_counter == 0:
        _p(f"{path.name}: No qualifying segments found (peaks={len(peaks)})")
    else:
        _p(f"{path.name}: {seg_counter} segment(s) isolated")


def main():
    parser = argparse.ArgumentParser(description='Isolate drone-assisted flight segments reaching ~100m and descending')
    parser.add_argument('--files', nargs='+', help='Specific telemetry CSV files')
    parser.add_argument('--directory','-d', help='Directory containing telemetry_*.csv')
    parser.add_argument('--out', default='outputs/flight_segments', help='Output directory for segments')
    parser.add_argument('--min-peak', dest='min_peak', type=float, default=60.0, help='Minimum peak altitude to consider (m)')
    parser.add_argument('--max-peak', dest='max_peak', type=float, default=150.0, help='Maximum peak altitude to consider (m)')
    parser.add_argument('--target-low', dest='target_low', type=float, default=80.0, help='Lower bound of accepted peak altitude window (m)')
    parser.add_argument('--target-high', dest='target_high', type=float, default=120.0, help='Upper bound of accepted peak altitude window (m)')
    parser.add_argument('--baseline-margin', type=float, default=5.0, help='Altitude above baseline considered ground (m)')
    parser.add_argument('--min-duration', type=float, default=5.0, help='Minimum segment duration (s)')
    parser.add_argument('--min-ascent-rate', type=float, default=1.0, help='Minimum average ascent rate (m/s)')
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        directory = Path(args.directory) if args.directory else Path('telemetry_logs')
        files = sorted(directory.glob('telemetry_*.csv'))

    if not files:
        _p('No files to process.')
        return

    summary_rows = []
    for f in files:
        _p(f'Processing {f.name} ...')
        try:
            process_file(f, args, summary_rows)
        except Exception as e:
            _p(f'Error processing {f}: {e}')

    if summary_rows:
        out_root = Path(args.out)
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_root / 'segments_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        # human report
        lines = ["Flight Segment Isolation Report", "="*60]
        for r in summary_rows:
            lines.append(f"{r['file']} seg{r['segment_id']}: peak={r['peak_alt']:.1f}m dur={r['duration_s']:.1f}s ascent={r['ascent_rate_mps']:.2f}m/s descent={r['descent_rate_mps']:.2f}m/s")
        (out_root / 'segments_report.txt').write_text('\n'.join(lines)+"\n", encoding='utf-8')
        _p(f"Summary written: {summary_csv}")
    else:
        _p('No valid segments found across all files.')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
