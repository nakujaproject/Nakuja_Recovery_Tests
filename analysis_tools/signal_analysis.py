#!/usr/bin/env python3
"""
Signal Strength Depreciation Analyzer
- Focuses on RSSI decay over time for WiFi/LoRa
- Detects significant drops and outages
- Compares the last two CSV logs or specific files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_palette('tab10')

RSSI_COLS = ['wifi_rssi', 'lora_rssi']

# Safe print helper to avoid Unicode issues on Windows terminals
def _p(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback: strip non-ascii
        try:
            print(msg.encode('ascii', 'ignore').decode())
        except Exception:
            print(str(msg))


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'iso_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['iso_timestamp'])
        df['t'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    else:
        # assume 10 Hz
        df['t'] = np.arange(len(df)) * 0.1
        df['timestamp'] = pd.Timestamp.now() + pd.to_timedelta(df['t'], unit='s')
    # numeric
    for c in RSSI_COLS + ['agl_altitude', 'kalman_altitude', 'velocity', 'kalman_vertical_velocity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def window_points(df: pd.DataFrame, seconds: float) -> int:
    if 't' not in df:
        return max(1, int(seconds * 10))
    dt = np.median(np.diff(df['t'].values)) if len(df) > 1 else 0.1
    return max(1, int(seconds / max(dt, 1e-3)))


def linear_fit(time_s: np.ndarray, rssi: np.ndarray):
    # Fit RSSI = a * time_minutes + b
    t_min = time_s / 60.0
    mask = np.isfinite(t_min) & np.isfinite(rssi)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(t_min[mask], rssi[mask], 1)
    # R^2
    y_pred = a * t_min[mask] + b
    ss_res = np.sum((rssi[mask] - y_pred) ** 2)
    ss_tot = np.sum((rssi[mask] - np.mean(rssi[mask])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def find_drop_events(time_s: np.ndarray, rssi: np.ndarray, window_s: float = 5.0, drop_db: float = 10.0):
    # Identify sharp drops greater than drop_db within window_s
    if len(rssi) < 3:
        return []
    # Rolling min of forward difference
    t = time_s
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    w = max(1, int(window_s / max(dt, 1e-3)))
    series = pd.Series(rssi)
    # change over window using rolling min of difference
    diff = series.diff(w)
    events = []
    for i in range(len(diff)):
        d = diff.iloc[i]
        if pd.notna(d) and d <= -abs(drop_db):
            events.append({'idx': i, 'time': t[i], 'drop_db': float(abs(d))})
    # Deduplicate nearby events (keep the larger drop within +/- window)
    pruned = []
    last_idx = -1e9
    for e in events:
        if e['idx'] - last_idx < w:
            if pruned and e['drop_db'] > pruned[-1]['drop_db']:
                pruned[-1] = e
        else:
            pruned.append(e)
        last_idx = e['idx']
    return pruned


def find_outages(time_s: np.ndarray, rssi: np.ndarray, threshold: float = -100.0, min_gap_s: float = 2.0):
    # Outage defined as NaN or rssi below threshold continuously for >= min_gap_s
    is_bad = (~np.isfinite(rssi)) | (rssi <= threshold)
    if is_bad.sum() == 0:
        return []
    dt = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.1
    min_len = max(1, int(min_gap_s / max(dt, 1e-3)))
    outages = []
    start = None
    for i, bad in enumerate(is_bad):
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            if i - start >= min_len:
                outages.append({'start_idx': start, 'end_idx': i-1, 'start_t': time_s[start], 'end_t': time_s[i-1]})
            start = None
    # end case
    if start is not None and len(time_s) - start >= min_len:
        outages.append({'start_idx': start, 'end_idx': len(time_s)-1, 'start_t': time_s[start], 'end_t': time_s[-1]})
    return outages


def analyze_file(csv_path: Path, out_dir: Path):
    df = load_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [f"Signal Analysis Report for {csv_path.name}", "="*60]

    # Plot both RSSI channels
    fig, ax = plt.subplots(figsize=(12, 7))
    t = df['t'].values

    for col, color in zip(RSSI_COLS, ['#1f77b4', '#ff7f0e']):
        if col in df.columns:
            rssi = df[col].values.astype(float)
            ax.scatter(t/60.0, rssi, s=6, alpha=0.35, label=f'{col} raw', color=color)

            # Rolling median
            w = window_points(df, seconds=5)
            smooth = pd.Series(rssi).rolling(window=w, min_periods=max(1, w//3)).median()
            ax.plot(t/60.0, smooth, lw=2, color=color, label=f'{col} 5s median')

            # Linear decay fit
            a, b, r2 = linear_fit(t, rssi)
            if np.isfinite(a):
                ax.plot(t/60.0, a*(t/60.0)+b, ls='--', color=color, alpha=0.7)
                summary_lines.append(f"{col}: slope={a:.2f} dB/min, R^2={r2:.3f}")

            # Drops
            drops = find_drop_events(t, rssi, window_s=5.0, drop_db=8.0)
            for d in drops:
                ax.axvline(d['time']/60.0, color=color, ls=':', alpha=0.4)
            if drops:
                summary_lines.append(f"{col}: sharp drops (>=8 dB/5s) count={len(drops)}")
                for d in drops[:10]:
                    summary_lines.append(f"  - t={d['time']:.1f}s, drop≈{d['drop_db']:.1f} dB")

            # Outages
            outages = find_outages(t, rssi, threshold=-95.0, min_gap_s=2.0)
            total_outage = sum(o['end_t']-o['start_t'] for o in outages)
            if outages:
                summary_lines.append(f"{col}: outages >=2s count={len(outages)}, total={total_outage:.1f}s")

    # Annotate phases if available
    if 'state' in df.columns and df['state'].notna().any():
        # Show phase bands for visual context
        states = df['state'].ffill()
        unique_states = list(states.unique())
        # Draw background bands for long phases
        last_state = None
        start_idx = 0
        for i, s in enumerate(states):
            if last_state is None:
                last_state = s
            elif s != last_state:
                if i - start_idx > 20:
                    ax.axvspan(df['t'].iloc[start_idx]/60.0, df['t'].iloc[i-1]/60.0, color='gray', alpha=0.05)
                    ax.text((df['t'].iloc[start_idx]+df['t'].iloc[i-1])/120.0, 
                            ax.get_ylim()[1]-3, str(last_state), fontsize=8, alpha=0.5)
                last_state = s
                start_idx = i

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('RSSI (dBm)')
    ax.set_title(f'Signal Strength Over Time - {csv_path.stem}')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'signal_over_time.png', dpi=300)
    plt.close()

    # Save summary
    (out_dir / 'signal_summary.txt').write_text("\n".join(summary_lines), encoding='utf-8')
    return df


def compare_two(files, out_root: Path):
    # Compare RSSI between two files
    dfs = []
    names = []
    for f in files:
        df = load_csv(f)
        dfs.append(df)
        names.append(f.stem)

    out_dir = out_root / f"compare_{names[0]}_vs_{names[1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overlay plots for WiFi and LoRa
    for col, title in [('wifi_rssi', 'WiFi RSSI'), ('lora_rssi', 'LoRa RSSI')]:
        if any(col in df.columns for df in dfs):
            fig, ax = plt.subplots(figsize=(12,7))
            for df, name, color in zip(dfs, names, ['#1f77b4', '#ff7f0e']):
                if col in df.columns:
                    t = df['t'].values
                    rssi = df[col].values.astype(float)
                    w = window_points(df, seconds=5)
                    smooth = pd.Series(rssi).rolling(window=w, min_periods=max(1, w//3)).median()
                    ax.plot(t/60.0, smooth, lw=2, color=color, label=f'{name}')
                    a, b, r2 = linear_fit(t, rssi)
                    if np.isfinite(a):
                        ax.plot(t/60.0, a*(t/60.0)+b, ls='--', color=color, alpha=0.6)
                        ax.text(0.01, 0.95 - 0.08*names.index(name), 
                                f"{name}: {a:.2f} dB/min (R^2={r2:.2f})", transform=ax.transAxes,
                                fontsize=9, color=color)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('RSSI (dBm)')
            ax.set_title(f'{title} Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'{col}_comparison.png', dpi=300)
            plt.close()

    # Quick numerical comparison summary
    lines = ["Signal Comparison Summary", "="*60]
    for col in RSSI_COLS:
        if all(col in df.columns for df in dfs):
            a0, *_ = linear_fit(dfs[0]['t'].values, dfs[0][col].values.astype(float))
            a1, *_ = linear_fit(dfs[1]['t'].values, dfs[1][col].values.astype(float))
            mean0 = np.nanmean(dfs[0][col].values)
            mean1 = np.nanmean(dfs[1][col].values)
            lines.append(f"{col}: mean {names[0]}={mean0:.1f} dBm, {names[1]}={mean1:.1f} dBm")
            lines.append(f"{col}: slope {names[0]}={a0:.2f} dB/min, {names[1]}={a1:.2f} dB/min")
    (out_dir / 'comparison_summary.txt').write_text("\n".join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Analyze RSSI depreciation and compare flights')
    parser.add_argument('--files', nargs='+', help='Specific CSV files to analyze')
    parser.add_argument('--last2', action='store_true', help='Use the last two CSV files in telemetry_logs')
    parser.add_argument('--out', default='signal_plots', help='Output root directory')
    args = parser.parse_args()

    out_root = Path(args.out)

    files = []
    if args.files:
        files = [Path(f) for f in args.files]
    elif args.last2:
        logs = sorted(Path('telemetry_logs').glob('telemetry_*.csv'), key=lambda p: p.stat().st_mtime)
        if len(logs) < 2:
            _p('ERROR: Need at least two CSV files in telemetry_logs')
            return
        files = logs[-2:]
        _p(f"Using last two CSVs: {files[0].name}, {files[1].name}")
    else:
        logs = sorted(Path('telemetry_logs').glob('telemetry_*.csv'), key=lambda p: p.stat().st_mtime)
        if not logs:
            _p('ERROR: No CSV files found')
            return
        files = [logs[-1]]
        _p(f"Using latest CSV: {files[0].name}")

    analyzed = []
    for f in files:
        _p(f"Analyzing {f.name}...")
        out_dir = out_root / f.stem
        df = analyze_file(f, out_dir)
        analyzed.append(df)

    if len(files) == 2:
        compare_two(files, out_root)

    _p('Signal analysis complete')

if __name__ == '__main__':
    main()
