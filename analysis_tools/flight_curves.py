#!/usr/bin/env python3
"""
Flight Curves Isolation and Plotting
- Detects flight segments (launch, powered, coast, apogee, descent, landing)
- Plots segmented altitude and velocity curves with annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

plt.style.use('seaborn-v0_8')

# Safe print for Windows terminals
_def_dt = 0.1

def _p(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode('ascii', 'ignore').decode())


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'iso_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['iso_timestamp'])
        df['t'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    else:
        df['t'] = np.arange(len(df)) * _def_dt
    # Make numeric
    for col in ['agl_altitude','gps_altitude','kalman_altitude','velocity','kalman_vertical_velocity','battery_voltage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def detect_events(df: pd.DataFrame) -> dict:
    events = {}
    t = df['t']
    # Use best altitude
    alt_col = 'kalman_altitude' if 'kalman_altitude' in df.columns and df['kalman_altitude'].notna().any() else 'agl_altitude'
    vel_col = 'kalman_vertical_velocity' if 'kalman_vertical_velocity' in df.columns and df['kalman_vertical_velocity'].notna().any() else 'velocity'

    alt = df[alt_col] if alt_col in df else pd.Series(index=df.index, dtype=float)
    vel = df[vel_col] if vel_col in df else pd.Series(index=df.index, dtype=float)

    # Launch detection: velocity or altitude rate exceeds threshold
    launch_idx = None
    if vel_col in df:
        try:
            v = vel.fillna(0).values
            # consider launch when velocity > 5 m/s sustained for > 0.5s
            dt = np.median(np.diff(t)) if len(t) > 1 else _def_dt
            window = max(1, int(0.5 / max(1e-3, dt)))
            v_smooth = pd.Series(v).rolling(window=window, min_periods=1).mean().values
            cand = np.where(v_smooth > 5)[0]
            launch_idx = int(cand[0]) if cand.size else None
        except Exception:
            launch_idx = None
    if launch_idx is None and alt_col in df:
        # fallback: altitude rising faster than 1 m/s over 1s
        try:
            dalt = np.gradient(alt.ffill().fillna(0), t)
            cand = np.where(dalt > 1.0)[0]
            launch_idx = int(cand[0]) if cand.size else None
        except Exception:
            launch_idx = None

    # Apogee
    apogee_idx = None
    if alt_col in df and df[alt_col].notna().any():
        apogee_idx = int(df[alt_col].idxmax())

    # Landing: altitude returns near start altitude and velocity small
    landing_idx = None
    if alt_col in df:
        base_alt = np.nanmedian(df[alt_col].head(100))
        thresh_alt = base_alt + 3.0  # 3m above base
        cond = (df[alt_col] < thresh_alt)
        if vel_col in df:
            cond = cond & (df[vel_col].abs() < 0.5)
        landing_candidates = np.where(cond.values)[0]
        if landing_candidates.size:
            landing_idx = int(landing_candidates[-1])

    # Powered/coast split approximate by peak velocity between launch and apogee
    powered_end_idx = None
    if launch_idx is not None and vel_col in df:
        try:
            start = launch_idx
            end = apogee_idx if apogee_idx is not None else len(df)-1
            if end > start + 5:
                seg = df[vel_col].iloc[start:end].ffill()
                powered_end_idx = int(seg.idxmax())
        except Exception:
            powered_end_idx = None

    events.update({
        'launch_idx': launch_idx,
        'powered_end_idx': powered_end_idx,
        'apogee_idx': apogee_idx,
        'landing_idx': landing_idx,
        'alt_col': alt_col,
        'vel_col': vel_col
    })
    return events


def _write_state_report(df: pd.DataFrame, events: dict, out_dir: Path):
    lines = []
    lines.append("Flight State Change Report")
    lines.append("="*60)
    # State counts and durations
    if 'state' in df.columns:
        states = df['state']
        counts = states.value_counts(dropna=True)
        dt = np.median(np.diff(df['t'])) if len(df) > 1 else _def_dt
        lines.append("State durations (approx):")
        for state, count in counts.items():
            duration = float(count) * float(dt)
            lines.append(f"- {state}: {count} samples, ~{duration:.1f}s")
    else:
        lines.append("No 'state' column found.")
    # Key events
    def fmt_time(idx):
        if idx is None:
            return "N/A"
        tval = df['t'].iloc[idx] if 0 <= idx < len(df) else np.nan
        if 'timestamp' in df.columns:
            ts = df['timestamp'].iloc[idx]
            return f"t={tval:.2f}s, ts={ts}"
        return f"t={tval:.2f}s"
    lines.append("")
    lines.append("Detected key events:")
    lines.append(f"- Launch:  {fmt_time(events.get('launch_idx'))}")
    lines.append(f"- Powered end: {fmt_time(events.get('powered_end_idx'))}")
    lines.append(f"- Apogee:  {fmt_time(events.get('apogee_idx'))}")
    lines.append(f"- Landing: {fmt_time(events.get('landing_idx'))}")
    # Write file
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'state_report.txt').write_text("\n".join(lines), encoding='utf-8')


def plot_flight_curves(df: pd.DataFrame, events: dict, out_dir: Path, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = df['t']
    alt_col = events['alt_col']
    vel_col = events['vel_col']

    # Altitude segmented
    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(t, df.get(alt_col, pd.Series(index=df.index)), color='#1f77b4', lw=2, label=f'{alt_col}')

    # Highlight segments
    li = events['launch_idx']
    pi = events['powered_end_idx']
    ai = events['apogee_idx']
    ei = events['landing_idx']

    def mark(idx, label):
        if idx is not None and 0 <= idx < len(df):
            ax.axvline(t.iloc[idx], color='k', ls='--', alpha=0.5)
            ymax = np.nanmax(df.get(alt_col, pd.Series([0])))
            ax.text(t.iloc[idx], ymax*0.9 if np.isfinite(ymax) else 0, label, rotation=90, va='top')

    mark(li, 'LAUNCH')
    mark(pi, 'POWERED END')
    mark(ai, 'APOGEE')
    mark(ei, 'LANDING')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Flight Altitude Curve - {title}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'altitude_segmented.png', dpi=300)
    plt.close()

    # Velocity
    if vel_col in df.columns:
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(t, df[vel_col], color='#ff7f0e', lw=2, label=f'{vel_col}')
        for idx, lbl in [(li,'LAUNCH'),(pi,'POWERED END'),(ai,'APOGEE'),(ei,'LANDING')]:
            if idx is not None and 0 <= idx < len(df):
                ax.axvline(t.iloc[idx], color='k', ls='--', alpha=0.5)
                ymax = np.nanmax(np.abs(df[vel_col]))
                ax.text(t.iloc[idx], (ymax*0.8) if np.isfinite(ymax) else 0, lbl, rotation=90, va='top')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Flight Velocity Curve - {title}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'velocity_segmented.png', dpi=300)
        plt.close()

    # Altitude vs Velocity phase plot
    if vel_col in df.columns and alt_col in df.columns:
        fig, ax = plt.subplots(figsize=(8,7))
        sc = ax.scatter(df[vel_col], df[alt_col], c=t, s=6, cmap='viridis')
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title(f'Altitude vs Velocity - {title}')
        cb = plt.colorbar(sc)
        cb.set_label('Time (s)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'altitude_vs_velocity.png', dpi=300)
        plt.close()



def main():
    parser = argparse.ArgumentParser(description='Isolate and plot flight curves from telemetry CSV')
    parser.add_argument('csv_file', nargs='?', help='CSV file path; defaults to latest in telemetry_logs')
    parser.add_argument('--out', default='flight_curves_plots', help='Output directory')
    args = parser.parse_args()

    # Determine file
    if args.csv_file:
        csv_path = Path(args.csv_file)
    else:
        logs = sorted(Path('telemetry_logs').glob('telemetry_*.csv'), key=lambda p: p.stat().st_mtime)
        if not logs:
            _p('No CSV files found in telemetry_logs')
            return
        csv_path = logs[-1]
        _p(f'Using latest CSV: {csv_path.name}')

    if not csv_path.exists():
        _p(f'File not found: {csv_path}')
        return

    df = load_csv(csv_path)
    events = detect_events(df)

    # If state column exists, print counts and write report
    out_dir = Path(args.out) / csv_path.stem
    if 'state' in df.columns:
        _p('State counts:')
        _p(str(df['state'].value_counts()))
        _write_state_report(df, events, out_dir)

    title = csv_path.stem
    plot_flight_curves(df, events, out_dir, title)

    _p(f'Flight curves saved in {out_dir}')

if __name__ == '__main__':
    main()
