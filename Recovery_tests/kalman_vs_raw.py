#!/usr/bin/env python3
"""
Kalman vs Raw Data Analyzer
- Plots raw sensor signals vs Kalman-filtered outputs
- Provides residual/error plots and basic statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

plt.style.use('seaborn-v0_8')
sns.set_palette('tab10')

# Safe print helper for Windows terminals
def _p(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(str(msg).encode('ascii', 'ignore').decode())
        except Exception:
            print(str(msg))

PAIRS = [
    # (filtered, raw)
    ('kalman_altitude', 'agl_altitude'),
    ('kalman_vertical_velocity', 'velocity')
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'iso_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['iso_timestamp'])
        df['t'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    else:
        df['t'] = np.arange(len(df)) * 0.1
    for c in set(sum(([f, r] for f, r in PAIRS), [])):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def plot_pair(df: pd.DataFrame, filtered: str, raw: str, out_dir: Path, unit: str, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = df['t']

    # Time series overlay
    fig, ax = plt.subplots(figsize=(12,7))
    if raw in df.columns:
        ax.plot(t, df[raw], color='#1f77b4', alpha=0.5, label=f'{raw} (raw)')
    if filtered in df.columns:
        ax.plot(t, df[filtered], color='#ff7f0e', lw=2, label=f'{filtered} (kalman)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(unit)
    ax.set_title(f'{title} - Time Series')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{filtered}_vs_{raw}_timeseries.png', dpi=300)
    plt.close()

    # Residuals
    if filtered in df.columns and raw in df.columns:
        residual = df[filtered] - df[raw]
        fig, axes = plt.subplots(2, 1, figsize=(12,9), sharex=False)
        # Residual over time
        axes[0].plot(t, residual, color='purple', alpha=0.8)
        axes[0].axhline(0, color='k', ls='--', alpha=0.5)
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals Over Time (Kalman - Raw)')
        axes[0].grid(True, alpha=0.3)
        # Histogram
        axes[1].hist(residual.dropna(), bins=50, color='purple', alpha=0.7)
        mu = np.nanmean(residual)
        sigma = np.nanstd(residual)
        axes[1].axvline(mu, color='k', ls='--', label=f'μ={mu:.3f}, σ={sigma:.3f}')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Residual Distribution')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'{filtered}_vs_{raw}_residuals.png', dpi=300)
        plt.close()

        # Scatter
        fig, ax = plt.subplots(figsize=(7,7))
        ax.scatter(df[raw], df[filtered], s=6, alpha=0.4)
        lims = [np.nanmin([ax.get_xlim()[0], ax.get_ylim()[0]]), np.nanmax([ax.get_xlim()[1], ax.get_ylim()[1]])]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'{raw} (raw)')
        ax.set_ylabel(f'{filtered} (kalman)')
        ax.set_title(f'{title} - Kalman vs Raw Scatter')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f'{filtered}_vs_{raw}_scatter.png', dpi=300)
        plt.close()

        # Stats summary (robust correlation)
        valid = df[[raw, filtered]].dropna()
        if len(valid) > 3 and valid[raw].std() > 0 and valid[filtered].std() > 0:
            cor = valid[raw].corr(valid[filtered])
        else:
            cor = np.nan
        summary = [
            f'{filtered} vs {raw}',
            f'Correlation: {cor:.3f}' if np.isfinite(cor) else 'Correlation: N/A',
            f'Residual mean: {mu:.3f}',
            f'Residual std: {sigma:.3f}'
        ]
        (out_dir / f'{filtered}_vs_{raw}_summary.txt').write_text("\n".join(summary), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Plot Kalman vs raw data')
    parser.add_argument('csv_file', nargs='?', help='CSV file to analyze (default: latest)')
    parser.add_argument('--out', default='kalman_plots', help='Output directory')
    args = parser.parse_args()

    # Determine file
    if args.csv_file:
        csv_path = Path(args.csv_file)
    else:
        logs = sorted(Path('telemetry_logs').glob('telemetry_*.csv'), key=lambda p: p.stat().st_mtime)
        if not logs:
            _p('ERROR: No CSV files found')
            return
        csv_path = logs[-1]
        _p(f'Using latest CSV: {csv_path.name}')

    if not csv_path.exists():
        _p(f'ERROR: File not found: {csv_path}')
        return

    df = load_csv(csv_path)
    out_dir = Path(args.out) / csv_path.stem

    # Altitude
    plot_pair(df, 'kalman_altitude', 'agl_altitude', out_dir, 'Altitude (m)', 'Altitude Kalman vs Raw')
    # Velocity
    plot_pair(df, 'kalman_vertical_velocity', 'velocity', out_dir, 'Velocity (m/s)', 'Velocity Kalman vs Raw')

    _p(f'Kalman vs Raw plots saved in {out_dir}')

if __name__ == '__main__':
    main()
