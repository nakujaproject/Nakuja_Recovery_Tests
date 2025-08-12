#!/usr/bin/env python3
"""
Rocket Performance Metrics Calculator
Calculates key flight performance metrics from telemetry data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import argparse

# Safe print helper for Windows terminals
def _p(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(str(msg).encode('ascii', 'ignore').decode())
        except Exception:
            print(str(msg))

class PerformanceCalculator:
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        self.csv_file = Path(csv_file_path)
        self.df = None
        self.metrics = {}
        
    def load_data(self):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv(self.csv_file)
            
            # Convert timestamp
            if 'iso_timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['iso_timestamp'])
                self.df['time_elapsed'] = (self.df['timestamp'] - self.df['timestamp'].min()).dt.total_seconds()
            else:
                self.df['time_elapsed'] = np.arange(len(self.df)) * 0.1  # Assume 10Hz
                
            # Convert numeric columns
            numeric_cols = ['agl_altitude', 'velocity', 'kalman_altitude', 'kalman_vertical_velocity',
                          'acceleration_x', 'acceleration_y', 'acceleration_z']
            
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            return True
        except Exception as e:
            _p(f"Error loading data: {e}")
            return False
    
    def calculate_apogee_metrics(self):
        """Calculate apogee-related metrics"""
        _p("\nAPOGEE ANALYSIS")
        _p("-" * 40)
        
        # Use best available altitude data
        alt_col = None
        for col in ['kalman_altitude', 'agl_altitude', 'gps_altitude']:
            if col in self.df.columns and self.df[col].notna().sum() > 0:
                alt_col = col
                break
        
        if not alt_col:
            _p("No altitude data available")
            return
        
        altitude_data = self.df[alt_col].dropna()
        time_data = self.df.loc[altitude_data.index, 'time_elapsed']
        
        # Find apogee
        max_altitude = altitude_data.max()
        apogee_idx = altitude_data.idxmax()
        apogee_time = time_data.loc[apogee_idx]
        
        self.metrics['max_altitude'] = max_altitude
        self.metrics['apogee_time'] = apogee_time
        
        _p(f"Maximum Altitude: {max_altitude:.2f} m")
        _p(f"Time to Apogee: {apogee_time:.2f} seconds")
        
        # Calculate ascent rate near apogee
        apogee_window = 5  # seconds around apogee
        window_mask = (time_data >= apogee_time - apogee_window) & (time_data <= apogee_time + apogee_window)
        window_alt = altitude_data[window_mask]
        window_time = time_data[window_mask]
        
        if len(window_alt) > 1:
            ascent_rate = np.gradient(window_alt, window_time)
            _p(f"Ascent Rate (±{apogee_window}s): {ascent_rate.mean():.2f} m/s")
    
    def calculate_velocity_metrics(self):
        """Calculate velocity-related metrics"""
        _p("\nVELOCITY ANALYSIS")
        _p("-" * 40)
        
        # Use best available velocity data
        vel_col = None
        for col in ['kalman_vertical_velocity', 'velocity']:
            if col in self.df.columns and self.df[col].notna().sum() > 0:
                vel_col = col
                break
        
        if not vel_col:
            _p("No velocity data available")
            return
        
        velocity_data = self.df[vel_col].dropna()
        time_data = self.df.loc[velocity_data.index, 'time_elapsed']
        
        # Find maximum velocity
        max_velocity = velocity_data.max()
        min_velocity = velocity_data.min()
        max_vel_time = time_data.loc[velocity_data.idxmax()]
        
        self.metrics['max_velocity'] = max_velocity
        self.metrics['min_velocity'] = min_velocity
        self.metrics['max_velocity_time'] = max_vel_time
        
        _p(f"Maximum Velocity: {max_velocity:.2f} m/s")
        _p(f"Minimum Velocity: {min_velocity:.2f} m/s")
        _p(f"Time to Max Velocity: {max_vel_time:.2f} seconds")
        
        # Calculate acceleration from velocity
        if len(velocity_data) > 2:
            acceleration = np.gradient(velocity_data, time_data)
            max_accel = acceleration.max()
            max_decel = acceleration.min()
            
            self.metrics['max_acceleration'] = max_accel
            self.metrics['max_deceleration'] = max_decel
            
            _p(f"Maximum Acceleration: {max_accel:.2f} m/s^2")
            _p(f"Maximum Deceleration: {max_decel:.2f} m/s^2")
    
    def analyze_flight_phases(self):
        """Analyze different flight phases"""
        _p("\nFLIGHT PHASE ANALYSIS")
        _p("-" * 40)
        
        if 'state' not in self.df.columns:
            _p("No state data available")
            return
        
        # Calculate phase durations
        phase_stats = {}
        
        for state in self.df['state'].unique():
            if pd.isna(state):
                continue
                
            state_data = self.df[self.df['state'] == state]
            duration = len(state_data) * 0.1  # Assume 10Hz sampling
            
            phase_stats[state] = {
                'duration': duration,
                'records': len(state_data),
                'start_time': state_data['time_elapsed'].min() if 'time_elapsed' in state_data else 0,
                'end_time': state_data['time_elapsed'].max() if 'time_elapsed' in state_data else 0
            }
        
        # Sort by start time
        sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1]['start_time'])
        
        for state, stats in sorted_phases:
            _p(f"{state}:")
            _p(f"   Duration: {stats['duration']:.1f}s ({stats['records']} records)")
            _p(f"   Time: {stats['start_time']:.1f}s - {stats['end_time']:.1f}s")
        
        self.metrics['phase_stats'] = phase_stats
    
    def calculate_g_forces(self):
        """Calculate G-force analysis"""
        _p("\nG-FORCE ANALYSIS")
        _p("-" * 40)
        
        # Check for accelerometer data
        accel_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        available_cols = [col for col in accel_cols if col in self.df.columns]
        
        if not available_cols:
            _p("No accelerometer data available")
            return
        
        # Calculate total acceleration magnitude
        accel_data = self.df[available_cols].dropna()
        
        if len(accel_data) == 0:
            _p("No valid accelerometer data")
            return
        
        # Calculate G-forces (divide by 9.81 m/s^2)
        g_force_magnitude = np.sqrt(
            (accel_data[available_cols] ** 2).sum(axis=1)
        ) / 9.81
        
        max_g_force = g_force_magnitude.max()
        avg_g_force = g_force_magnitude.mean()
        
        self.metrics['max_g_force'] = max_g_force
        self.metrics['avg_g_force'] = avg_g_force
        
        _p(f"Maximum G-Force: {max_g_force:.2f} G")
        _p(f"Average G-Force: {avg_g_force:.2f} G")
        
        # Analysis by axis
        for col in available_cols:
            axis_g = self.df[col].dropna() / 9.81
            if len(axis_g) > 0:
                axis_name = col.split('_')[1].upper()
                _p(f"{axis_name}-axis G-Force: max={axis_g.max():.2f}G, min={axis_g.min():.2f}G")
    
    def estimate_drag_coefficient(self):
        """Estimate drag coefficient during descent"""
        _p("\nDRAG ANALYSIS")
        _p("-" * 40)
        
        # Find descent phase data
        if 'state' not in self.df.columns:
            _p("No state data for descent analysis")
            return
        
        # Look for descent states
        descent_states = ['DROGUE_DESCENT', 'MAIN_DESCENT', 'DESCENT']
        descent_data = self.df[self.df['state'].isin(descent_states)]
        
        if len(descent_data) < 10:
            _p("Insufficient descent data")
            return
        
        # Use velocity data during descent
        vel_col = None
        for col in ['kalman_vertical_velocity', 'velocity']:
            if col in descent_data.columns and descent_data[col].notna().sum() > 5:
                vel_col = col
                break
        
        if not vel_col:
            _p("No velocity data during descent")
            return
        
        descent_velocity = descent_data[vel_col].dropna()
        descent_time = descent_data.loc[descent_velocity.index, 'time_elapsed']
        
        # Calculate terminal velocity (average of latter part of descent)
        if len(descent_velocity) > 10:
            terminal_velocity = abs(descent_velocity.iloc[-len(descent_velocity)//3:].mean())
            self.metrics['terminal_velocity'] = terminal_velocity
            _p(f"Terminal Velocity: {terminal_velocity:.2f} m/s")
            
            # Estimate drag coefficient (simplified calculation)
            # Assumes: Cd = 2*m*g / (ρ*A*v^2)
            mass = 1.0  # kg (estimate)
            area = 0.01  # m^2 (estimate)
            air_density = 1.225  # kg/m^3 at sea level
            
            if terminal_velocity > 0:
                drag_coeff = (2 * mass * 9.81) / (air_density * area * terminal_velocity**2)
                self.metrics['estimated_drag_coefficient'] = drag_coeff
                _p(f"Estimated Drag Coefficient: {drag_coeff:.2f}")
                _p("   Note: Assumes 1kg mass, 0.01m^2 area")
    
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        _p("\n" + "="*60)
        _p("PERFORMANCE SUMMARY")
        _p("="*60)
        
        summary = []
        
        # Key metrics
        key_metrics = [
            ('Maximum Altitude', 'max_altitude', 'm'),
            ('Time to Apogee', 'apogee_time', 's'),
            ('Maximum Velocity', 'max_velocity', 'm/s'),
            ('Maximum G-Force', 'max_g_force', 'G'),
            ('Terminal Velocity', 'terminal_velocity', 'm/s'),
        ]
        
        for name, key, unit in key_metrics:
            if key in self.metrics:
                value = self.metrics[key]
                _p(f"{name}: {value:.2f} {unit}")
                summary.append(f"{name}: {value:.2f} {unit}")
        
        # Save summary to file
        summary_file = f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("N4 ROCKET PERFORMANCE SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.csv_file.name}\n\n")
            
            for line in summary:
                f.write(line + "\n")
        
        _p(f"\nSummary saved to: {summary_file}")
        
        return self.metrics
    
    def create_performance_plots(self, output_dir="performance_plots"):
        """Create performance-focused plots"""
        plot_dir = Path(output_dir)
        plot_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Altitude-Velocity trajectory
        if all(col in self.df.columns for col in ['agl_altitude', 'velocity']):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color by time
            time_data = self.df['time_elapsed']
            scatter = ax.scatter(self.df['velocity'], self.df['agl_altitude'], 
                               c=time_data, cmap='viridis', alpha=0.6, s=20)
            
            ax.set_xlabel('Velocity (m/s)')
            ax.set_ylabel('Altitude (m)')
            ax.set_title('Altitude vs Velocity Trajectory')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Time (seconds)')
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'altitude_velocity_trajectory.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. G-Force profile
        accel_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        available_cols = [col for col in accel_cols if col in self.df.columns]
        
        if available_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate G-force magnitude
            accel_data = self.df[available_cols].fillna(0)
            g_force = np.sqrt((accel_data ** 2).sum(axis=1)) / 9.81
            
            ax.plot(self.df['time_elapsed'], g_force, 'red', linewidth=2, label='Total G-Force')
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1G Reference')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('G-Force')
            ax.set_title('G-Force Profile During Flight')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'g_force_profile.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        _p(f"Performance plots saved to {plot_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Calculate rocket performance metrics')
    parser.add_argument('csv_file', nargs='?', help='Path to CSV file (optional)')
    parser.add_argument('--plots', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    # Find CSV file
    if args.csv_file:
        csv_file = Path(args.csv_file)
    else:
        log_dir = Path("telemetry_logs")
        if log_dir.exists():
            csv_files = list(log_dir.glob("telemetry_*.csv"))
            if csv_files:
                csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                _p(f"Using latest CSV: {csv_file.name}")
            else:
                _p("No CSV files found")
                return
        else:
            _p("telemetry_logs directory not found")
            return
    
    if not csv_file.exists():
        _p(f"File not found: {csv_file}")
        return
    
    # Calculate performance metrics
    calculator = PerformanceCalculator(csv_file)
    
    if not calculator.load_data():
        return
    
    _p("N4 ROCKET PERFORMANCE ANALYSIS")
    _p(f"Data file: {csv_file.name}")
    _p(f"Total records: {len(calculator.df)}")
    
    calculator.calculate_apogee_metrics()
    calculator.calculate_velocity_metrics()
    calculator.analyze_flight_phases()
    calculator.calculate_g_forces()
    calculator.estimate_drag_coefficient()
    
    metrics = calculator.generate_performance_summary()
    
    if args.plots:
        calculator.create_performance_plots()
    
    _p("\nPerformance analysis complete!")

if __name__ == "__main__":
    from datetime import datetime
    main()
