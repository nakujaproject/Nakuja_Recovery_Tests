#!/usr/bin/env python3
"""
Comprehensive Telemetry Data Analysis Tool
Analyzes rocket telemetry CSV files for flight performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

class TelemetryAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        self.csv_file = Path(csv_file_path)
        self.df = None
        self.flight_phases = {}
        self.disable_emojis = False
        
    def load_data(self):
        """Load and clean CSV data"""
        try:
            print(f"Loading data from: {self.csv_file.name}")
            self.df = pd.read_csv(self.csv_file)
            
            # Convert timestamp to datetime
            if 'iso_timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['iso_timestamp'])
                
            # Convert numeric columns
            numeric_cols = [
                'agl_altitude', 'gps_altitude', 'kalman_altitude',
                'velocity', 'kalman_vertical_velocity', 
                'acceleration_x', 'acceleration_y', 'acceleration_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'mag_x', 'mag_y', 'mag_z',
                'pressure', 'temperature', 'battery_voltage'
            ]
            
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            print(f"Loaded {len(self.df)} records")
            if 'timestamp' in self.df.columns:
                print(f"Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _p(self, msg):
        """Safe print without emojis for Windows cp1252 terminals"""
        try:
            print(msg)
        except UnicodeEncodeError:
            self.disable_emojis = True
            # strip non-ASCII
            print(msg.encode('ascii', 'ignore').decode())
    
    def basic_stats(self):
        """Generate basic statistics"""
        self._p("\n" + "="*60)
        self._p("BASIC STATISTICS")
        self._p("="*60)
        
        # Flight duration
        if 'timestamp' in self.df.columns:
            duration = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds()
            self._p(f"Flight Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Data completeness
        total_records = len(self.df)
        self._p(f"Total Records: {total_records}")
        
        # Key metrics
        key_metrics = {
            'Max AGL Altitude': 'agl_altitude',
            'Max GPS Altitude': 'gps_altitude', 
            'Max Kalman Altitude': 'kalman_altitude',
            'Max Velocity': 'velocity',
            'Max Kalman Velocity': 'kalman_vertical_velocity',
            'Min Battery Voltage': 'battery_voltage',
            'Max Temperature': 'temperature'
        }
        
        for metric_name, col_name in key_metrics.items():
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                if len(values) > 0:
                    value = values.min() if 'Min' in metric_name else values.max()
                    self._p(f"{metric_name}: {value:.2f}")
        
        # State analysis
        if 'state' in self.df.columns:
            self._p(f"\nFlight States:")
            state_counts = self.df['state'].value_counts()
            for state, count in state_counts.items():
                percentage = (count / total_records) * 100
                self._p(f"   {state}: {count} records ({percentage:.1f}%)")

    def flight_phase_analysis(self):
        """Analyze different flight phases"""
        self._p("\n" + "="*60)
        self._p("FLIGHT PHASE ANALYSIS")
        self._p("="*60)
        
        if 'state' not in self.df.columns:
            self._p("No state column found for phase analysis")
            return
        
        # Define phase transitions
        phases = {
            'Pre-Flight': ['IDLE', 'READY', 0],
            'Powered Flight': ['POWERED_FLIGHT', 'BOOST'],
            'Coasting': ['COASTING', 'UNPOWERED_FLIGHT'],
            'Apogee': ['APOGEE', 1],
            'Descent': ['DROGUE_DESCENT', 'MAIN_DESCENT', 'DESCENT'],
            'Recovery': ['LANDED', 'RECOVERY']
        }
        
        for phase_name, states in phases.items():
            phase_data = self.df[self.df['state'].isin(states)] if 'state' in self.df.columns else pd.DataFrame()
            if len(phase_data) > 0:
                duration = len(phase_data)  # Number of records
                
                alt_col = 'agl_altitude' if 'agl_altitude' in phase_data.columns else None
                if alt_col:
                    alt_data = phase_data[alt_col].dropna()
                    if len(alt_data) > 0:
                        self._p(f"\n{phase_name}:")
                        self._p(f"   Duration: {duration} records")
                        self._p(f"   Altitude Range: {alt_data.min():.1f}m - {alt_data.max():.1f}m")
                        
                        if 'velocity' in phase_data.columns:
                            vel_data = phase_data['velocity'].dropna()
                            if len(vel_data) > 0:
                                self._p(f"   Velocity Range: {vel_data.min():.1f} - {vel_data.max():.1f} m/s")

    def sensor_health_check(self):
        """Check sensor data health and completeness"""
        self._p("\n" + "="*60)
        self._p("SENSOR HEALTH CHECK")
        self._p("="*60)
        
        sensor_groups = {
            'Altitude Sensors': ['agl_altitude', 'gps_altitude', 'kalman_altitude'],
            'Velocity Sensors': ['velocity', 'kalman_vertical_velocity'],
            'Accelerometer': ['acceleration_x', 'acceleration_y', 'acceleration_z'],
            'Gyroscope': ['gyro_x', 'gyro_y', 'gyro_z'],
            'Magnetometer': ['mag_x', 'mag_y', 'mag_z'],
            'Environmental': ['pressure', 'temperature'],
            'Power': ['battery_voltage'],
            'Communication': ['wifi_rssi', 'lora_rssi']
        }
        
        for group_name, sensors in sensor_groups.items():
            self._p(f"\n{group_name}:")
            
            for sensor in sensors:
                if sensor in self.df.columns:
                    data = self.df[sensor]
                    total_count = len(data)
                    valid_count = data.dropna().count()
                    zero_count = (data == 0).sum()
                    
                    completeness = (valid_count / total_count) * 100 if total_count else 0
                    zero_percentage = (zero_count / total_count) * 100 if total_count else 0
                    
                    status = "OK" if completeness > 90 else "WARN" if completeness > 50 else "BAD"
                    
                    self._p(f"   {status} {sensor}: {completeness:.1f}% valid, {zero_percentage:.1f}% zeros")
                else:
                    self._p(f"   MISSING {sensor}")

    def create_plots(self, output_dir="analysis_plots"):
        """Generate comprehensive plots"""
        self._p(f"Generating plots in {output_dir}/")
        
        # Create output directory
        plot_dir = Path(output_dir)
        plot_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Altitude vs Time
        if any(col in self.df.columns for col in ['agl_altitude', 'gps_altitude', 'kalman_altitude']):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if 'timestamp' in self.df.columns:
                time_data = (self.df['timestamp'] - self.df['timestamp'].min()).dt.total_seconds()
            else:
                time_data = range(len(self.df))
            
            if 'agl_altitude' in self.df.columns:
                ax.plot(time_data, self.df['agl_altitude'], label='AGL Altitude', linewidth=2)
            if 'gps_altitude' in self.df.columns:
                ax.plot(time_data, self.df['gps_altitude'], label='GPS Altitude', alpha=0.7)
            if 'kalman_altitude' in self.df.columns:
                ax.plot(time_data, self.df['kalman_altitude'], label='Kalman Altitude', alpha=0.7)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Altitude (m)')
            ax.set_title('Rocket Altitude Profile')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'altitude_profile.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Velocity vs Time
        if any(col in self.df.columns for col in ['velocity', 'kalman_vertical_velocity']):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if 'velocity' in self.df.columns:
                ax.plot(time_data, self.df['velocity'], label='Barometric Velocity', linewidth=2)
            if 'kalman_vertical_velocity' in self.df.columns:
                ax.plot(time_data, self.df['kalman_vertical_velocity'], label='Kalman Velocity', alpha=0.7)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title('Rocket Velocity Profile')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'velocity_profile.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 3D Acceleration
        if all(col in self.df.columns for col in ['acceleration_x', 'acceleration_y', 'acceleration_z']):
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            axes[0].plot(time_data, self.df['acceleration_x'], 'r-', label='X-axis')
            axes[0].set_ylabel('Accel X (m/s²)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_data, self.df['acceleration_y'], 'g-', label='Y-axis')
            axes[1].set_ylabel('Accel Y (m/s²)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(time_data, self.df['acceleration_z'], 'b-', label='Z-axis')
            axes[2].set_ylabel('Accel Z (m/s²)')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle('3-Axis Acceleration Profile')
            plt.tight_layout()
            plt.savefig(plot_dir / 'acceleration_profile.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. System Health Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Battery voltage
        if 'battery_voltage' in self.df.columns:
            axes[0,0].plot(time_data, self.df['battery_voltage'], 'orange', linewidth=2)
            axes[0,0].set_title('Battery Voltage')
            axes[0,0].set_ylabel('Voltage (V)')
            axes[0,0].grid(True, alpha=0.3)
        
        # Temperature
        if 'temperature' in self.df.columns:
            axes[0,1].plot(time_data, self.df['temperature'], 'red', linewidth=2)
            axes[0,1].set_title('Temperature')
            axes[0,1].set_ylabel('Temperature (°C)')
            axes[0,1].grid(True, alpha=0.3)
        
        # RSSI
        if 'wifi_rssi' in self.df.columns:
            axes[1,0].plot(time_data, self.df['wifi_rssi'], 'blue', linewidth=2)
            axes[1,0].set_title('WiFi Signal Strength')
            axes[1,0].set_ylabel('RSSI (dBm)')
            axes[1,0].set_xlabel('Time (seconds)')
            axes[1,0].grid(True, alpha=0.3)
        
        # Pressure
        if 'pressure' in self.df.columns:
            axes[1,1].plot(time_data, self.df['pressure'], 'purple', linewidth=2)
            axes[1,1].set_title('Atmospheric Pressure')
            axes[1,1].set_ylabel('Pressure (Pa)')
            axes[1,1].set_xlabel('Time (seconds)')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'system_health.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self._p(f"Plots saved to {Path(output_dir)}/")
    
    def generate_report(self, output_file="flight_analysis_report.txt"):
        """Generate comprehensive text report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("N4 ROCKET FLIGHT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.csv_file.name}\n\n")
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                self.basic_stats()
                self.flight_phase_analysis()
                self.sensor_health_check()
            finally:
                sys.stdout = original_stdout
        print(f"Report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze rocket telemetry CSV data')
    parser.add_argument('csv_file', nargs='?', help='Path to CSV file (optional)')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    
    args = parser.parse_args()
    
    # Find CSV file
    if args.csv_file:
        csv_file = Path(args.csv_file)
    else:
        # Find latest CSV in telemetry_logs
        log_dir = Path("telemetry_logs")
        if log_dir.exists():
            csv_files = list(log_dir.glob("telemetry_*.csv"))
            if csv_files:
                csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                print(f"Using latest CSV: {csv_file.name}")
            else:
                print("No CSV files found in telemetry_logs/")
                return
        else:
            print("telemetry_logs directory not found")
            return
    
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return
    
    # Analyze data
    analyzer = TelemetryAnalyzer(csv_file)
    
    if not analyzer.load_data():
        return
    
    print("N4 TELEMETRY ANALYSIS")
    analyzer.basic_stats()
    analyzer.flight_phase_analysis()
    analyzer.sensor_health_check()
    
    if args.plots:
        analyzer.create_plots()
    
    if args.report:
        analyzer.generate_report()
    
    print(f"\nAnalysis complete!")
    print(f"Run with --plots to generate visualizations")
    print(f"Run with --report to generate a text report")

if __name__ == "__main__":
    main()
