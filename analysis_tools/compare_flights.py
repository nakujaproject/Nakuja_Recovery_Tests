#!/usr/bin/env python3
"""
Flight Data Comparison Tool
Compare multiple rocket flights for performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

class FlightComparator:
    def __init__(self):
        """Initialize flight comparator"""
        self.flights = {}
        self.comparison_metrics = {}
        
    def load_flight(self, csv_file, flight_name=None):
        """Load a flight CSV file"""
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"❌ File not found: {csv_file}")
            return False
        
        if not flight_name:
            flight_name = csv_path.stem
        
        try:
            df = pd.read_csv(csv_path)
            
            if len(df) == 0:
                # store empty df but mark
                df['time_elapsed'] = []
                self.flights[flight_name] = df
                print(f"⚠️ Loaded flight '{flight_name}': 0 records (skipping metrics)")
                return True
            
            # Convert timestamp
            if 'iso_timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['iso_timestamp'])
                df['time_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            else:
                df['time_elapsed'] = np.arange(len(df)) * 0.1
            
            # Convert numeric columns
            numeric_cols = ['agl_altitude', 'velocity', 'kalman_altitude', 'kalman_vertical_velocity']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.flights[flight_name] = df
            print(f"✅ Loaded flight '{flight_name}': {len(df)} records")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            return False
    
    def load_all_flights(self, directory="telemetry_logs"):
        """Load all CSV files from directory"""
        log_dir = Path(directory)
        if not log_dir.exists():
            print(f"❌ Directory not found: {directory}")
            return
        
        csv_files = list(log_dir.glob("telemetry_*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {directory}")
            return
        
        print(f"🔍 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            # Create readable flight name from filename
            flight_name = csv_file.stem.replace('telemetry_', 'Flight_')
            self.load_flight(csv_file, flight_name)
    
    def calculate_flight_metrics(self, df):
        """Calculate key metrics for a single flight"""
        metrics = {}
        total_records = len(df)
        metrics['total_records'] = total_records
        if total_records == 0:
            return metrics
        
        # Flight duration
        if 'time_elapsed' in df.columns:
            metrics['duration'] = df['time_elapsed'].max()
        
        # Altitude metrics
        for alt_col in ['agl_altitude', 'kalman_altitude', 'gps_altitude']:
            if alt_col in df.columns:
                alt_data = df[alt_col].dropna()
                if len(alt_data) > 0:
                    metrics[f'max_{alt_col}'] = alt_data.max()
                    # Time to apogee
                    apogee_idx = alt_data.idxmax()
                    if 'time_elapsed' in df.columns:
                        metrics[f'apogee_time_{alt_col}'] = df.loc[apogee_idx, 'time_elapsed']
        
        # Velocity metrics
        for vel_col in ['velocity', 'kalman_vertical_velocity']:
            if vel_col in df.columns:
                vel_data = df[vel_col].dropna()
                if len(vel_data) > 0:
                    metrics[f'max_{vel_col}'] = vel_data.max()
                    metrics[f'min_{vel_col}'] = vel_data.min()
        
        # Flight phases
        if 'state' in df.columns:
            phase_counts = df['state'].value_counts()
            for state, count in phase_counts.items():
                if pd.notna(state):
                    metrics[f'phase_{state}_duration'] = count * 0.1  # Assume 10Hz
        
        # Data quality metrics
        for col in ['agl_altitude', 'velocity', 'battery_voltage']:
            if col in df.columns:
                valid_data = df[col].dropna()
                metrics[f'{col}_completeness'] = (len(valid_data) / total_records * 100) if total_records else np.nan
        
        return metrics
    
    def compare_flights(self):
        """Compare all loaded flights"""
        if len(self.flights) < 2:
            print("❌ Need at least 2 flights to compare")
            return
        
        print(f"\n🔍 COMPARING {len(self.flights)} FLIGHTS")
        print("="*60)
        
        # Calculate metrics for each flight
        all_metrics = {}
        for flight_name, df in self.flights.items():
            all_metrics[flight_name] = self.calculate_flight_metrics(df)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics).T
        
        # Drop flights with zero records to avoid NaNs in plots
        if 'total_records' in comparison_df.columns:
            non_empty = comparison_df['total_records'].fillna(0) > 0
            if non_empty.any():
                self.comparison_metrics = comparison_df[non_empty]
            else:
                self.comparison_metrics = comparison_df
        else:
            self.comparison_metrics = comparison_df
        
        # Key metrics to compare
        key_metrics = [
            'duration', 'max_agl_altitude', 'max_kalman_altitude',
            'max_velocity', 'max_kalman_vertical_velocity',
            'total_records', 'agl_altitude_completeness'
        ]
        
        print("\n📊 KEY PERFORMANCE METRICS")
        print("-" * 40)
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                print(f"\n🎯 {metric.replace('_', ' ').title()}:")
                
                for flight in comparison_df.index:
                    value = comparison_df.loc[flight, metric]
                    if pd.notna(value):
                        if 'altitude' in metric or 'velocity' in metric:
                            print(f"   {flight}: {value:.2f}")
                        elif 'completeness' in metric:
                            print(f"   {flight}: {value:.1f}%")
                        elif 'duration' in metric:
                            print(f"   {flight}: {value:.1f}s")
                        else:
                            print(f"   {flight}: {value}")
                
                # Show best and worst
                if comparison_df[metric].notna().sum() > 1:
                    best_flight = comparison_df[metric].idxmax()
                    worst_flight = comparison_df[metric].idxmin()
                    
                    if best_flight != worst_flight:
                        print(f"   🏆 Best: {best_flight}")
                        print(f"   ⚠️  Worst: {worst_flight}")
        
        self.comparison_metrics = comparison_df
        return comparison_df
    
    def create_comparison_plots(self, output_dir="comparison_plots"):
        """Create comparison visualizations"""
        if len(self.flights) < 2:
            print("❌ Need at least 2 flights for plots")
            return
        
        plot_dir = Path(output_dir)
        plot_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.flights)))
        
        # 1. Altitude comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, (flight_name, df) in enumerate(self.flights.items()):
            # Use best available altitude
            alt_col = None
            for col in ['kalman_altitude', 'agl_altitude', 'gps_altitude']:
                if col in df.columns and df[col].notna().sum() > 0:
                    alt_col = col
                    break
            
            if alt_col and 'time_elapsed' in df.columns:
                time_data = df['time_elapsed']
                alt_data = df[alt_col]
                
                # Filter out obvious outliers
                valid_mask = (alt_data >= -100) & (alt_data <= 10000)  # Reasonable altitude range
                
                ax.plot(time_data[valid_mask], alt_data[valid_mask], 
                       color=colors[i], linewidth=2, label=flight_name, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Flight Altitude Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'altitude_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Velocity comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, (flight_name, df) in enumerate(self.flights.items()):
            vel_col = None
            for col in ['kalman_vertical_velocity', 'velocity']:
                if col in df.columns and df[col].notna().sum() > 0:
                    vel_col = col
                    break
            
            if vel_col and 'time_elapsed' in df.columns:
                time_data = df['time_elapsed']
                vel_data = df[vel_col]
                
                # Filter outliers
                valid_mask = (vel_data >= -200) & (vel_data <= 200)  # Reasonable velocity range
                
                ax.plot(time_data[valid_mask], vel_data[valid_mask], 
                       color=colors[i], linewidth=2, label=flight_name, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Flight Velocity Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'velocity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance metrics bar chart
        if hasattr(self, 'comparison_metrics') and len(self.comparison_metrics) > 0:
            metrics_to_plot = ['max_agl_altitude', 'max_velocity', 'duration']
            available_metrics = [m for m in metrics_to_plot if m in self.comparison_metrics.columns]
            
            if available_metrics:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
                if len(available_metrics) == 1:
                    axes = [axes]
                
                for i, metric in enumerate(available_metrics):
                    data = self.comparison_metrics[metric].dropna()
                    
                    bars = axes[i].bar(range(len(data)), data.values, color=colors[:len(data)])
                    axes[i].set_xticks(range(len(data)))
                    axes[i].set_xticklabels(data.index, rotation=45, ha='right')
                    axes[i].set_title(metric.replace('_', ' ').title())
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(plot_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📊 Comparison plots saved to {plot_dir}/")
    
    def generate_comparison_report(self, output_file="flight_comparison_report.txt"):
        """Generate detailed comparison report"""
        if len(self.flights) < 2:
            print("❌ Need at least 2 flights for report")
            return
        
        with open(output_file, 'w') as f:
            f.write("N4 ROCKET FLIGHT COMPARISON REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Flights Analyzed: {len(self.flights)}\n\n")
            
            # Flight summary
            f.write("FLIGHT SUMMARY\n")
            f.write("-"*30 + "\n")
            for flight_name, df in self.flights.items():
                f.write(f"{flight_name}:\n")
                f.write(f"  Records: {len(df)}\n")
                if 'time_elapsed' in df.columns:
                    f.write(f"  Duration: {df['time_elapsed'].max():.1f}s\n")
                f.write("\n")
            
            # Detailed metrics comparison
            if hasattr(self, 'comparison_metrics'):
                f.write("DETAILED METRICS COMPARISON\n")
                f.write("-"*40 + "\n")
                f.write(self.comparison_metrics.to_string())
        
        print(f"📄 Comparison report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare multiple rocket flights')
    parser.add_argument('--directory', '-d', default='telemetry_logs', 
                       help='Directory containing CSV files')
    parser.add_argument('--files', '-f', nargs='+', 
                       help='Specific CSV files to compare')
    parser.add_argument('--plots', action='store_true', 
                       help='Generate comparison plots')
    parser.add_argument('--report', action='store_true', 
                       help='Generate comparison report')
    
    args = parser.parse_args()
    
    comparator = FlightComparator()
    
    if args.files:
        # Load specific files
        for csv_file in args.files:
            comparator.load_flight(csv_file)
    else:
        # Load all files from directory
        comparator.load_all_flights(args.directory)
    
    if len(comparator.flights) == 0:
        print("❌ No flight data loaded")
        return
    
    print(f"🚀 N4 FLIGHT COMPARISON ANALYSIS")
    
    comparison_df = comparator.compare_flights()
    
    if args.plots:
        comparator.create_comparison_plots()
    
    if args.report:
        comparator.generate_comparison_report()
    
    print("\n✅ Flight comparison complete!")
    print("💡 Run with --plots to generate visualizations")
    print("💡 Run with --report to generate detailed report")

if __name__ == "__main__":
    main()
