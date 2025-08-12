#!/usr/bin/env python3
"""
Telemetry Data Quality Checker
Quick assessment of CSV data quality and issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

class DataQualityChecker:
    def __init__(self, csv_file_path):
        """Initialize with CSV file"""
        self.csv_file = Path(csv_file_path)
        self.df = None
        self.issues = []
        
    def load_data(self):
        """Load CSV data"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df)} records from {self.csv_file.name}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def check_basic_structure(self):
        """Check basic data structure"""
        print("\n📊 DATA STRUCTURE CHECK")
        print("-" * 40)
        
        # Check if file is empty
        if len(self.df) == 0:
            self.issues.append("❌ File is empty")
            print("❌ File is empty")
            return
        
        print(f"✅ Records: {len(self.df)}")
        print(f"✅ Columns: {len(self.df.columns)}")
        
        # Expected columns
        expected_cols = [
            'iso_timestamp', 'record_number', 'state', 
            'agl_altitude', 'velocity', 'battery_voltage'
        ]
        
        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        if missing_cols:
            self.issues.append(f"❌ Missing columns: {missing_cols}")
            print(f"❌ Missing expected columns: {missing_cols}")
        else:
            print("✅ All expected columns present")
    
    def check_data_completeness(self):
        """Check data completeness"""
        print("\n📈 DATA COMPLETENESS")
        print("-" * 40)
        
        total_records = len(self.df)
        
        # Key columns to check
        key_columns = [
            'agl_altitude', 'velocity', 'kalman_altitude', 'kalman_vertical_velocity',
            'battery_voltage', 'state', 'pressure', 'temperature'
        ]
        
        for col in key_columns:
            if col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                zero_count = (self.df[col] == 0).sum()
                valid_count = total_records - null_count
                
                completeness = (valid_count / total_records) * 100
                zero_percentage = (zero_count / total_records) * 100
                
                # Status indicators
                if completeness >= 95:
                    status = "✅"
                elif completeness >= 80:
                    status = "⚠️"
                else:
                    status = "❌"
                    self.issues.append(f"Low completeness in {col}: {completeness:.1f}%")
                
                print(f"{status} {col}: {completeness:.1f}% complete, {zero_percentage:.1f}% zeros")
            else:
                print(f"❌ {col}: Column not found")
    
    def check_data_ranges(self):
        """Check for reasonable data ranges"""
        print("\n🎯 DATA RANGE CHECK")
        print("-" * 40)
        
        # Define reasonable ranges for rocket telemetry
        expected_ranges = {
            'agl_altitude': (-50, 5000),  # meters
            'gps_altitude': (0, 5000),    # meters
            'velocity': (-200, 200),      # m/s
            'acceleration_x': (-100, 100), # m/s²
            'acceleration_y': (-100, 100), # m/s²
            'acceleration_z': (-100, 100), # m/s²
            'battery_voltage': (3.0, 12.0), # volts
            'temperature': (-20, 80),     # celsius
            'pressure': (50000, 110000),  # pascals
            'wifi_rssi': (-100, 0),       # dBm
            'lora_rssi': (-120, 0)        # dBm
        }
        
        for col, (min_val, max_val) in expected_ranges.items():
            if col in self.df.columns:
                data = self.df[col].dropna()
                
                if len(data) == 0:
                    print(f"⚠️  {col}: No valid data")
                    continue
                
                actual_min = data.min()
                actual_max = data.max()
                
                # Check for outliers
                outliers_low = (data < min_val).sum()
                outliers_high = (data > max_val).sum()
                total_outliers = outliers_low + outliers_high
                
                outlier_percentage = (total_outliers / len(data)) * 100
                
                if outlier_percentage > 10:
                    status = "❌"
                    self.issues.append(f"High outliers in {col}: {outlier_percentage:.1f}%")
                elif outlier_percentage > 5:
                    status = "⚠️"
                else:
                    status = "✅"
                
                print(f"{status} {col}: range {actual_min:.2f} to {actual_max:.2f}, {outlier_percentage:.1f}% outliers")
    
    def check_time_consistency(self):
        """Check timestamp consistency"""
        print("\n⏱️  TIME CONSISTENCY CHECK")
        print("-" * 40)
        
        if 'iso_timestamp' not in self.df.columns:
            print("❌ No timestamp column found")
            return
        
        try:
            # Convert to datetime
            timestamps = pd.to_datetime(self.df['iso_timestamp'])
            
            # Check for duplicates
            duplicate_count = timestamps.duplicated().sum()
            if duplicate_count > 0:
                self.issues.append(f"❌ Duplicate timestamps: {duplicate_count}")
                print(f"❌ Duplicate timestamps: {duplicate_count}")
            else:
                print("✅ No duplicate timestamps")
            
            # Check time gaps
            time_diffs = timestamps.diff().dropna()
            
            if len(time_diffs) > 0:
                avg_interval = time_diffs.mean().total_seconds()
                max_gap = time_diffs.max().total_seconds()
                
                print(f"✅ Average interval: {avg_interval:.2f} seconds")
                
                # Look for large gaps (> 5 seconds)
                large_gaps = (time_diffs.dt.total_seconds() > 5).sum()
                if large_gaps > 0:
                    self.issues.append(f"⚠️ Large time gaps: {large_gaps}")
                    print(f"⚠️  Large time gaps (>5s): {large_gaps}")
                    print(f"   Maximum gap: {max_gap:.2f} seconds")
                else:
                    print("✅ No significant time gaps")
            
        except Exception as e:
            self.issues.append(f"❌ Timestamp parsing error: {e}")
            print(f"❌ Error parsing timestamps: {e}")
    
    def check_flight_logic(self):
        """Check flight state logic"""
        print("\n🚀 FLIGHT LOGIC CHECK")
        print("-" * 40)
        
        if 'state' not in self.df.columns:
            print("❌ No state column for flight logic check")
            return
        
        # Expected state progression
        expected_states = ['IDLE', 'READY', 'POWERED_FLIGHT', 'COASTING', 'APOGEE', 'DESCENT', 'LANDED']
        
        actual_states = self.df['state'].dropna().unique()
        print(f"📊 States found: {list(actual_states)}")
        
        # Check for reasonable state durations
        state_counts = self.df['state'].value_counts()
        
        for state, count in state_counts.items():
            duration_estimate = count * 0.1  # Assume 10Hz
            
            # Flag suspicious state durations
            if state == 'POWERED_FLIGHT' and duration_estimate > 30:
                self.issues.append(f"⚠️ Long powered flight: {duration_estimate:.1f}s")
                print(f"⚠️  {state}: {duration_estimate:.1f}s (unusually long)")
            elif state == 'APOGEE' and duration_estimate > 10:
                self.issues.append(f"⚠️ Long apogee phase: {duration_estimate:.1f}s")
                print(f"⚠️  {state}: {duration_estimate:.1f}s (unusually long)")
            else:
                print(f"✅ {state}: {duration_estimate:.1f}s")
    
    def check_sensor_correlation(self):
        """Check sensor data correlation"""
        print("\n🔗 SENSOR CORRELATION CHECK")
        print("-" * 40)
        
        # Check altitude sensors correlation
        alt_sensors = ['agl_altitude', 'gps_altitude', 'kalman_altitude']
        available_alt = [col for col in alt_sensors if col in self.df.columns]
        
        if len(available_alt) >= 2:
            print("📊 Altitude sensor correlation:")
            
            for i in range(len(available_alt)):
                for j in range(i+1, len(available_alt)):
                    col1, col2 = available_alt[i], available_alt[j]
                    
                    # Calculate correlation on valid data
                    valid_mask = self.df[col1].notna() & self.df[col2].notna()
                    valid_data = self.df[valid_mask]
                    
                    if len(valid_data) > 10:
                        correlation = valid_data[col1].corr(valid_data[col2])
                        
                        if correlation > 0.8:
                            status = "✅"
                        elif correlation > 0.5:
                            status = "⚠️"
                        else:
                            status = "❌"
                            self.issues.append(f"Poor correlation: {col1} vs {col2}: {correlation:.3f}")
                        
                        print(f"   {status} {col1} vs {col2}: {correlation:.3f}")
                    else:
                        print(f"   ❌ {col1} vs {col2}: Insufficient data")
        else:
            print("⚠️  Only one altitude sensor available")
        
        # Check velocity sensors correlation
        vel_sensors = ['velocity', 'kalman_vertical_velocity']
        available_vel = [col for col in vel_sensors if col in self.df.columns]
        
        if len(available_vel) == 2:
            col1, col2 = available_vel
            valid_mask = self.df[col1].notna() & self.df[col2].notna()
            valid_data = self.df[valid_mask]
            
            if len(valid_data) > 10:
                correlation = valid_data[col1].corr(valid_data[col2])
                
                if correlation > 0.7:
                    status = "✅"
                elif correlation > 0.4:
                    status = "⚠️"
                else:
                    status = "❌"
                    self.issues.append(f"Poor velocity correlation: {correlation:.3f}")
                
                print(f"📊 Velocity correlation: {correlation:.3f} {status}")
            else:
                print("⚠️  Insufficient velocity data for correlation")
    
    def generate_summary(self):
        """Generate quality summary"""
        print("\n" + "="*60)
        print("📋 DATA QUALITY SUMMARY")
        print("="*60)
        
        if not self.issues:
            print("🎉 DATA QUALITY: EXCELLENT")
            print("✅ No significant issues found")
            quality_score = 100
        else:
            issue_count = len(self.issues)
            quality_score = max(0, 100 - (issue_count * 10))
            
            if quality_score >= 80:
                quality_level = "GOOD"
                emoji = "👍"
            elif quality_score >= 60:
                quality_level = "FAIR"
                emoji = "⚠️"
            else:
                quality_level = "POOR"
                emoji = "❌"
            
            print(f"{emoji} DATA QUALITY: {quality_level} ({quality_score}%)")
            print(f"\n📝 Issues found ({issue_count}):")
            
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if quality_score < 80:
            print("   • Check sensor connections and calibration")
            print("   • Verify data logging configuration")
            print("   • Consider filtering or interpolation for missing data")
        
        if any("correlation" in issue.lower() for issue in self.issues):
            print("   • Check sensor mounting and alignment")
            print("   • Verify Kalman filter tuning")
        
        if any("outlier" in issue.lower() for issue in self.issues):
            print("   • Implement outlier detection and filtering")
            print("   • Check for electromagnetic interference")
        
        if not self.issues:
            print("   • Data quality is excellent, ready for analysis")
            print("   • Consider this dataset as a reference for future flights")
        
        return quality_score

def main():
    parser = argparse.ArgumentParser(description='Check telemetry data quality')
    parser.add_argument('csv_file', nargs='?', help='Path to CSV file (optional)')
    
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
                print(f"🔍 Using latest CSV: {csv_file.name}")
            else:
                print("❌ No CSV files found")
                return
        else:
            print("❌ telemetry_logs directory not found")
            return
    
    if not csv_file.exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    # Check data quality
    checker = DataQualityChecker(csv_file)
    
    if not checker.load_data():
        return
    
    print("🔍 N4 TELEMETRY DATA QUALITY CHECK")
    print(f"📁 File: {csv_file.name}")
    
    checker.check_basic_structure()
    checker.check_data_completeness()
    checker.check_data_ranges()
    checker.check_time_consistency()
    checker.check_flight_logic()
    checker.check_sensor_correlation()
    
    quality_score = checker.generate_summary()
    
    print(f"\n✅ Quality check complete! Score: {quality_score}%")

if __name__ == "__main__":
    main()
