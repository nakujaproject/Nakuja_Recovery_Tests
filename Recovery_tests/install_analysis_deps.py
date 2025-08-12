#!/usr/bin/env python3
"""
Install Analysis Dependencies
Installs required packages for telemetry analysis tools
"""

import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scipy'
    ]
    
    print("🔧 Installing analysis dependencies...")
    
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            print(f"💡 Try: pip install {package}")
    
    print("\n✅ Installation complete!")
    print("\n🚀 Available analysis tools:")
    print("   • python view_telemetry.py - Real-time CSV viewer")
    print("   • python analyze_telemetry.py - Comprehensive analysis")
    print("   • python performance_metrics.py - Flight performance")
    print("   • python compare_flights.py - Multi-flight comparison")
    print("   • python data_quality_check.py - Data quality assessment")

if __name__ == "__main__":
    install_packages()
