import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV file ===
# Replace with your actual CSV file path
csv_file = r"C:\Users\nyong\OneDrive\Documents\Nakuja_Recovery_Tests\Recovery_tests\outputs\flight_segments\telemetry_20250807_171047_seg1.csv"

df = pd.read_csv(csv_file)

# === Ensure relevant columns exist ===
if not {"agl_altitude", "kalman_altitude"}.issubset(df.columns):
    raise ValueError("CSV file must contain 'agl_altitude' and 'kalman_altitude' columns.")

# === Plot Raw vs Filtered Altitude ===
plt.figure(figsize=(10, 6))
plt.plot(df["agl_altitude"], label="Raw Altitude (AGL)", color="red", linewidth=1)
plt.plot(df["kalman_altitude"], label="Filtered Altitude (Kalman)", color="blue", linewidth=1)

# === Labels and Styling ===
plt.title("Raw vs Kalman Filtered Altitude", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Altitude (meters)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Show Plot ===
plt.show()
