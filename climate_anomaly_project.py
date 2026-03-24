import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """Calculates the rolling mean using a 1D convolution."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Generate Synthetic Climate Data (50 Years of Global Temps)
np.random.seed(42)
years, months, stations = 50, 12, 100

# Simulation: 15°C base + 2°C warming trend + 10°C seasonal swing + noise
trend = np.linspace(0, 2.5, years).reshape(years, 1, 1) 
seasonal_cycle = np.sin(np.linspace(0, 2 * np.pi, months)).reshape(1, months, 1) * 10
noise = np.random.normal(0, 3, (years, months, stations))
temp_cube = 15 + trend + seasonal_cycle + noise

baseline = np.mean(temp_cube, axis=0) 
anomalies = temp_cube - baseline  # Shape: (50, 12, 100)

annual_anomaly = np.mean(anomalies, axis=(1, 2)) 
# Standard deviation across stations (Uncertainty)
uncertainty = np.std(anomalies, axis=(1, 2))

# Calculate 5-Year Moving Average
window = 5
smoothed_trend = moving_average(annual_anomaly, window)
years_range = np.arange(years)
smooth_range = np.arange(window - 1, years)

# Visualization
plt.figure(figsize=(12, 7))

# Plot the raw annual bars (Red for hot, Blue for cold)
colors = ['#e74c3c' if x > 0 else '#3498db' for x in annual_anomaly]
plt.bar(years_range, annual_anomaly, alpha=0.3, color=colors, label='Annual Anomaly')

# Plot the Uncertainty Cloud (Shaded Area)
plt.fill_between(years_range, 
                 annual_anomaly - uncertainty, 
                 annual_anomaly + uncertainty, 
                 color='gray', alpha=0.15, label='Station Variance (±1σ)')

# Plot the Smoothed Trend Line
plt.plot(smooth_range, smoothed_trend, color='black', linewidth=3, 
         label=f'{window}-Year Moving Average', linestyle='-')

# Formatting
plt.axhline(0, color='black', linewidth=1.5)
plt.title("Global Temperature Anomaly & Long-Term Trend", fontsize=16, pad=20)
plt.xlabel("Years from Start of Simulation", fontsize=12)
plt.ylabel("Temperature Departure from Baseline (°C)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper left', frameon=True)

plt.text(years-10, smoothed_trend[-1]+0.5, "Warming Signal", 
         color='red', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()