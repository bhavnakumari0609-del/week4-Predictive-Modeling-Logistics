import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Number of records
n_samples = 5000

# Generate logistics features
distance_km = np.random.uniform(1, 35, n_samples)
traffic_index = np.random.uniform(1, 10, n_samples)
package_weight_kg = np.random.uniform(0.5, 25, n_samples)
driver_experience_yr = np.random.uniform(0.5, 15, n_samples)

weather_severity = np.random.choice(
    [0, 1, 2],
    size=n_samples,
    p=[0.70, 0.25, 0.05]
)

is_peak_hour = np.random.choice(
    [0, 1],
    size=n_samples,
    p=[0.65, 0.35]
)

num_stops_enroute = np.random.randint(0, 6, n_samples)

# Calculate delivery time
base_time = 8.0

transit_time = distance_km * (
    1.8 + 0.35 * traffic_index
)

handling_time = 0.6 * package_weight_kg

driver_efficiency = -0.4 * np.log1p(driver_experience_yr)

weather_delay = weather_severity * 6.5

peak_delay = is_peak_hour * 8.2

stops_delay = num_stops_enroute * 4.0

noise = np.random.normal(0, 3.5, n_samples)

delivery_time_min = (
    base_time
    + transit_time
    + handling_time
    + driver_efficiency
    + weather_delay
    + peak_delay
    + stops_delay
    + noise
)

# Avoid unrealistic delivery times
delivery_time_min = np.maximum(delivery_time_min, 5.0)

# Create DataFrame
df = pd.DataFrame({
    "distance_km": np.round(distance_km, 2),
    "traffic_index": np.round(traffic_index, 2),
    "package_weight_kg": np.round(package_weight_kg, 2),
    "driver_experience_yr": np.round(driver_experience_yr, 2),
    "weather_severity": weather_severity,
    "is_peak_hour": is_peak_hour,
    "num_stops_enroute": num_stops_enroute,
    "delivery_time_min": np.round(delivery_time_min, 2)
})

# Save dataset
df.to_csv("logistics_delivery_data.csv", index=False)

# Display information
print("Dataset created successfully!")
print(f"Total records: {len(df)}")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nFile saved as: logistics_delivery_data.csv")