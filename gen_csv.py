import pandas as pd
import numpy as np

# Generate timestamps (one per hour over a period)
timestamps = pd.date_range(start="2024-10-01", end="2024-10-15", freq='H')

# Generate water level data for 100 roads
data = {'Timestamp': timestamps}
for i in range(1, 101):
    road_id = f"Road_{i}"
    data[road_id] = np.round(np.random.uniform(0, 5, size=len(timestamps)), 3)
 # Random water levels between 0 and 5 meters

# Create the DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('road_water_levels_large.csv', index=False)
