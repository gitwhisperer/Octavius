import pandas as pd
import numpy as np

# Load your weather dataset
weather_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/Dataset/2022-01-01_2022-12-31.csv'
weather_data = pd.read_csv(weather_data_path)

# Convert datetime to datetime with dayfirst=True
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], dayfirst=True)

# Generate synthetic power load data
# Example: power load is influenced by temperature, humidity, and wind speed
np.random.seed(42)
temperature = weather_data['temp']
humidity = weather_data['humidity']
wind_speed = weather_data['windspeed']

power_load = (
    1000 + 10 * temperature - 5 * humidity + 2 * wind_speed + np.random.normal(scale=50, size=len(weather_data))
)

# Combine data into a DataFrame
data = pd.DataFrame({
    'datetime': weather_data['datetime'],
    'power_load': power_load
})

# Save the DataFrame to a CSV file
data.to_csv('synthetic_power_load_data.csv', index=False)

print("Synthetic power load data generated and saved to 'synthetic_power_load_data.csv'")