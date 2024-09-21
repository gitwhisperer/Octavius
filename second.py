import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load weather data
weather_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/Dataset/2022-01-01_2022-12-31.csv'
weather_data = pd.read_csv(weather_data_path)

# Load power load data
power_load_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/Dataset/power_load.csv'
power_load_data = pd.read_csv(power_load_data_path)

# Load holidays data
holidays_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/delhi_holidays_2022.csv'
holidays_data = pd.read_csv(holidays_data_path)

# Convert datetime columns
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], dayfirst=True)
power_load_data['datetime'] = pd.to_datetime(power_load_data['datetime'], format='%Y-%m-%d')

# Filter out non-date entries in holidays data
holidays_data = holidays_data[pd.to_datetime(holidays_data['Date'], format='%Y-%m-%d', errors='coerce').notna()]

# Convert datetime for holidays data
holidays_data['Date'] = pd.to_datetime(holidays_data['Date'], format='%Y-%m-%d')

# Merge datasets
data = pd.merge(weather_data, power_load_data, on='datetime')

# Add holiday feature
data['is_holiday'] = data['datetime'].isin(holidays_data['Date']).astype(int)

# Extract datetime features
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour

# Select features and target
features = ['temp', 'humidity', 'windspeed', 'day_of_week', 'month', 'hour', 'is_holiday']
target = 'power_load'

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save model
joblib.dump(model, 'power_load_model2.pkl')