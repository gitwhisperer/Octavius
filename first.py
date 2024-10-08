import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load your weather dataset
weather_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/Dataset/2022-01-01_2022-12-31.csv'
weather_data = pd.read_csv(weather_data_path)

# Load your power load dataset
power_load_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/Dataset/power_load.csv'
power_load_data = pd.read_csv(power_load_data_path)

# Convert datetime to datetime with dayfirst=True for weather data
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], dayfirst=True)

# Convert datetime to datetime with the correct format for power load data
power_load_data['datetime'] = pd.to_datetime(power_load_data['datetime'], format='%Y-%m-%d')

# Merge the datasets on the datetime column
data = pd.merge(weather_data, power_load_data, on='datetime')

# Extract features from the datetime
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour

# Select features and target variable
features = ['temp', 'humidity', 'windspeed', 'day_of_week', 'month', 'hour']
target = 'power_load'  # Assuming 'power_load' is the target column in the power load dataset

# Ensure the target column exists
if target not in data.columns:
    raise KeyError(f"The column '{target}' does not exist in the dataset.")

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the model
joblib.dump(model, 'power_load_model.pkl')