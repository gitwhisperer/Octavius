import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load your dataset
# Assuming you have a CSV file with relevant data
data = pd.read_csv('power_load_data.csv')

# Preprocess the data
# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract features from the date
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['hour'] = data['date'].dt.hour

# Select features and target variable
features = ['temperature', 'humidity', 'wind_speed', 'day_of_week', 'month', 'hour', 'holiday', 'real_estate_development']
target = 'power_load'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the model for future use
joblib.dump(model, 'power_load_model.pkl')

# Function to predict power load
def predict_power_load(features):
    model = joblib.load('power_load_model.pkl')
    prediction = model.predict([features])
    return prediction

# Example usage
example_features = [30, 50, 10, 2, 7, 15, 0, 1]  # Example feature values
predicted_load = predict_power_load(example_features)
print(f'Predicted Power Load: {predicted_load}')