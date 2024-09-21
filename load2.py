import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
	'n_estimators': [100, 200, 300],
	'max_depth': [None, 10, 20, 30],
	'min_samples_split': [2, 5, 10],
	'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mean_actual = np.mean(y_test)
percentage_error = (mae / mean_actual) * 100

print(f'Mean Absolute Error: {mae}')
print(f'Percentage Error: {percentage_error:.2f}%')

# Save the best model
joblib.dump(best_model, 'power_load_model.pkl')

# Data Visualization
# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Power Load')
plt.show()

# Plot feature importance
feature_importances = pd.DataFrame(best_model.feature_importances_,
								   index = X_train.columns,
								   columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()