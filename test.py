import unittest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

class TestPowerLoadModel(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		# Load the model
		cls.model = joblib.load('power_load_model.pkl')
		
		# Load the weather dataset
		weather_data_path = 'C:/Users/shail/OneDrive/Desktop/GIT/Dataset/2022-01-01_2022-12-31.csv'
		weather_data = pd.read_csv(weather_data_path)
		
		# Load the power load dataset
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
		target = 'power_load'
		
		cls.X = data[features]
		cls.y = data[target]

	def test_model_prediction_shape(self):
		# Test if the model's prediction shape matches the input shape
		y_pred = self.model.predict(self.X)
		self.assertEqual(y_pred.shape, self.y.shape)

	def test_model_prediction_accuracy(self):
		# Test if the model's mean absolute error is within an acceptable range
		y_pred = self.model.predict(self.X)
		mae = mean_absolute_error(self.y, y_pred)
		self.assertLess(mae, 50)  # Example threshold, adjust as needed

	def test_model_predictions_not_null(self):
		# Test if the model's predictions are not null
		y_pred = self.model.predict(self.X)
		self.assertFalse(np.any(pd.isnull(y_pred)))

	def test_model_prediction_value_range(self):
		# Test if the model's predictions fall within a reasonable range
		y_pred = self.model.predict(self.X)
		self.assertTrue(np.all(y_pred > 0))  # Example condition, adjust as needed
		self.assertTrue(np.all(y_pred < 10000))  # Example condition, adjust as needed

	def test_model_loading(self):
		# Test if the model loads correctly
		model = joblib.load('power_load_model.pkl')
		self.assertIsNotNone(model)

if __name__ == '__main__':
	unittest.main()