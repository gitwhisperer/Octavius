import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta

# URL of the website containing holiday data
url = 'https://www.officeholidays.com/countries/india/delhi/2022'

# Send a GET request to the website
response = requests.get(url)
response.raise_for_status()  # Check if the request was successful

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the holiday data
table = soup.find('table', {'class': 'country-table'})

# Extract the table rows
rows = table.find_all('tr')

# Initialize lists to store the data
dates = []
holidays = []

# Iterate over the rows and extract the date and holiday name
for row in rows[1:]:  # Skip the header row
    cols = row.find_all('td')
    date = cols[0].text.strip()
    holiday = cols[1].text.strip()
    dates.append(date)
    holidays.append(holiday)

# Add every Sunday of 2022 as a holiday
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)
current_date = start_date

while current_date <= end_date:
    if current_date.weekday() == 6:  # Sunday
        dates.append(current_date.strftime('%Y-%m-%d'))
        holidays.append('Sunday')
    current_date += timedelta(days=1)

# Create a DataFrame from the extracted data
df = pd.DataFrame({
    'Date': dates,
    'Holiday': holidays
})

# Save the DataFrame to a CSV file
csv_file_path = 'delhi_holidays_2022.csv'
df.to_csv(csv_file_path, index=False)

print('Holiday data saved to delhi_holidays_2022.csv')

# Open the CSV file with the default application
os.startfile(csv_file_path)