import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gower import gower_matrix

# Creating a sample dataset
data = {
    'Station': ['Station1', 'Station1', 'Station2', 'Station2', 'Station3'],
    'Latitude': [34.05, 34.05, 40.71, 40.71, 36.12],
    'Longitude': [-118.24, -118.24, -74.01, -74.01, -115.17],
    'Day': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-01', '2021-01-02', '2021-01-01']),
    'Temperature': [20, 21, 3, 5, 13],  # In Celsius
    'Humidity': [30, 35, 80, 85, 50],    # Percentage
    'Wind Speed': [10, 15, 20, 25, 10]   # km/h
}

df = pd.DataFrame(data)

# Normalize the continuous numerical data
scaler = MinMaxScaler()
df[['Latitude', 'Longitude', 'Temperature', 'Humidity', 'Wind Speed']] = scaler.fit_transform(df[['Latitude', 'Longitude', 'Temperature', 'Humidity', 'Wind Speed']])



# Convert Day to just the date component for easier grouping
df['Day'] = df['Day'].dt.date

# Aggregate data by day
daily_weather = df.groupby('Day').agg({
    'Temperature': 'mean',
    'Humidity': 'mean',
    'Wind Speed': 'mean'
}).reset_index()

# Normalize the aggregated data
scaler = MinMaxScaler()
daily_weather[['Temperature', 'Humidity', 'Wind Speed']] = scaler.fit_transform(daily_weather[['Temperature', 'Humidity', 'Wind Speed']])

# Compute the Gower distance matrix
gower_dist_matrix = gower_matrix(daily_weather.drop('Day', axis=1))


print(gower_dist_matrix)

# Get the number of days
n = gower_dist_matrix.shape[0]

# Create a list to store the days and their similarities
similar_days = []

# Iterate through each pair
for i in range(n):
    for j in range(i + 1, n):
        similar_days.append((daily_weather.iloc[i]['Day'], daily_weather.iloc[j]['Day'], gower_dist_matrix[i, j]))

# Sort the list by distance
similar_days.sort(key=lambda x: x[2])

# Organize results to show each day and its most similar days
results = {}
for day1, day2, dist in similar_days:
    if day1 not in results:
        results[day1] = []
    results[day1].append((day2, dist))

# Print the most similar days for each day
for day, sims in results.items():
    similar_days_str = ', '.join([f"{d} (distance: {dist:.2f})" for d, dist in sims[:3]])  # top 3 similar days
    print(f"Day {day} is similar to: {similar_days_str}")