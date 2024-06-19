import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import math

def mean_angle_deg(angles):
    sum_sin = 0
    sum_cos = 0
    for angle in angles:
        radians = math.radians(angle)
        sum_sin += math.sin(radians)
        sum_cos += math.cos(radians)
    mean_sin = sum_sin / len(angles)
    mean_cos = sum_cos / len(angles)
    mean_angle = math.atan2(mean_sin, mean_cos)
    return math.degrees(mean_angle)
# Read the text file
file_path = "D:/Github_extras/1977_weather.aux"  # Update with the actual path of your text file
with open(file_path, 'r') as file:
    data = file.read()

# Define the search header
header_stations = f"DATA (WeatherMeasurement,"
header_measurements = f"DATA (TimePointWeather,"

# Find the header and then the curly braces following it
start_index_stations = data.find(header_stations)
start_index_measurements = data.find(header_measurements)


if start_index_stations == -1:
    print("Header for Stations not found")
else:
    # Find the next occurrence of { and } after the header
    start_station = data.find('{', start_index_stations) + 1
    end_station = data.find('}', start_station)
    data_block_stations = data[start_station:end_station].strip()
    # Split the extracted string into lines and remove quotes
    data_lines_stations = data_block_stations.split('\n')
    cleaned_lines_stations  = [line.replace('"', '').strip().split() for line in data_lines_stations if line.strip()]
    # Load the cleaned lines into a DataFrame
    df_stations = pd.DataFrame(cleaned_lines_stations, columns=['Name','Latitude','Longitude','ElevationMeters','ICAO','WMO','Country2','Region'])

# get a list of files in the pww_filepath directory
# filepath =
# files = os.listdir(filepath)
# Find the header and then the curly braces following it in the pw aux file
if start_index_measurements == -1:
    print("Header for Stations not found")
else:
    # Find the next occurrence of { and } after the header
    start_station = data.find('{', start_index_measurements) + 1
    end_station = data.find('}', start_station)
    data_block_measurements = data[start_station:end_station].strip()

    # Split the extracted string into lines and remove quotes
    data_lines_measurements = data_block_measurements.split('\n')
    cleaned_lines_measurements  = [line.replace('"', '').strip().split() for line in data_lines_measurements if line.strip()]

    # Load the cleaned lines into a DataFrame
    df_measurements = pd.DataFrame(cleaned_lines_measurements, columns=['UTCISO8601','WhoAmI','TempF','DewPointF',
                                                                        'WindSpeedmph','WindDirection','CloudCoverPerc'])

#print(df_measurements)

# filter stations to only include the ones with 'TX' in the 'Region' column
df_stations_tx = df_stations[df_stations['Region'].str.contains('TX')]

# filter measurements to only include the df_stations_tx in the 'WhoAmI' column
df_measurements_tx = df_measurements[df_measurements['WhoAmI'].isin(df_stations_tx['Name'])]
#separate the date and time from the 'UTCISO8601' column into 'Date' and 'Time' columns
df_measurements_tx[['Date', 'Hour']] = df_measurements_tx['UTCISO8601'].str.split('T', expand=True)

# rename WhoAmI to Station, TempF to Temperature, DewPointF to DewPoint, WindSpeedmph to Wind Speed
# delete the 'UTCISO8601' column, 'WindDirection' column, and 'CloudCoverPerc' column
# renumber the index
df_measurements_tx = df_measurements_tx.rename(columns={'WhoAmI': 'Station', 'TempF': 'Temperature', 'DewPointF': 'Dew Point',
                                                        'WindSpeedmph': 'Wind Speed',
                                                        'CloudCoverPerc': 'Cloud Cover'})
df_measurements_tx = df_measurements_tx.drop(columns=['UTCISO8601', 'WindDirection'])
df_measurements_tx = df_measurements_tx.reset_index(drop=True)
#convert Temperature, DewPoint, and Wind Speed to float
# drop rows with none values
#df_measurements_tx = df_measurements_tx.dropna()
df_measurements_tx[['Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover']] = (
    df_measurements_tx[['Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover']].astype(float))
# print(df_measurements_tx)
# print(df_stations_tx)

df = df_measurements_tx.interpolate(method='linear')

#average the temperature, dewpoint, and wind speed for each time of the day
# df = df.groupby(['Date', 'Hour']).agg({'Temperature': 'mean', 'DewPoint': 'mean',
#                                        'Wind Speed': 'mean', 'Cloud Cover': 'mean'}).reset_index()
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hour'])
#calculate mean of the wind direciton of all stations at one time point
# df['Wind Direction'] = df_measurements_tx.groupby(['Date', 'Hour'])['Wind Direction'].transform(mean_angle_deg)
# plt.figure(figsize=(15, 10))
# for date in df['Date'].unique():
#     daily_data = df[df['Date'] == date]
#     plt.plot(daily_data['DateTime'].dt.hour, daily_data['Temperature'], label=f'Day {date}', marker='o')
#
# plt.title('Hourly Temperature Profiles for Each Day')
# plt.xlabel('Hour of the Day')
# plt.ylabel('Temperature')
# plt.xticks(range(24))  # Set x-ticks to display each hour
# plt.legend()
# plt.grid(True)
# plt.show()

# print(df.head(48))
# Normalize the data
scaler = StandardScaler()
numeric_columns = ['Temperature', 'Dew Point', 'Wind Speed']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Reshape data: each row is a day with all hourly readings flattened into a single vector
daily_vectors = df.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
daily_vectors.columns = ['Date', 'Features']

print(daily_vectors.head(5))

# Convert features into a proper 2D array
features_matrix = np.stack(daily_vectors['Features'].values)

min_components = min(features_matrix.shape)
pca = PCA(n_components=0.95)#min(min_components, 5))  # PCA with a safe number of components
pca_features = pca.fit_transform(features_matrix)

# Nearest Neighbors for finding similar days
neighbors = NearestNeighbors(n_neighbors=2)
neighbors.fit(pca_features)
distances, indices = neighbors.kneighbors(pca_features)

similar_days_indices = indices[0]  # gets day similar to day 1
dates_to_plot = daily_vectors.iloc[similar_days_indices]['Date']

# Filtering DataFrame to include only the similar days
data_to_plot = df[df['Date'].isin(dates_to_plot)].copy()
# get hour
data_to_plot['Hour'] = data_to_plot['DateTime'].dt.hour  # Extract hour for plotting

# Define colors for each day
colors = ['blue', 'green', 'red']
fig, axes = plt.subplots(nrows=3, figsize=(14, 18), sharex=True)
metrics = ['Temperature', 'Dew Point', 'Wind Speed']
y_labels = ['Normalized Temperature', 'Normalized DewPoint', 'Normalized Wind Speed']

for ax, metric, y_label in zip(axes, metrics, y_labels):
    for i, day in enumerate(dates_to_plot):
        day_data = data_to_plot[data_to_plot['Date'] == day]
        ax.plot(day_data['Hour'], day_data[metric], label=f"Day {day_data['Date'].iloc[0]}", color=colors[i % len(colors)], marker='o')
    ax.set_title(f'{metric} Profiles for Similar Days')
    ax.set_ylabel(y_label)
    ax.grid(True)

# Only set x-label on the last subplot
axes[-1].set_xlabel('Hour of the Day')
plt.xticks(np.arange(0, 24, 1))  # ticks for every hour

# Add legends and adjust layout
axes[0].legend()
plt.tight_layout()
plt.show()

# plot daily weather (averaged over all stations) for the two similar days
fig, ax = plt.subplots(3, 1, figsize=(15, 15))
day_data3 = df.groupby(['Date', 'Hour']).agg({'Temperature': 'mean', 'Dew Point': 'mean',
                                        'Wind Speed': 'mean', 'Cloud Cover': 'mean'}).reset_index()
print(dates_to_plot)
for i, day in enumerate(dates_to_plot):
    day_data2 = day_data3[day_data3['Date'] == day]
    ax[0].plot(day_data2['Hour'], day_data2['Temperature'], label=f"Day {day}", color=colors[i], marker='o')
    ax[1].plot(day_data2['Hour'], day_data2['Dew Point'], label=f"Day {day}", color=colors[i], marker='o')
    ax[2].plot(day_data2['Hour'], day_data2['Wind Speed'], label=f"Day {day}", color=colors[i], marker='o')


plt.show()

days_data = df_measurements_tx.groupby(['Date', 'Hour']).agg({'Temperature': 'mean', 'Dew Point': 'mean', 'Wind Speed': 'mean', 'Cloud Cover': 'mean'}).reset_index()
day1_data = days_data[days_data['Date'] == dates_to_plot[0]]
day2_data = days_data[days_data['Date'] == dates_to_plot[18]]

print(day1_data)
print(day2_data)

#match index of day1_data and day2_data
day2_data = day2_data.reset_index(drop=True)

print()
# calculate mape of temperature, dewpoint, and wind speed together
MAPE = (abs((day1_data[['Temperature', 'Dew Point', 'Wind Speed']]
             - day2_data[['Temperature', 'Dew Point', 'Wind Speed']])
            /day1_data[['Temperature', 'Dew Point', 'Wind Speed']])).mean() * 100


print(MAPE)