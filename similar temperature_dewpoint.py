import pandas as pd
import os
import numpy as np
import pythoncom
import win32com.client
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import math
import warnings
warnings.filterwarnings("ignore")
def CheckResultForError(SimAutoOutput, Message):
    # Test if powerworld sim auto works
    if SimAutoOutput[0] != '':
        print('Error: ' + SimAutoOutput[0])
    else:
        print(Message)
        return Message

try:
    pw_object = win32com.client.Dispatch("pwrworld.SimulatorAuto")
except Exception as e:
    print(f"Error connecting to PowerWorld: {e}")
####### Defining common elements #######
# Define the search header
header_stations = f"DATA (WeatherMeasurement,"
header_measurements = f"DATA (TimePointWeather,"
# Directory with all aux files
common_path = "D:/Github_extras/Texas_1940-2023/Texas_ByTwoYears/"
# List of files in the directory
files = os.listdir(common_path)
init_source1 = 'D:/Github_extras/Texas_1940-2023/Texas_1940_1941.pww'
source1 = init_source1
for file in files:
    print(file)
    dest = os.path.join(common_path, 'Texas_1940_2023.pww')
    source2 = os.path.join(common_path, file)
    r = pw_object.RunScriptCommand(f'WeatherPWWFileCombine2(\"{source1}\", \"{source2}\", \"{dest}\");')

    source1 = dest






# all_years_cleaned_df = pd.DataFrame(columns = ['Station', 'Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover', 'Region', 'Date', 'Hour', 'DateTime'])
#
# for file in files:
#     file_path = os.path.join(common_path, file)
#     print(file_path)
#     with open(file_path, 'r') as file:
#         data = file.read()
#
#     # Find the header and then the curly braces following it
#     start_index_stations = data.find(header_stations)
#     start_index_measurements = data.find(header_measurements)
#
#     # read station information
#     if start_index_stations == -1:
#         print("Header for Stations not found")
#     else:
#         # Find the next occurrence of { and } after the header
#         start_station = data.find('{', start_index_stations) + 1
#         end_station = data.find('}', start_station)
#         data_block_stations = data[start_station:end_station].strip()
#         # Split the extracted string into lines and remove quotes
#         data_lines_stations = data_block_stations.split('\n')
#         cleaned_lines_stations  = [line.replace('"', '').strip().split() for line in data_lines_stations if line.strip()]
#         # Load the cleaned lines into a DataFrame
#         df_stations = pd.DataFrame(cleaned_lines_stations, columns=['Name','Latitude','Longitude','ElevationMeters','ICAO','WMO','Country2','Region'])
#
#     # read actual measurements for time points
#     if start_index_measurements == -1:
#         print("Header for Stations not found")
#     else:
#         # Find the next occurrence of { and } after the header
#         start_station = data.find('{', start_index_measurements) + 1
#         end_station = data.find('}', start_station)
#         data_block_measurements = data[start_station:end_station].strip()
#         # Split the extracted string into lines and remove quotes
#         data_lines_measurements = data_block_measurements.split('\n')
#         cleaned_lines_measurements  = [line.replace('"', '').strip().split() for line in data_lines_measurements if line.strip()]
#         # Load the cleaned lines into a DataFrame
#         df_measurements = pd.DataFrame(cleaned_lines_measurements, columns=['UTCISO8601','WhoAmI','TempF','DewPointF',
#                                                                             'WindSpeedmph','WindDirection','CloudCoverPerc'])
#
#     # filter stations to only include the ones with 'TX' in the 'Region' column
#     df_stations_tx = df_stations[df_stations['Region'].str.contains('TX')]
#     # filter measurements to only include the df_stations_tx in the 'WhoAmI' column
#     df_measurements_tx = df_measurements[df_measurements['WhoAmI'].isin(df_stations_tx['Name'])]
#     # add 'Region' column to df_measurements_tx based on 'WhoAmI' column
#     df_measurements_tx['Region'] = df_measurements_tx['WhoAmI'].map(df_stations_tx.set_index('Name')['Region'])
#
#     #separate the date and time from the 'UTCISO8601' column into 'Date' and 'Time' columns
#     df_measurements_tx[['Date', 'Hour']] = df_measurements_tx['UTCISO8601'].str.split('T', expand=True)
#
#     # rename WhoAmI to Station, TempF to Temperature, DewPointF to DewPoint, WindSpeedmph to Wind Speed
#     # delete the 'UTCISO8601' column, 'WindDirection' column, and 'CloudCoverPerc' column
#     # renumber the index
#     df_measurements_tx = df_measurements_tx.rename(columns={'WhoAmI': 'Station', 'TempF': 'Temperature', 'DewPointF': 'Dew Point',
#                                                         'WindSpeedmph': 'Wind Speed',
#                                                         'CloudCoverPerc': 'Cloud Cover'})
#
#
#     df_measurements_tx = df_measurements_tx.drop(columns=['UTCISO8601', 'WindDirection'])
#     df_measurements_tx = df_measurements_tx.reset_index(drop=True)
#
#     # clean cloud cover data column to replace strings with numbers
#     # SKC = 0, FEW = 30, SCT = 50, BKN = 90, OVC = 100
#     df_measurements_tx[['Cloud Cover']].replace({'SKC': 0, 'FEW': 30, 'SCT': 50, 'BKN': 90, 'OVC': 100, 'CLR': 75, '': 0}).astype(float)
#
#     #convert Temperature, DewPoint, and Wind Speed to float
#     df_measurements_tx[['Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover']] = df_measurements_tx[['Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover']].apply(pd.to_numeric, errors='coerce')
#     df_interpolated = df_measurements_tx.interpolate(method='linear')
#
#     cleaned_df = df_interpolated
#     cleaned_df['DateTime'] = pd.to_datetime(cleaned_df['Date'].astype(str) + ' ' + cleaned_df['Hour'])
#
#     print(cleaned_df.shape)
#     all_years_cleaned_df = pd.concat([all_years_cleaned_df, cleaned_df], axis=0)
#     print(all_years_cleaned_df.shape)
#
# df = all_years_cleaned_df.reset_index(drop=True)
#
# mean_df = df.groupby('DateTime').agg({'Temperature': 'mean', 'Dew Point': 'mean', 'Wind Speed': 'mean', 'Cloud Cover': 'mean'}).reset_index()
# mean_df.to_csv('D:/Github_extras/Texas_1940-2023/mean_weather_data_1950_2021_by_windspeed.csv')
#
# # Scale the data for PCA
# scaler = StandardScaler()
# numeric_columns = ['Cloud Cover']#, 'Wind Speed']
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
#
# # Reshape data: each row is a day with all hourly readings flattened into a single vector
# daily_vectors = df.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
# daily_vectors.columns = ['Date', 'Features']
#
# # Find the maximum length
# max_length = max(len(arr) for arr in daily_vectors['Features'].values)
#
# # Pad each array to the maximum length
# padded_features = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in daily_vectors['Features'].values]
#
# # Convert to a 2D array
# features_matrix = np.stack(padded_features)
#
# # Impute NaNs with the mean of the column
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# features_matrix = imputer.fit_transform(features_matrix)
#
# pca = PCA(n_components=0.7) # get components till 70% of variance is represented (#TODO: try changing this to check effect on MAPE)
# pca_features = pca.fit_transform(features_matrix)
#
# # Nearest Neighbors for finding similar days
# neighbors = NearestNeighbors(n_neighbors=20)
# neighbors.fit(pca_features)
# distances, indices = neighbors.kneighbors(pca_features)
#
# #send indices to csv file
# similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
#
#
# similar_day_indices.to_csv('D:/Github_extras/Texas_1940-2023/similar_days_indices_1950_2021_by_cloudcover.csv')
# # convert distance to dataframe and save to csv with indices
# distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=['Day1', 'Day2', 'Day3', 'Day4', 'Day5',
#                                                                          'Day6', 'Day7', 'Day8', 'Day9', 'Day10', 'Day11',
#                                                                              'Day12', 'Day13', 'Day14', 'Day15', 'Day16', 'Day17',
#                                                                              'Day18', 'Day19', 'Day20'])
#
# distances_df.to_csv('D:/Github_extras/Texas_1940-2023/distances1950_2021_by_cloudcover.csv')

#     similar_days_indices = indices[0]  # gets day similar to day 1
#     dates_to_plot = daily_vectors.iloc[similar_days_indices]['Date']
#
#     # Filtering DataFrame to include only the similar days
#     data_to_plot = df[df['Date'].isin(dates_to_plot)].copy()
#     # get hour
#     data_to_plot['Hour'] = data_to_plot['DateTime'].dt.hour  # Extract hour for plotting
#
#     # Define colors for each day
#     colors = ['blue', 'green', 'red']
#     fig, axes = plt.subplots(nrows=3, figsize=(14, 18), sharex=True)
#     metrics = ['Temperature', 'Dew Point', 'Wind Speed']
#     y_labels = ['Normalized Temperature', 'Normalized DewPoint', 'Normalized Wind Speed']
#
#     for ax, metric, y_label in zip(axes, metrics, y_labels):
#         for i, day in enumerate(dates_to_plot):
#             day_data = data_to_plot[data_to_plot['Date'] == day]
#             ax.plot(day_data['Hour'], day_data[metric], label=f"Day {day_data['Date'].iloc[0]}", color=colors[i % len(colors)], marker='o')
#         ax.set_title(f'{metric} Profiles for Similar Days')
#         ax.set_ylabel(y_label)
#         ax.grid(True)
#
#     # Only set x-label on the last subplot
#     axes[-1].set_xlabel('Hour of the Day')
#     plt.xticks(np.arange(0, 24, 1))  # ticks for every hour
#
#     # Add legends and adjust layout
#     axes[0].legend()
#     plt.tight_layout()
#     plt.show()
#
# # plot daily weather (averaged over all stations) for the two similar days
# fig, ax = plt.subplots(3, 1, figsize=(15, 15))
# day_data3 = df.groupby(['Date', 'Hour']).agg({'Temperature': 'mean', 'Dew Point': 'mean',
#                                         'Wind Speed': 'mean', 'Cloud Cover': 'mean'}).reset_index()
# print(dates_to_plot)
# for i, day in enumerate(dates_to_plot):
#     day_data2 = day_data3[day_data3['Date'] == day]
#     ax[0].plot(day_data2['Hour'], day_data2['Temperature'], label=f"Day {day}", color=colors[i], marker='o')
#     ax[1].plot(day_data2['Hour'], day_data2['Dew Point'], label=f"Day {day}", color=colors[i], marker='o')
#     ax[2].plot(day_data2['Hour'], day_data2['Wind Speed'], label=f"Day {day}", color=colors[i], marker='o')
#
#
# plt.show()
#
# days_data = df_measurements_tx.groupby(['Date', 'Hour']).agg({'Temperature': 'mean', 'Dew Point': 'mean', 'Wind Speed': 'mean', 'Cloud Cover': 'mean'}).reset_index()
# day1_data = days_data[days_data['Date'] == dates_to_plot[0]]
# day2_data = days_data[days_data['Date'] == dates_to_plot[18]]
# day3_data = days_data[days_data['Date'] == dates_to_plot[10]]
# print(day1_data)
# print(day2_data)
# print(day3_data)
# #match index of day1_data and day2_data
# day2_data = day2_data.reset_index(drop=True)
# day3_data = day3_data.reset_index(drop=True)
#
# print()
# # calculate mape of temperature, dewpoint, and wind speed together
# MAPE1 = (abs((day1_data[['Temperature', 'Dew Point', 'Wind Speed']]
#              - day2_data[['Temperature', 'Dew Point', 'Wind Speed']])
#             /day1_data[['Temperature', 'Dew Point', 'Wind Speed']])).mean() * 100
# MAPE2 = (abs((day1_data[['Temperature', 'Dew Point', 'Wind Speed']]
#              - day3_data[['Temperature', 'Dew Point', 'Wind Speed']])
#             /day1_data[['Temperature', 'Dew Point', 'Wind Speed']])).mean() * 100
#
# print("MAPE1: ")
# print(MAPE1)
# print("MAPE2: ")
# print(MAPE2)