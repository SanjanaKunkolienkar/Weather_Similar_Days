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

# Directory with all csv files
common_path = "C:/Users/sanja/OneDrive - Texas A&M University/csv_files/"
station_path = 'D:/Github/Weather_Similar_Days/station.csv'
stations = pd.read_csv(station_path)
stations.columns = ['WhoAmI', 'Latitude', 'Longitude', 'ElevationMeters', 'ICAO', 'WMO', 'Region', 'Country2']
stations = stations[1:]
# delete nan values
stations = stations.dropna(subset=['Region'])
stations['WhoAmI'] = stations['WhoAmI'].astype(str)
stations = stations[stations['Region'].str.contains('TX')]
stations_list_TX = stations['WhoAmI'].tolist()

# List of files in the directory
files = os.listdir(common_path)
tx_files = os.listdir('D:/Github_extras/Texas Weather/')
all_data = pd.DataFrame()
col = ['UTCISO8601','WhoAmI','DewPointF','tempF','GlobalHorizontalIrradianceWM2','CloudCoverPerc','DirectHorizontalIrradianceWM2','WindSpeedmph','WindDirection','WindSpeed100mph']
for file in files:
    if file.endswith('.csv'):
        print(file)
        if any(file[:-4] in x for x in tx_files):
            filepath = os.path.join('D:/Github_extras/Texas Weather/', f'{file[:-4]}_TX.csv')
            print(filepath)
            data = pd.read_csv(filepath)
            # delete first columns and rename columns
            data = data.drop(columns=['Unnamed: 0'])
            data.columns = col

            # convert 'WhoAmI' column to string type
            data['WhoAmI'] = data['WhoAmI'].astype(str)
        else:
            filepath = os.path.join(common_path, file)
            print(filepath)
            data = pd.read_csv(filepath)
            #delete first columns and rename columns
            data = data.drop(columns=['Unnamed: 0'])
            data.columns = col

            # convert 'WhoAmI' column to string type
            data['WhoAmI'] = data['WhoAmI'].astype(str)
            # replace the nan values in UTCISO8601 with the previous value
            data['UTCISO8601'] = data['UTCISO8601'].fillna(method='ffill')
            # filter data to include stations in stations_list_TX
            data = data[data['WhoAmI'].isin(stations_list_TX)]

            data.reset_index(drop=True, inplace=True)

            data.to_csv('D:/Github_extras/Texas Weather/' + file[:-4] + '_TX.csv')

        all_data = pd.concat([all_data, data], axis=0)

all_data.reset_index(drop=True, inplace=True)
all_data.to_csv('D:/Github_extras/All_Weather_TX.csv')

col = ['UTCISO8601', 'WhoAmI', 'DewPointF', 'tempF', 'GlobalHorizontalIrradianceWM2', 'CloudCoverPerc', 'DirectHorizontalIrradianceWM2', 'WindSpeedmph', 'WindDirection', 'WindSpeed100mph']
all_data.columns = ['UTCISO8601', 'WhoAmI', 'Dew Point', 'Temperature', 'Global Horizontal Irradiance', 'Cloud Cover', 'Direct Horizontal Irradiance', 'Wind Speed', 'Wind Direction', 'Wind Speed 100']
all_data[['Dew Point', 'Temperature', 'Global Horizontal Irradiance', 'Cloud Cover', 'Direct Horizontal Irradiance', 'Wind Speed', 'Wind Direction', 'Wind Speed 100']] = all_data[['Dew Point', 'Temperature', 'Global Horizontal Irradiance', 'Cloud Cover', 'Direct Horizontal Irradiance', 'Wind Speed', 'Wind Direction', 'Wind Speed 100']].apply(pd.to_numeric, errors='coerce')
all_data[['Date', 'Hour']] = pd.to_datetime(all_data['UTCISO8601'].str.split('T', expand=True))

mean_all_data = all_data.groupby('Date').agg({'Temperature': 'mean', 'Dew Point': 'mean',
                                              'Wind Speed': 'mean', 'Cloud Cover': 'mean',
                                              'Global Horizontal Irradiance': 'mean', 'Direct Horizontal Irradiance': 'mean',
                                              'Wind Speed 100': 'mean'}).reset_index()
mean_all_data.to_csv('D:/Github_extras/mean_weather_data_1950_2023.csv')

df_temp_dp = all_data[['UTCISO8601', 'Date', 'Time', 'WhoAmI', 'Temperature', 'Dew Point']]
df_wind = all_data[['UTCISO8601', 'Date', 'Time', 'WhoAmI', 'Wind Speed', 'Wind Speed 100']]
df_sun = all_data[['UTCISO8601', 'Date', 'Time', 'WhoAmI', 'Global Horizontal Irradiance', 'Direct Horizontal Irradiance', 'Cloud Cover']]

# calculate days for temperature and dewpoint
current_df = df_temp_dp
scaler = StandardScaler()
numeric_columns = ['Temperature', 'Dew Point']
current_df[numeric_columns] = scaler.fit_transform(current_df[numeric_columns])

# Reshape data: each row is a day with all hourly readings flattened into a single vector
daily_vectors = current_df.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
daily_vectors.columns = ['Date', 'Features']

# Find the maximum length
max_length = max(len(arr) for arr in daily_vectors['Features'].values)
print("Max Length: ", max_length)

# Pad each array to the maximum length
padded_features = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in daily_vectors['Features'].values]

# Convert to a 2D array
features_matrix = np.stack(padded_features)
pca = PCA(n_components=0.7) # get components till 70% of variance is represented (#TODO: try changing this to check effect on MAPE)
pca_features = pca.fit_transform(features_matrix)

# Nearest Neighbors for finding similar days
neighbors = NearestNeighbors(n_neighbors=20)
neighbors.fit(pca_features)
distances, indices = neighbors.kneighbors(pca_features)

#send indices to csv file
similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
similar_day_indices.to_csv('D:/Github_extras/similar_days_indices_1950_2023_by_temp_and_dewpoint.csv')
# convert distance to dataframe and save to csv with indices
distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=['Day1', 'Day2', 'Day3', 'Day4', 'Day5',
                                                                         'Day6', 'Day7', 'Day8', 'Day9', 'Day10', 'Day11',
                                                                             'Day12', 'Day13', 'Day14', 'Day15', 'Day16', 'Day17',
                                                                             'Day18', 'Day19', 'Day20'])
distances_df.to_csv('D:/Github_extras/Texas_1940-2023/distances1950_2021_by_temp_and_dewpoint.csv')

# calculate days for wind speed
current_df = df_wind
scaler = StandardScaler()
numeric_columns = ['Wind Speed', 'Wind Speed 100']
current_df[numeric_columns] = scaler.fit_transform(current_df[numeric_columns])

# Reshape data: each row is a day with all hourly readings flattened into a single vector
daily_vectors = current_df.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
daily_vectors.columns = ['Date', 'Features']

# Find the maximum length
max_length = max(len(arr) for arr in daily_vectors['Features'].values)
print("Max Length: ", max_length)

# Pad each array to the maximum length
padded_features = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in daily_vectors['Features'].values]

# Convert to a 2D array
features_matrix = np.stack(padded_features)
pca = PCA(n_components=0.7) # get components till 70% of variance is represented (#TODO: try changing this to check effect on MAPE)
pca_features = pca.fit_transform(features_matrix)

# Nearest Neighbors for finding similar days
neighbors = NearestNeighbors(n_neighbors=20)
neighbors.fit(pca_features)
distances, indices = neighbors.kneighbors(pca_features)

#send indices to csv file
similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
similar_day_indices.to_csv('D:/Github_extras/similar_days_indices_1950_2023_by_wind.csv')
# convert distance to dataframe and save to csv with indices
distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=['Day1', 'Day2', 'Day3', 'Day4', 'Day5',
                                                                            'Day6', 'Day7', 'Day8', 'Day9', 'Day10', 'Day11',
                                                                                'Day12', 'Day13', 'Day14', 'Day15', 'Day16', 'Day17',
                                                                                'Day18', 'Day19', 'Day20'])
distances_df.to_csv('D:/Github_extras/Texas_1940-2023/distances1950_2021_by_wind.csv')

# calculate days for sun
current_df = df_sun
scaler = StandardScaler()
numeric_columns = ['Global Horizontal Irradiance', 'Direct Horizontal Irradiance', 'Cloud Cover']

current_df[numeric_columns] = scaler.fit_transform(current_df[numeric_columns])

# Reshape data: each row is a day with all hourly readings flattened into a single vector
daily_vectors = current_df.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
daily_vectors.columns = ['Date', 'Features']

# Find the maximum length
max_length = max(len(arr) for arr in daily_vectors['Features'].values)
print("Max Length: ", max_length)

# Pad each array to the maximum length
padded_features = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in daily_vectors['Features'].values]

# Convert to a 2D array
features_matrix = np.stack(padded_features)
pca = PCA(n_components=0.7) # get components till 70% of variance is represented (#TODO: try changing this to check effect on MAPE)
pca_features = pca.fit_transform(features_matrix)

# Nearest Neighbors for finding similar days
neighbors = NearestNeighbors(n_neighbors=20)
neighbors.fit(pca_features)
distances, indices = neighbors.kneighbors(pca_features)

#send indices to csv file
similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
similar_day_indices.to_csv('D:/Github_extras/similar_days_indices_1950_2023_by_sun.csv')
# convert distance to dataframe and save to csv with indices
distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=['Day1', 'Day2', 'Day3', 'Day4', 'Day5',
                                                                            'Day6', 'Day7', 'Day8', 'Day9', 'Day10', 'Day11',
                                                                                'Day12', 'Day13', 'Day14', 'Day15', 'Day16', 'Day17',
                                                                                'Day18', 'Day19', 'Day20'])
distances_df.to_csv('D:/Github_extras/Texas_1940-2023/distances1950_2021_by_sun.csv')




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