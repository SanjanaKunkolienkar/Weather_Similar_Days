
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA as daskPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import gc
# Read the parquet file into a Dask DataFrame
df = dd.read_parquet('D:/Github_extras/All_Weather_TX_1940_2023_cleaned.parquet')

# Rename columns
new_column_names = {
    'DewPointF': 'Dew Point',
    'tempF': 'Temperature',
    'GlobalHorizontalIrradianceWM2': 'Global Horizontal Irradiance',
    'CloudCoverPerc': 'Cloud Cover',
    'DirectHorizontalIrradianceWM2': 'Direct Horizontal Irradiance',
    'WindSpeedmph': 'Wind Speed',
    'WindSpeed100mph': 'Wind Speed 100'
}
df = df.rename(columns=new_column_names)

# Drop the 'WindDirection' column
df = df.drop(columns=['WindDirection'])

# Split 'UTCISO8601' into 'Date' and 'Time'
df['Date'] = dd.to_datetime(df['UTCISO8601'].str.slice(0, 10), format="%Y-%m-%d")
df['Time'] = df['UTCISO8601'].str.slice(11, 19).str.replace('Z', '')


def preprocess_and_analyze(data, numeric_columns, output_prefix):
    # Standardize numeric columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Perform groupby operation in Dask and convert to pandas
    daily_vectors = data.groupby('Date')[numeric_columns].apply(
        lambda g: np.ravel(g.values)
    ).compute().reset_index()

    daily_vectors.columns = ['Date', 'Features']

    # Pad the features arrays
    max_length = max(len(arr) for arr in daily_vectors['Features'])
    print(f"Max Length for {output_prefix}: ", max_length)

    padded_features = np.array([
        np.pad(arr, (0, max_length - len(arr)), mode='constant')
        for arr in daily_vectors['Features']
    ])

    # Perform PCA using Incremental PCA for large datasets
    print('doing PCA')
    ipca = IncrementalPCA(n_components=50, batch_size=5000)
    pca_features = ipca.fit_transform(padded_features)

    # Calculate variance explained by the components
    explained_variance = np.sum(ipca.explained_variance_ratio_)
    print(f'Total variance explained by 50 components: {explained_variance:.2f}')

    # Nearest Neighbors for finding similar days
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='auto')
    neighbors.fit(pca_features)
    distances, indices = neighbors.kneighbors(pca_features)

    print('calculated indices and distances, saving data')
    # Save indices and distances to CSV
    similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
    similar_day_indices.to_csv(f'similar_days_indices_{output_prefix}.csv')

    distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=[f'Day{i + 1}' for i in range(4)])
    distances_df.to_csv(f'distances_{output_prefix}.csv')

    # Free memory
    del data, daily_vectors, padded_features, pca_features, distances, indices
    gc.collect()

# Example usage
df_all = df
all_numeric_columns = ['Temperature', 'Dew Point', 'Wind Speed', 'Wind Speed 100',
                       'Global Horizontal Irradiance', 'Direct Horizontal Irradiance', 'Cloud Cover']
preprocess_and_analyze(df_all, all_numeric_columns, 'all')

# # Function to preprocess data and perform PCA and Nearest Neighbors analysis
# def preprocess_and_analyze(data, numeric_columns, output_prefix):
#     # Standardize the numeric columns using Dask
#     def standardize_partition(df):
#         scaler = StandardScaler()
#         df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
#         return df
#
#     data = data.map_partitions(standardize_partition)
#
#     # Compute the data to a pandas DataFrame
#     data = data.compute()
#
#     # Reshape data: each row is a day with all hourly readings flattened into a single vector
#     daily_vectors = data.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
#     daily_vectors.columns = ['Date', 'Features']
#
#     # Find the maximum length of the features arrays
#     max_length = max(len(arr) for arr in daily_vectors['Features'])
#     print(f"Max Length for {output_prefix}: ", max_length)
#
#     # Pad each array to the maximum length
#     padded_features = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in daily_vectors['Features']]
#
#     # Convert to a 2D array
#     features_matrix = np.vstack(padded_features)
#
#     # Perform PCA using randomized solver
#     print('doing PCA')
#     pca = PCA(n_components=50, svd_solver='randomized')  # Adjust the variance threshold as needed
#     pca_features = pca.fit_transform(features_matrix)
#
#     # Calculate variance explained by the components
#     explained_variance = np.sum(pca.explained_variance_ratio_)
#     print(f'Total variance explained by 50 components: {explained_variance:.2f}')
#
#     # Nearest Neighbors for finding similar days
#     neighbors = NearestNeighbors(n_neighbors=4, algorithm='auto')
#     neighbors.fit(pca_features)
#     distances, indices = neighbors.kneighbors(pca_features)
#
#     print('calculated indices and distances, saving data')
#     # Save indices to CSV
#     similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
#     similar_day_indices.to_csv(f'similar_days_indices_{output_prefix}.csv')
#
#     # Convert distance to DataFrame and save to CSV with indices
#     distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=[f'Day{i + 1}' for i in range(4)])
#     distances_df.to_csv(f'distances_{output_prefix}.csv')
#
#     # Free memory
#     del data, daily_vectors, features_matrix, pca_features, distances, indices
#     gc.collect()
#
#
# # # Create sub-dataframes for specific analyses
# # df_temp_dp = df[['UTCISO8601', 'Date', 'Time', 'Temperature', 'Dew Point']]
# # df_wind = df[['UTCISO8601', 'Date', 'Time', 'Wind Speed', 'Wind Speed 100']]
# # df_sun = df[['UTCISO8601', 'Date', 'Time', 'Global Horizontal Irradiance',
# #              'Direct Horizontal Irradiance', 'Cloud Cover']]
# df_all = df
#
# # Analyze temperature and dew point
# # preprocess_and_analyze(df_temp_dp, ['Temperature', 'Dew Point'], 'temp_and_dewpoint')
# #
# # # Analyze wind speed
# # preprocess_and_analyze(df_wind, ['Wind Speed', 'Wind Speed 100'], 'wind')
# #
# # # Analyze solar irradiance and cloud cover
# # preprocess_and_analyze(df_sun, ['Global Horizontal Irradiance', 'Direct Horizontal Irradiance', 'Cloud Cover'], 'sun')
#
# # Analyze all features
# all_numeric_columns = ['Temperature', 'Dew Point', 'Wind Speed', 'Wind Speed 100',
#                        'Global Horizontal Irradiance', 'Direct Horizontal Irradiance', 'Cloud Cover']
# preprocess_and_analyze(df_all, all_numeric_columns, 'all')
