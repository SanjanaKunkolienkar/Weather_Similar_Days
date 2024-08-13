import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import gc

# Create a small test dataset
data = {
    'UTCISO8601': [
        '2023-01-01T00:00:00Z', '2023-01-01T01:00:00Z', '2023-01-01T02:00:00Z',
        '2023-01-02T00:00:00Z', '2023-01-02T01:00:00Z', '2023-01-02T02:00:00Z'
    ],
    'DewPointF': [30, 31, 29, 28, 30, 32],
    'tempF': [60, 61, 59, 58, 60, 62],
    'GlobalHorizontalIrradianceWM2': [200, 210, 190, 180, 200, 220],
    'CloudCoverPerc': [10, 20, 15, 5, 10, 25],
    'DirectHorizontalIrradianceWM2': [500, 510, 490, 480, 500, 520],
    'WindSpeedmph': [5, 6, 4, 3, 5, 7],
    'WindSpeed100mph': [15, 16, 14, 13, 15, 17],
    'WindDirection': ['N', 'NE', 'E', 'S', 'SE', 'SW']
}

# Convert to Dask DataFrame
df = dd.from_pandas(pd.DataFrame(data), npartitions=1)

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


# Function to preprocess data and perform PCA and Nearest Neighbors analysis
def preprocess_and_analyze(data, numeric_columns, output_prefix):
    # Standardize the numeric columns using Dask
    def standardize_partition(df):
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df

    data = data.map_partitions(standardize_partition)

    # Compute the data to a pandas DataFrame
    data = data.compute()

    # Reshape data: each row is a day with all hourly readings flattened into a single vector
    daily_vectors = data.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
    daily_vectors.columns = ['Date', 'Features']

    print(daily_vectors.head())  # Debug output to inspect daily_vectors

    # Find the maximum length of the features arrays
    max_length = max(len(arr) for arr in daily_vectors['Features'])
    print(f"Max Length for {output_prefix}: ", max_length)

    # Pad each array to the maximum length
    padded_features = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in daily_vectors['Features']]

    # Convert to a 2D array
    features_matrix = np.vstack(padded_features)

    # Perform PCA
    pca = PCA(n_components=0.4)  # Adjust the variance threshold as needed
    pca_features = pca.fit_transform(features_matrix)

    # Nearest Neighbors for finding similar days
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors.fit(pca_features)
    distances, indices = neighbors.kneighbors(pca_features)

    # Save indices to CSV
    similar_day_indices = pd.DataFrame(indices, index=daily_vectors['Date'])
    similar_day_indices.to_csv(f'similar_days_indices_{output_prefix}.csv')

    # Convert distance to DataFrame and save to CSV with indices
    distances_df = pd.DataFrame(distances, index=daily_vectors['Date'], columns=[f'Day{i + 1}' for i in range(20)])
    distances_df.to_csv(f'distances_{output_prefix}.csv')

    # Free memory
    del data, daily_vectors, features_matrix, pca_features, distances, indices
    gc.collect()


# Create sub-dataframe for temperature and dew point analysis
df_temp_dp = df[['UTCISO8601', 'Date', 'Time', 'Temperature', 'Dew Point']]

# Analyze temperature and dew point
preprocess_and_analyze(df_temp_dp, ['Temperature', 'Dew Point'], 'temp_and_dewpoint')
