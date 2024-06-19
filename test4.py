import pandas as pd
import numpy as np

# Number of stations
num_stations = 10

# Measurements and hours
measurements = ['Temp', 'DewPoint', 'WindSpeed', 'WindDir', 'CloudCover']
hours = list(range(24))

# Create sample columns for all stations and all hours
columns = [f"Station{station}_{measure}_{hour}" for station in range(1, num_stations+1) for measure in measurements for hour in hours]

# Generate sample data for 365 days (1 year for simplicity)
num_days = 365
data = np.random.rand(num_days, len(columns)) * 100  # Random data scaled to 100 for illustration

# Create DataFrame
sample_df = pd.DataFrame(data, columns=columns)

sample_df.head()
df = sample_df

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Adjust n_components based on explained variance requirement
principal_components = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Print the explained variance ratio for each component
print("Explained variance ratio by component:")
print(explained_variance_ratio)

# Print the cumulative explained variance
print("\nCumulative explained variance:")
print(cumulative_variance)


# Convert PCA results to DataFrame
pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Example of saving to CSV
pca_df.to_csv('daily_weather_pca_results.csv', index=False)
