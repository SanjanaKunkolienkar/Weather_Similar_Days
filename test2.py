# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN, KMeans
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
#
# np.random.seed(0)
# dates = pd.date_range(start='2021-01-01', periods=50, freq='D')
# data = {
#     'Date': np.repeat(dates, 3),
#     'Weather Station': np.tile(['Station1', 'Station2', 'Station3'], 50),
#     'Temperature': np.random.normal(loc=20, scale=5, size=150),
#     'Humidity': np.random.normal(loc=50, scale=10, size=150),
#     'Wind Speed': np.random.normal(loc=10, scale=2, size=150)
# }
#
# df = pd.DataFrame(data)
#
# # Pivot data to wide format
# df_pivot = df.pivot_table(index='Date', columns='Weather Station', values=['Temperature', 'Humidity', 'Wind Speed'])
#
# # Flatten the columns after pivoting
# df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
#
# # Fill NaN values if any
# df_pivot.fillna(method='ffill', inplace=True)  # or consider another method depending on your data
#
# print(df_pivot.head())

# # Normalize the data
# scaler = StandardScaler()
# X_pivot_scaled = scaler.fit_transform(df_pivot)
#
# # Adjust DBSCAN
# dbscan = DBSCAN(eps=0.03, min_samples=3)
# clusters_dbscan = dbscan.fit_predict(X_pivot_scaled)
#
# # K-Means with an assumed number of clusters
# kmeans = KMeans(n_clusters=3, random_state=0)
# clusters_kmeans = kmeans.fit_predict(X_pivot_scaled)
#
# # Add cluster information back to the DataFrame
# df_pivot['Cluster_DBSCAN'] = clusters_dbscan
# df_pivot['Cluster_KMeans'] = clusters_kmeans
#
# # Print results
# print(df_pivot[['Cluster_DBSCAN', 'Cluster_KMeans']])
#
# # Visualization with t-SNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# # Re-run t-SNE if necessary with new clusters
# X_tsne = tsne.fit_transform(X_pivot_scaled)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters_kmeans, cmap='viridis')
# plt.colorbar()
# plt.show()
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters_dbscan, cmap='viridis')
# plt.colorbar()
# plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Generate synthetic weather data for 10 days at 3 stations each day
np.random.seed(0)
dates = pd.date_range(start='2023-06-01', periods=10, freq='D')
stations = ['Station1', 'Station2', 'Station3']
times = ['12:00', '14:00', '16:00']  # assuming different times for simplicity

# Creating the DataFrame
data = {
    'Date': np.tile(np.repeat(dates, 3), 3),  # Each day repeated for each station
    'Time': times * 30,  # Each time repeated for each combination of day and station
    'Station': np.repeat(stations, 30),  # Each station repeated for each time slot over 10 days
    'Temperature': np.random.randint(15, 35, size=90),
    'Humidity': np.random.randint(30, 90, size=90),
    'Wind Speed': np.random.uniform(5, 20, size=90)
}

df = pd.DataFrame(data)
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])

# Normalize the data
scaler = StandardScaler()
numeric_columns = ['Temperature', 'Humidity', 'Wind Speed']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Use Nearest Neighbors to find similarities
neighbors = NearestNeighbors(n_neighbors=3)  # You can adjust the number of neighbors
neighbors.fit(df[numeric_columns])

# Find the nearest neighbors for each observation
distances, indices = neighbors.kneighbors(df[numeric_columns])

# Determine similar days
similar_days_result = {}

for i, date in enumerate(df['Date']):
    similar_dates = df['Date'].iloc[indices[i]].values
    if date not in similar_days_result:
        similar_days_result[date] = list(similar_dates)

# Displaying the results for the first few entries
for key, value in list(similar_days_result.items())[:3]:
    print(f"Day {key} has similar weather patterns on days: {set(value)}")

# Extract the day from the DateTime for visualization purposes
df['Day'] = df['DateTime'].dt.date

import matplotlib.pyplot as plt
# Visualize the similarities
plt.figure(figsize=(10, 6))
for i, (dist, index) in enumerate(zip(distances[0], indices[0])):
    day_data = df.iloc[index]
    plt.plot(numeric_columns, day_data[numeric_columns], label=f'Similar Day {i+1} on {day_data["Day"]}')

plt.legend()
plt.title('Weather Profiles of Days Similar to the First Entry')
plt.xlabel('Weather Metrics')
plt.ylabel('Normalized Values')
plt.grid(True)
plt.show()