import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Define the start date and the number of days
start_date = '2023-06-01'
num_days = 10

# Create a date range for the days
dates = pd.date_range(start=start_date, periods=num_days, freq='D')

# Create an hourly time range
hours = pd.date_range("00:00", "23:00", freq="H").time

# Define the stations
stations = ['Station1', 'Station2', 'Station3']

# Create a MultiIndex from the product of dates, hours, and stations
multi_index = pd.MultiIndex.from_product([dates, hours, stations], names=["Date", "Hour", "Station"])

# Create the DataFrame with the MultiIndex
df = pd.DataFrame(index=multi_index).reset_index()

# Add temperature, humidity, and wind speed with random values
df['Temperature'] = np.random.randint(15, 35, size=len(df))
df['Humidity'] = np.random.randint(30, 90, size=len(df))
df['Wind Speed'] = np.random.uniform(5, 20, size=len(df))

print(df.head(48))

df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hour'].astype(str))

plt.figure(figsize=(15, 10))
for date in df['Date'].unique():
    daily_data = df[df['Date'] == date]
    plt.plot(daily_data['DateTime'].dt.hour, daily_data['Temperature'], label=f'Day {date}', marker='o')

plt.title('Hourly Temperature Profiles for Each Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Temperature')
plt.xticks(range(24))  # Set x-ticks to display each hour
plt.legend()
plt.grid(True)
plt.show()
# Normalize the data
scaler = StandardScaler()
numeric_columns = ['Temperature', 'Humidity', 'Wind Speed']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Reshape data: each row is a day with all hourly readings flattened into a single vector
daily_vectors = df.groupby('Date')[numeric_columns].apply(lambda g: np.ravel(g.values)).reset_index()
daily_vectors.columns = ['Date', 'Features']

# Convert features into a proper 2D array
features_matrix = np.stack(daily_vectors['Features'].values)

# Applying PCA if possible (with correct dimensionality)
min_components = min(features_matrix.shape)
pca = PCA(n_components=min(min_components, 5))  # PCA with a safe number of components
pca_features = pca.fit_transform(features_matrix)

# Nearest Neighbors for finding similar days
neighbors = NearestNeighbors(n_neighbors=2)
neighbors.fit(pca_features)
distances, indices = neighbors.kneighbors(pca_features)

# Example of plotting two similar days
similar_days_indices = indices[0]  # First set of indices for similar days
dates_to_plot = daily_vectors.iloc[similar_days_indices]['Date']

# Filtering DataFrame to include only the similar days
data_to_plot = df[df['Date'].isin(dates_to_plot)].copy()
data_to_plot['Hour'] = data_to_plot['DateTime'].dt.hour  # Extract hour for plotting

# # Setting up the plot
# plt.figure(figsize=(12, 6))
# colors = ['blue', 'green']  # Different colors for each day
#
# for i, day in enumerate(dates_to_plot):
#     day_data = data_to_plot[data_to_plot['Date'] == day]
#     plt.plot(day_data['Hour'], day_data['Temperature'], label=f"Day {day_data['Date'].iloc[0]}", color=colors[i % len(colors)], marker='o')
#
# plt.title('Temperature Profiles for Similar Days')
# plt.xlabel('Hour of the Day')
# plt.ylabel('Normalized Temperature')
# plt.xticks(np.arange(24))  # Set ticks for every hour
# plt.legend()
# plt.grid(True)
# plt.show()

# Define colors for each day
colors = ['blue', 'green', 'red']  # Extend or change colors as needed

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, figsize=(14, 18), sharex=True)  # 3 rows for each weather metric

# Plot each metric
metrics = ['Temperature', 'Humidity', 'Wind Speed']
y_labels = ['Normalized Temperature', 'Normalized Humidity', 'Normalized Wind Speed']

for ax, metric, y_label in zip(axes, metrics, y_labels):
    for i, day in enumerate(dates_to_plot):
        day_data = data_to_plot[data_to_plot['Date'] == day]
        ax.plot(day_data['Hour'], day_data[metric], label=f"Day {day_data['Date'].iloc[0]}", color=colors[i % len(colors)], marker='o')
    ax.set_title(f'{metric} Profiles for Similar Days')
    ax.set_ylabel(y_label)
    ax.grid(True)

# Only set x-label on the last subplot
axes[-1].set_xlabel('Hour of the Day')
plt.xticks(np.arange(0, 24, 1))  # Set ticks for every hour

# Add legends and adjust layout
axes[0].legend()
plt.tight_layout()
plt.show()
