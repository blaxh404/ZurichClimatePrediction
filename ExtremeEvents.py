import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'data/Zurich_weather_cleaned.pkl'
cleaned_data = pd.read_pickle(file_path)

# Select the features for clustering
selected_features = ['precip', 'cloudcover', 'temp', 'windspeed', 'humidity', 'sealevelpressure']
data_for_clustering = cleaned_data[selected_features]

# Standardize the data (important for K-Means)
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

# Define a function to identify extreme climate events
def is_extreme_event(row):
    return (row['temp'] > 29)

# Add a new column 'extreme_event' to the DataFrame
cleaned_data['extreme_event'] = cleaned_data.apply(is_extreme_event, axis=1)

# Perform K-Means clustering
k = 6  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=k, random_state=0)
cleaned_data['cluster'] = kmeans.fit_predict(data_for_clustering_scaled)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Scatter plot of 'temp' vs 'precip' with clusters (different colors)
for cluster in range(k):
    cluster_data = cleaned_data[cleaned_data['cluster'] == cluster]
    ax1.scatter(cluster_data['temp'], cluster_data['precip'], label=f'Cluster {cluster}', s=2)

ax1.set_xlabel('Temperature')
ax1.set_ylabel('Precipitation')
ax1.set_title('K-Means Clustering of Weather Data (Temperature vs Precipitation)')
ax1.set_xlim(-15, 40)
ax1.legend()
ax1.grid(True)

# Subplot 2: Scatter plot of 'temp' vs 'precip' colored by extreme events (extreme events in black)
extreme_events = cleaned_data[cleaned_data['extreme_event']]
non_extreme_events = cleaned_data[~cleaned_data['extreme_event']]

ax2.scatter(non_extreme_events['temp'], non_extreme_events['precip'], c='lightgrey', label='Non-Extreme Events', s=2)
ax2.scatter(extreme_events['temp'], extreme_events['precip'], c='black', label='Extreme Events', s=10)

ax2.set_xlabel('Temperature')
ax2.set_ylabel('Precipitation')
ax2.set_title('Extreme Climate Events vs Non-Extreme Events (Temperature vs Precipitation)')
ax2.set_xlim(-15, 40)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
