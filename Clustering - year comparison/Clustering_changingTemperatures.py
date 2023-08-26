import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
file_path = 'datasets/Zurich_weather_cleaned.pkl'
cleaned_data = pd.read_pickle(file_path)
cleaned_data = cleaned_data.reset_index(drop=True)

# Add a new column 'extremeevent' based on conditions
cleaned_data['extremeevent'] = 0  # Initialize the column with 0

# Set 'extremeevent' to 1 where precip > 20 or temp > 30
cleaned_data.loc[(cleaned_data['precip'] > 20) | (cleaned_data['temp'] > 30), 'extremeevent'] = 1

# Add features
cleaned_data['temp_rolling_mean_5'] = cleaned_data['temp'].rolling(window=5).mean()
cleaned_data['precip_rolling_mean_5'] = cleaned_data['precip'].rolling(window=5).mean()

# Drop rows with missing values (resulting from rolling means and lag)
cleaned_data.dropna(inplace=True)

# Define date ranges
date_ranges = [(1994, 1999), (2017, 2022)]

# Select the features for clustering
selected_features = ['precip', 'temp', 'windspeed', 'humidity', 'temp_rolling_mean_5', 'precip_rolling_mean_5']

# Standardize the features (important for K-Means)
scaler = StandardScaler()

for start_year, end_year in date_ranges:
    # Filter data for the current date range
    current_data = cleaned_data[(cleaned_data['year'] >= start_year) & (cleaned_data['year'] <= end_year)]
    
    # Extract the features and scale them
    X = current_data[selected_features]
    X_scaled = scaler.fit_transform(X)
    
    # Calculate silhouette scores to determine the optimal number of clusters
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 because we started from k=2
    
    # Perform K-Means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to the original dataset
    current_data['cluster'] = kmeans.labels_
    
    # Save the dataset with cluster labels
    file_name = f'datasets/zurich_weather_clustered_{start_year}_{end_year}.pkl'
    current_data.to_pickle(file_name)
