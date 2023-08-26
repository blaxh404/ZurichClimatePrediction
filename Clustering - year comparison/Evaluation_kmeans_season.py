import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
data_1994_to_1999 = pd.read_pickle('../datasets/zurich_weather_clustered_1994_1999.pkl')
data_2017_to_2022 = pd.read_pickle('../datasets/zurich_weather_clustered_2017_2022.pkl')

# Get unique cluster labels for each dataset
clusters_1994_to_1999 = sorted(data_1994_to_1999['cluster'].unique())
clusters_2017_to_2022 = sorted(data_2017_to_2022['cluster'].unique())

# Define a mapping of season codes to season names and colors
season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
season_colors = {'Spring': 'green', 'Summer': 'red', 'Fall': 'orange', 'Winter': 'blue'}

# Create a single plot with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Initialize a dictionary to keep track of unique legend labels and handles
legend_dict = {}

# Create stacked bar plots for each dataset
for ax, dataset, clusters in zip(axes, [data_1994_to_1999, data_2017_to_2022], [clusters_1994_to_1999, clusters_2017_to_2022]):
    
    for cluster in clusters:
        cluster_data = dataset[dataset['cluster'] == cluster]
        season_counts = cluster_data['season'].value_counts().rename(season_mapping)
        season_counts = season_counts.reindex(sorted(season_mapping.values()))
        season_percentage = season_counts / season_counts.sum() * 100  # Calculate percentage
        bottom = 0
        
        for season, percentage in season_percentage.items():
            bars = ax.bar(cluster, percentage, label=season, bottom=bottom, color=season_colors[season], alpha=0.7)
            bottom += percentage
            
            # Add to the legend dictionary if not already present
            if season not in legend_dict:
                legend_dict[season] = bars[0]
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Season Composition by Cluster ({dataset["year"].min()}-{dataset["year"].max()})')
    ax.set_xticks(clusters)

# Create a unique legend for both subplots
legend_labels = list(legend_dict.keys())
legend_handles = [legend_dict[label] for label in legend_labels]

fig.legend(legend_handles, legend_labels, title='Season', loc='upper right')

plt.tight_layout()
plt.show()

# Filter the DataFrame where 'cluster' column equals 1
filtered_data = data_1994_to_1999[data_1994_to_1999['cluster'] == 2]

# Calculate the percentage of occurrence of each unique entry in 'season'
season_counts = filtered_data['season'].value_counts(normalize=True) * 100

# Print or use these percentages as needed
print("Percentage of Occurrence of Each Season:")
print(season_counts)
