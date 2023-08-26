import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the datasets
data_1994_to_1999 = pd.read_pickle('../datasets/zurich_weather_clustered_1994_1999.pkl')
data_2017_to_2022 = pd.read_pickle('../datasets/zurich_weather_clustered_2017_2022.pkl')

# Assuming the data contains a column named 'month' and a column named 'cluster'
# You may need to adjust these column names based on your actual dataset structure

# Group data by month and cluster and calculate the average percentage composition of each cluster per month
def calculate_cluster_percentage(data):
    grouped = data.groupby(['month', 'cluster']).size().unstack(fill_value=0)
    total = grouped.sum(axis=1)
    percentage = (grouped.T / total).T * 100
    return percentage

# Find unique cluster values in your data
unique_clusters_1994_to_1999 = data_1994_to_1999['cluster'].unique()
unique_clusters_2017_to_2022 = data_2017_to_2022['cluster'].unique()

# Create subplots in a loop
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

datasets = [(data_1994_to_1999, unique_clusters_1994_to_1999), (data_2017_to_2022, unique_clusters_2017_to_2022)]
titles = ['1994-1999 Average Cluster Composition per Month', '2017-2022 Average Cluster Composition per Month']

# Initialize a list to store the handles for legend labels
legend_handles = []

# Initialize a list to store the colormap values for clusters 0, 1, 2, 3 with colors similar to Viridis
cluster_colormap_values = sns.color_palette("Set3", n_colors=4)

for i, (data, unique_clusters) in enumerate(datasets):
    ax = axes[i]
    percentage = calculate_cluster_percentage(data)
    percentage = percentage[unique_clusters]  # Select available clusters
    
    
    percentage.plot(kind='bar', stacked=True, ax=ax, width=0.7, color=cluster_colormap_values)
    ax.set_title(titles[i])
    ax.set_xlabel('Month')
    ax.set_ylabel('% Composition')
    ax.set_xticks(np.arange(12))  # Set ticks for 12 months
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # Remove the legend for this subplot
    ax.get_legend().remove()
    # Set cluster legends as '0', '1', '2', etc.
    legend_labels = [str(cluster) for cluster in unique_clusters]
    
    # Get the handles (containers) of the bars from the last subplot
    if i == len(datasets) - 1:
        handles = ax.containers
        legend_handles.extend(handles)  # Extend the list of handles for the final legend

# Create a single custom legend for both graphs manually outside of the subplots
fig.legend(handles=legend_handles, labels=legend_labels, title='Cluster', loc='lower right')

# Show the plots
plt.show()
