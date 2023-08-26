import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load the datasets
data_1994_to_1999 = pd.read_pickle('../datasets/zurich_weather_clustered_1994_1999.pkl')
data_2017_to_2022 = pd.read_pickle('../datasets/zurich_weather_clustered_2017_2022.pkl')

# Create a 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Define the colormap for clusters with reordered rows
cluster_colormap_values = sns.color_palette("Set3", n_colors=4)
cluster_colormap_values = [cluster_colormap_values[3], cluster_colormap_values[1], cluster_colormap_values[0], cluster_colormap_values[2]]

# Define the cluster order
cluster_order = [1, 2, 3, 4]

# Iterate through each dataset and subplot
datasets = [(data_1994_to_1999, '1994-1999'), (data_2017_to_2022, '2017-2022')]
for i, (data, title) in enumerate(datasets):
    # Create violin plots for Temperature ('temp')
    sns.violinplot(data=data, x='cluster', y='temp', ax=axes[0, i], inner='quartile', scale='width', palette=cluster_colormap_values, order=cluster_order)
    axes[0, i].set_title(f'Temperature ({title})')
    axes[0, i].set_xlabel('Cluster')
    axes[0, i].set_ylabel('Temperature (°C)')

    # Calculate and overlay average ± standard deviation on Temperature ('temp')
    cluster_means = data.groupby('cluster')['temp'].mean()
    cluster_std = data.groupby('cluster')['temp'].std()
    for j, cluster in enumerate(cluster_order):
        axes[0, i].text(j, cluster_means[cluster], f'{cluster_means[cluster]:.2f} ± {cluster_std[cluster]:.2f}', ha='center', va='center', fontsize=10, color='black')

    # Create violin plots for Precipitations ('precip') with wider violins
    sns.violinplot(data=data, x='cluster', y='precip', ax=axes[1, i], inner='quartile', scale='width', palette=cluster_colormap_values, order=cluster_order)
    axes[1, i].set_title(f'Precipitations ({title})')
    axes[1, i].set_xlabel('Cluster')
    axes[1, i].set_ylabel('Precipitations (mm)')

    # Calculate and overlay average ± standard deviation on Precipitations ('precip')
    cluster_means = data.groupby('cluster')['precip'].mean()
    cluster_std = data.groupby('cluster')['precip'].std()
    for j, cluster in enumerate(cluster_order):
        axes[1, i].text(j, cluster_means[cluster], f'{cluster_means[cluster]:.2f} ± {cluster_std[cluster]:.2f}', ha='center', va='center', fontsize=10, color='black')

# Create an empty list to store the results
results = []

# Perform t-tests for each cluster and variable (Temperature and Precipitations)
for cluster in cluster_order:
    for variable in ['temp', 'precip']:
        cluster_1994_to_1999 = data_1994_to_1999[data_1994_to_1999['cluster'] == cluster][variable]
        cluster_2017_to_2022 = data_2017_to_2022[data_2017_to_2022['cluster'] == cluster][variable]

        # Perform t-test
        t_stat, p_value = ttest_ind(cluster_1994_to_1999, cluster_2017_to_2022, equal_var=False)

        # Calculate -log(p-value) and round to 2 significant figures
        neg_log_pvalue = round(-np.log10(p_value), 2)

        # Append the results as a dictionary to the list
        results.append({'Cluster comparison across datasets': f'Cluster {cluster}', 'Variable': variable, '-log(p-value)': neg_log_pvalue})

# Create a DataFrame from the list of dictionaries
results_df = pd.DataFrame(results)

# Plot the table with larger text in cells
ax = axes[0, 2]
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2', '#f2f2f2', '#f2f2f2'])

# Set font size for the table cells
table.auto_set_font_size(False)
table.set_fontsize(10)

# Adjust subplot layout
plt.tight_layout()

# Show the plot
plt.show()
