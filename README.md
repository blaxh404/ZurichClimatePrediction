# ZurichClimatePrediction

The aim of this project is to showcase my ability to establish predictive models,
evaluate the model, display statistics and provide conclusions.

I chose to work on Zurich climatic data from my birthyear 1994 until today to make it more personal

Currently this project contains 2 analytical branches

>> 1) Logistic regression model to predict if rain will occur the next day based on historical data -
>> 2) Clustering of climatic data in order to understand similarities and variations
    between 1994-1999 and 2017-2022. The aim is to charactarize each cluster in an unbiased manner
    and to get numbers on global warming here in Zurich


1) For now the predictive accuracy achieves 74%, it can (and will) be widely improved
using better feature selection/engineering and change reorganization of the data
The model currently performs better on predicting rainy events (81%) than non rainy events (65%)
The dataset is surprisingly very balanced with 50% of rainy days in the past 29 years.
A rainy day is characterized by > 0mm of rain in that day.


2) The clustering has been made using Kmeans and the number of cluster
has been selected using the silhouette score approach
Even though those clusters don't separate the data very well, they capture general trends
and stratify the dataset in an unsupervised manner. I could add a metric such as inertia or davies-bouldin index
to give information about the performance of the clustering.

Each cluster somewhat describe each season. We see that cluster 1 is enriched in summer days
while cluster 4 is enriched in winter days. The other clusters are not so clear.
Maybe cluster 2 is closer to describe spring climate while cluster 3 contains a high mix of winter and fall days.
The cluster composition per month shows that for the 1994-1999 dataset, cluster 4 is close
to describing winter wheras for the 2017-2022 dataset, it appears to be cluster 3
Further characterization is required to understand the composition of those clusters and certainly,
improve the clustering with hyperparameter tuning and different feature selection.

Nevertheless, cluster 1 clearly describe a warm climate which spans across summer and beyond.
What the significance test shows is that in the past 20 years we have a significant increase in temperature
in warm climate of more than 1 degree. In fact in any climates stratified by the clusters
we see a significant increase in temperature.
precipitations also increase especially in cluster describing cooler temperatures.