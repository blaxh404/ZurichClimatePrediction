import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import table


file_path = 'data/Zurich_weather_cleaned.pkl'

cleaned_data = pd.read_pickle(file_path)

# Calculate a 3-day rolling mean for the 'precip' column
cleaned_data['precip_rolling_mean_3'] = cleaned_data['precip'].rolling(window=3).mean()

# Calculate a 3-day rolling mean for the 'sealevelpressure' column
cleaned_data['sealevelpressure_rolling_mean_3'] = cleaned_data['sealevelpressure'].rolling(window=3).mean()

# Add a lagged value of 1 day for the 'humidity' column
cleaned_data['humidity_lag_1'] = cleaned_data['humidity'].shift(1)

# Drop rows with missing values (resulting from rolling means and lag)
cleaned_data.dropna(inplace=True)

# Split the dataset into features (X) and target (y)
X = cleaned_data.drop(columns=['rainTomorrow'])
y = cleaned_data['rainTomorrow']

###### Doing some feature selection to improve the model

# selector = SelectKBest(score_func=mutual_info_classif, k=7)
# X_new = selector.fit_transform(X, y)
# selected_indices = selector.get_support(indices=True)
# selected_features = X.columns[selected_indices]
# print("Selected Features:", selected_features)

# X = X_new
######


X = preprocessing.StandardScaler().fit(X).transform(X)

# Calculate the split index based on the desired split ratio
split_index = int(len(cleaned_data) * 0.8)

# Split the data into a training set and a testing set
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train a logistic regression model
model = LogisticRegression(C=0.01, solver='liblinear')
model.fit(X_train, y_train)

# Predict rain tomorrow based on the testing set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred).round(3)
precision = precision_score(y_test, y_pred).round(3)
recall = recall_score(y_test, y_pred).round(3)
f1 = f1_score(y_test, y_pred).round(3)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Create subplots
fig, ax = plt.subplots(3, 1, figsize=(6, 13))  # Three subplots: Confusion Matrices and Table

# Plot the confusion matrix with percentages using Seaborn (left subplot)
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size": 12}, ax=ax[0])
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
ax[0].set_title('Confusion Matrix (Percentages)')

# Plot the confusion matrix without percentages using Seaborn (middle subplot)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, ax=ax[1])
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
ax[1].set_title('Confusion Matrix (Counts)')

# Create a DataFrame for the performance metrics
metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
}

metrics_df = pd.DataFrame(metrics_data)

# Create a table with performance metrics (right subplot)
ax[2].axis('off')  # Hide axis for the table subplot
tab = table(ax[2], metrics_df, loc='center', colWidths=[0.2, 0.2])
tab.auto_set_font_size(False)
tab.set_fontsize(12)
tab.scale(1, 1.5)  # Adjust the table size as needed

plt.tight_layout()
plt.show()