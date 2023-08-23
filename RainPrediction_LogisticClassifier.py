import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data/Zurich_weather_cleaned.pkl'

cleaned_data = pd.read_pickle(file_path)



# Split the dataset into features (X) and target (y)
X = cleaned_data.drop(columns=['rainTomorrow'])
y = cleaned_data['rainTomorrow']

###### Doing some feature selection to improve the model

# selector = SelectKBest(score_func=mutual_info_classif, k=5)
# X_new = selector.fit_transform(X, y)
# selected_indices = selector.get_support(indices=True)
# selected_features = X.columns[selected_indices]
# print("Selected Features:", selected_features)

# X = X_new
######

X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create and train a logistic regression model
model = LogisticRegression(C=0.01, solver='liblinear')
model.fit(X_train, y_train)

# Predict rain tomorrow based on the testing set
y_pred = model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot the confusion matrix with percentages using Seaborn
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size": 12}, ax=ax[0])
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
ax[0].set_title('Confusion Matrix (Percentages)')

# Plot the confusion matrix without percentages using Seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, ax=ax[1])
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
ax[1].set_title('Confusion Matrix (Counts)')

plt.tight_layout()
plt.show()



