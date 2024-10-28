import numpy as np
import pandas as pd

# Data loading
dataset = pd.read_csv('..\\ML2023-24-hwk1.csv', delimiter=';')
train_data = dataset.iloc[:100, :]
validate_data = dataset.iloc[100:150, :]

# Separating features and target variable
features_train = train_data.iloc[:, :-1].values
target_train = train_data.iloc[:, -1].values

# Adding bias term to features
bias_added_features = np.hstack([np.ones((features_train.shape[0], 1)), features_train])

# Linear regression weights calculation
regression_weights = np.linalg.pinv(bias_added_features.T @ bias_added_features) @ bias_added_features.T @ target_train

# Extracting feature names and printing weights
column_names = train_data.columns[:-1].tolist()
for index in range(min(len(column_names), len(regression_weights))):
    print(f"Weights for {column_names[index]}: {regression_weights[index]}")
