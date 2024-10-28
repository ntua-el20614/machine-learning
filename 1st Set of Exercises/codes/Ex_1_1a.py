import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading
csv_path = '..\\ML2023-24-hwk1.csv'  
dataset = pd.read_csv(csv_path, delimiter=';')

# Data normalization
feature_columns = dataset.columns[:-1]  # Excluding target column
normalized_features = (dataset[feature_columns] - dataset[feature_columns].mean()) / dataset[feature_columns].std()
normalized_df = pd.DataFrame(normalized_features, columns=feature_columns)

# Correlation calculation
correlation_norm = normalized_df.corr()
ninth_tenth_corr = correlation_norm.iloc[8, 9]  # Correlation between 9th and 10th columns
print(f"Correlation between 9th and 10th Features: {ninth_tenth_corr}")
print(f"Correlation Matrix:\n{correlation_norm}")


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_norm, annot=True, fmt=".2f", cmap='coolwarm', square=True)
heatmap_path = '..\\plots\\norm_corr_heatmap.png'  
plt.savefig(heatmap_path)
plt.close()
 

plt.figure(figsize=(6, 6))
sns.scatterplot(x='pH', y='sulphates', data=normalized_df)
scatter_path = '..\\plots\\pH_sulphates_plot.png'  
plt.savefig(scatter_path)
plt.close()
