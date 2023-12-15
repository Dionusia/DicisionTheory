import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('BreastTissue.csv')

# Split into input and output
X = df.drop('Class', axis=1)
y = df['Class'].values

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# DataFrames to store metrics
df_results_KNN = pd.DataFrame(columns=['k', 'Accuracy'])

# Parameter grid for GridSearchCV
param_grid = {'n_neighbors': list(range(3, 16))}

for k in param_grid['n_neighbors']:
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Calculate accuracy using cross-validation
    accuracy_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

    # Append mean accuracy to the dataframe
    df_results_KNN.loc[len(df_results_KNN.index)] = [k, accuracy_scores.mean()]

# Print the DataFrame
print("\nIntermediate Results for KNN:")
print(df_results_KNN)

# Plot the results
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x='k', y='Accuracy', data=df_results_KNN, marker='o')
plt.title('Intermediate Results for KNN - Varying k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(param_grid['n_neighbors'])
plt.show()
