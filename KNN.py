import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('BreastTissue.csv')

X = df.drop('Class', axis=1)
y = df['Class'].values

#standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#dataframes to store metrics
df_results_KNN = pd.DataFrame(columns=['k', 'Accuracy'])

#parameter
param_grid = {'n_neighbors': list(range(3, 16))}

for k in param_grid['n_neighbors']:
    knn = KNeighborsClassifier(n_neighbors=k)

    accuracy_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

    df_results_KNN.loc[len(df_results_KNN.index)] = [k, accuracy_scores.mean()]

print("\nResults for KNN:")
print(df_results_KNN)

#plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x='k', y='Accuracy', data=df_results_KNN, marker='o')
plt.title('Results for KNN - Varying k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(param_grid['n_neighbors'])
plt.show()
