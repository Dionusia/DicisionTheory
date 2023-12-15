import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('BreastTissue.csv')

# Split into input and output
X = df.drop('Class', axis=1)
y = df['Class'].values

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing data using 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# DataFrames to store metrics
df_metrics_NB = pd.DataFrame(columns=['Fold', 'Sensitivity', 'Specificity'])
df_metrics_SVM = pd.DataFrame(columns=['Fold', 'Sensitivity', 'Specificity'])
df_metrics_KNN = pd.DataFrame(columns=['Fold', 'Sensitivity', 'Specificity'])

# Models
gnb = GaussianNB()
clf_svm = SVC(C=21, gamma=0.5, kernel='rbf')
knn = KNeighborsClassifier(n_neighbors=9, metric='l1')

for i, (train, test) in enumerate(kfold.split(X)):
    # Fit the models
    gnb.fit(X[train], y[train])
    clf_svm.fit(X[train], y[train])
    knn.fit(X[train], y[train])
    
    # Predictions
    y_pred_NB = gnb.predict(X[test])
    y_pred_SVM = clf_svm.predict(X[test])
    y_pred_KNN = knn.predict(X[test])
    
    # Confusion matrices
    cm_nb = metrics.confusion_matrix(y[test], y_pred_NB)
    cm_svm = metrics.confusion_matrix(y[test], y_pred_SVM)
    cm_knn = metrics.confusion_matrix(y[test], y_pred_KNN)
    
    # Calculate metrics
    # NB
    total_nb = sum(sum(cm_nb))
    sensitivity_nb = cm_nb[1, 1] / (cm_nb[1, 0] + cm_nb[1, 1] + 1e-10)
    specificity_nb = cm_nb[0, 0] / (cm_nb[0, 0] + cm_nb[0, 1] + 1e-10)
    
    # SVM
    total_svm = sum(sum(cm_svm))
    sensitivity_svm = cm_svm[1, 1] / (cm_svm[1, 0] + cm_svm[1, 1] + 1e-10)
    specificity_svm = cm_svm[0, 0] / (cm_svm[0, 0] + cm_svm[0, 1] + 1e-10)
    
    # KNN
    total_knn = sum(sum(cm_knn))
    sensitivity_knn = cm_knn[1, 1] / (cm_knn[1, 0] + cm_knn[1, 1] + 1e-10)
    specificity_knn = cm_knn[0, 0] / (cm_knn[0, 0] + cm_knn[0, 1] + 1e-10)
    
    # Append values to dataframes
    df_metrics_NB.loc[len(df_metrics_NB.index)] = [i, sensitivity_nb, specificity_nb]
    df_metrics_SVM.loc[len(df_metrics_SVM.index)] = [i, sensitivity_svm, specificity_svm]
    df_metrics_KNN.loc[len(df_metrics_KNN.index)] = [i, sensitivity_knn, specificity_knn]

# Print the DataFrames
print("\nMetrics for Naive Bayes:")
print(df_metrics_NB)

print("\nMetrics for SVM:")
print(df_metrics_SVM)

print("\nMetrics for KNN:")
print(df_metrics_KNN)

# Calculate the mean of the metrics
geometric_mean_NB = sqrt(df_metrics_NB['Specificity'].mean() * df_metrics_NB['Sensitivity'].mean())
geometric_mean_SVM = sqrt(df_metrics_SVM['Specificity'].mean() * df_metrics_SVM['Sensitivity'].mean())
geometric_mean_KNN = sqrt(df_metrics_KNN['Specificity'].mean() * df_metrics_KNN['Sensitivity'].mean())

print("\nGeometric Mean (SVM):", geometric_mean_SVM)
print("Geometric Mean (Naive Bayes):", geometric_mean_NB)
print("Geometric Mean (KNN):", geometric_mean_KNN)
