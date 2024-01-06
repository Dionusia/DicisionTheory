import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from math import sqrt

df = pd.read_csv('BreastTissue.csv')

#column class has values 1-6
class_labels = range(1, 7)

#class pairs
class_pairs = [(class1, class2) for class1 in class_labels for class2 in class_labels if class1 < class2]

all_features = ['Case #', 'I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP', 'DR', 'P']

#dataframe to store
results_df = pd.DataFrame(columns=['Feature', 'T-Statistic', 'P-Value'])

for feature in all_features:
    t_stat_list = []
    p_value_list = []
    for class1, class2 in class_pairs:
        class1_data = df[df['Class'] == class1][feature]
        class2_data = df[df['Class'] == class2][feature]
        stat, p_value = ttest_ind(class1_data, class2_data)
        t_stat_list.append(stat)
        p_value_list.append(p_value)
    
    #t-statistic and p-value
    avg_t_stat = sum(t_stat_list) / len(t_stat_list)
    avg_p_value = sum(p_value_list) / len(p_value_list)

    results_df = results_df.append({'Feature': feature, 'T-Statistic': avg_t_stat, 'P-Value': avg_p_value}, ignore_index=True)

print("Results Table:")
print(results_df)

#4 most important features with the smallest p-values
top_features = results_df.nsmallest(4, 'P-Value')['Feature'].tolist()

#print names of 4 most important features
print("\n4 Most Important Features:")
for feature in top_features:
    print(feature)


X = df[top_features].values
Y = df['Class'].values

#standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split the data using 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True)
df_metrics_NB = pd.DataFrame(columns=['Fold', 'Sensitivity', 'Specificity'])

df_metrics = pd.DataFrame(columns=['Fold', 'Specificity', 'Sensitivity'])

gnb = GaussianNB()

for i, (train, test) in enumerate(kfold.split(X), 1):
    #fit the model
    gnb.fit(X[train], Y[train])

    #predict
    y_pred_NB = gnb.predict(X[test])

    #confusion matrix
    cm_nb = metrics.confusion_matrix(Y[test], y_pred_NB)

    total_nb = sum(sum(cm_nb))
    sensitivity_nb = cm_nb[1, 1] / (cm_nb[1, 0] + cm_nb[1, 1] + 1e-10)
    specificity_nb = cm_nb[0, 0] / (cm_nb[0, 0] + cm_nb[0, 1] + 1e-10)

    df_metrics_NB.loc[len(df_metrics_NB.index)] = [i, sensitivity_nb, specificity_nb]

geometric_mean_NB = sqrt(df_metrics_NB['Specificity'].mean() * df_metrics_NB['Sensitivity'].mean())

print("\nMetrics for Naive Bayes:")
print(df_metrics_NB)

print("\nGeometricMean after students t-test:", geometric_mean_NB)
