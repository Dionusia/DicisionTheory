import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib import pyplot

# Load the dataset
df = pd.read_csv('BreastTissue.csv')

# Split into input and output
X = df.drop('Class', axis=1)
y = df['Class'].values

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot

# Split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Search for the best C in [1-200] with step=5 and using linear kernel
param_grid_linear = {'C': list(range(1, 200, 5))}
grid_search_linear = GridSearchCV(SVC(kernel='linear'), param_grid_linear, cv=5)
grid_search_linear.fit(X_train, y_train)

best_C_linear = grid_search_linear.best_params_['C']

# Print results for linear kernel
print("Best C for Linear Kernel:", best_C_linear)

# Search for the best gamma in [0:10] with step 0.5 and with RBF kernel
param_grid_rbf = {'gamma': np.arange(0, 10.5, 0.5)}
grid_search_rbf = GridSearchCV(SVC(kernel='rbf', C=best_C_linear), param_grid_rbf, cv=5)
grid_search_rbf.fit(X_train, y_train)

best_gamma_rbf = grid_search_rbf.best_params_['gamma']

# Print results for RBF kernel
print("Best Gamma for RBF Kernel:", best_gamma_rbf)

# Plot results for linear kernel
linear_results = pd.DataFrame(grid_search_linear.cv_results_)
pyplot.figure(figsize=(12, 6))
pyplot.plot(linear_results['param_C'], linear_results['mean_test_score'], label='Linear Kernel')
pyplot.ylabel("Accuracy")
pyplot.xlabel("C")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

# Plot results for RBF kernel
rbf_results = pd.DataFrame(grid_search_rbf.cv_results_)
pyplot.figure(figsize=(12, 6))
pyplot.plot(rbf_results['param_gamma'], rbf_results['mean_test_score'], label='RBF Kernel')
pyplot.ylabel("Accuracy")
pyplot.xlabel("Gamma")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()
