import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#load csv
df = pd.read_csv("BreastTissue.csv")

#Class is my target
X = df.drop('Class', axis=1)
y = df['Class']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train naive bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

#train SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

#train KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

#evaluate NB
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions, average='weighted')
nb_precision = precision_score(y_test, nb_predictions, average='weighted')

#evaluate SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_precision = precision_score(y_test, svm_predictions, average='weighted')

#evaluate KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_precision = precision_score(y_test, knn_predictions, average='weighted')

print("Naive Bayes:")
print("Accuracy:", nb_accuracy)
print("Recall:", nb_recall)
print("Precision:", nb_precision)

print("\nSVM:")
print("Accuracy:", svm_accuracy)
print("Recall:", svm_recall)
print("Precision:", svm_precision)

print("\nKNN:")
print("Accuracy:", knn_accuracy)
print("Recall:", knn_recall)
print("Precision:", knn_precision)
