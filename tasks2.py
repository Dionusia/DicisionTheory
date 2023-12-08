from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd

#load csv
df = pd.read_csv("BreastTissue.csv")

#Class is my target
X = df.drop('Class', axis=1)
y = df['Class']

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, train_size=0.7)

#standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#create and train KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#create and train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

#create and train SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

#make predictions
knn_pred = knn.predict(X_test)
nb_pred = nb_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

#extract unique labels from predictions to avoid error
knn_labels = sorted(set(knn_pred))
nb_labels = sorted(set(nb_pred))
svm_labels = sorted(set(svm_pred))

#evaluate and print for KNN
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred, average='weighted')
knn_precision = precision_score(y_test, knn_pred, average='weighted', labels=knn_labels)
print("\nKNN:")
print("Accuracy:", knn_accuracy)
print("Recall:", knn_recall)
print("Precision:", knn_precision)

#evaluate and print for Naive Bayes
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred, average='weighted')
nb_precision = precision_score(y_test, nb_pred, average='weighted', labels=nb_labels)
print("\nNaive Bayes:")
print("Accuracy:", nb_accuracy)
print("Recall:", nb_recall)
print("Precision:", nb_precision)

#evaluate and print for SVM
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred, average='weighted')
svm_precision = precision_score(y_test, svm_pred, average='weighted', labels=svm_labels)
print("\nSVM:")
print("Accuracy:", svm_accuracy)
print("Recall:", svm_recall)
print("Precision:", svm_precision)
