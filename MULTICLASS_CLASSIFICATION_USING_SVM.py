import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = fetch_openml('iris', version=1, parser = 'auto')
x, y = iris.data.astype('float32'), iris.target.astype('string')

# Displaying input and output dataset
x_total = np.c_[x,y]
df = pd.DataFrame(data = x_total, columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'output'])
print(df)

# Taking only 2 features = petallength and petalwidth
x = x.iloc[:,2:4].to_numpy()
y = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

# Using Logistic regression
print("========== Using Logistic Regression =========")
clf = LogisticRegression().fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Predicted values:")
print(y_pred)
score = clf.score(x_test,y_test)
print("Accuracy:", score)

# Using Linear SVM
print("\n========== Using Linear SVM =========")
lsvm = SVC(kernel='linear')
lsvm.fit(x_train, y_train)
y_pred_lsvm = lsvm.predict(x_test)
print("Predicted values:")
print(y_pred_lsvm)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Using Soft SVM
print("\n========== Using Soft SVM =========")
ssvm = SVC(kernel='linear',C=1.0)
ssvm.fit(x_train, y_train)
y_pred_ssvm = ssvm.predict(x_test)
print("Predicted values:")
print(y_pred_ssvm)
accuracy_ssvm = accuracy_score(y_test, y_pred_ssvm)
print("Accuracy:", accuracy_ssvm)


