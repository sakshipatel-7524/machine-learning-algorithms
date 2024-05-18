import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score

#training from data to find optimal weight vector
def perceptron_classification(training_X,training_y):
    mistake_count=0
    sz=training_X.shape[0]
    w=np.zeros(training_X.shape[1])  #inital guess,weight vector
    for it in range(100):
        i=random.randint(0,sz-1)
        h=np.sign(np.dot(w,training_X[i,:]))
        if h!=training_y[i]:
            w+=np.dot(training_y[i],training_X[i])
            w=w/np.linalg.norm(w)
            mistake_count+=1
            print(w)
                
    return w,mistake_count

#Testing the data
def misclassification_count(testing_X,testing_y,trained_w):
    misClasscount=0
    y_pred=[]
    for i in range(len(testing_X)):
        x=testing_X[i]
        y=testing_y[i]
        h=np.sign(np.dot(trained_w,x))     
        y_pred.append(h)   
        print("Actual:",y," Predicted:",h)
        if h!=y :
            misClasscount +=1
            print("Mis-classified here")
    return y_pred,misClasscount

# generate a binary classification dataset with 2 classes and 5 features
X,y=make_classification(n_samples=100, n_features=5, n_classes=2)
#make classes {-1,1} instead of {0,1}
y=2*y-1 

#normalize the data set  of X with its own norm xi
for i in range(100):
    X[i]=X[i]/(np.linalg.norm(X[i,:])) #preprocessing the data

#split test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#call for training
weight_vector,mistakeCount=perceptron_classification(X_train,y_train)
#display results
print("Optimal Weight Vector:",weight_vector)
print("Mistakes Counted by Perceptron Algorithm is ",mistakeCount)


#call for testing the data set
y_pred,misclassification_count=misclassification_count(X_test,y_test,weight_vector)
print("\nMisclassifications Counted using Test Data is",misclassification_count)

#decision boundary
y_predicted=np.multiply(np.transpose(weight_vector),X_test)

# Create a single figure and axis object
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Overall Data Set
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, label="Overall data set")
axes[0, 0].set_title("Overall Data Set")
axes[0, 0].set_xlabel("Feature 1")
axes[0, 0].set_ylabel("Feature 2")
axes[0, 0].legend()

# Plot 2: Splitted Data Set
axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='^', edgecolors='black', label="training data set") 
axes[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='*', edgecolors='green', label="testing data set")
axes[0, 1].set_title("Splitted Data Set")
axes[0, 1].set_xlabel("Feature 1")
axes[0, 1].set_ylabel("Feature 2")
axes[0, 1].legend()

# Plot 3: Splitted Data Set with Predicted Labels
axes[1, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='*', edgecolors='black', label="training data set") 
axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', edgecolors='red', label="testing data set")
axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='o', edgecolors='green', label="predicted data set")
axes[1, 0].set_title("With Predicted DataSet")
axes[1, 0].set_xlabel("Feature 1")
axes[1, 0].set_ylabel("Feature 2")
axes[1, 0].legend()

# Plot 4: Decision Boundary
axes[1, 1].scatter(X[:, 0], X[:, 1], c=y, label="Overall data set")
xmin, xmax = plt.xlim()
ymin, ymax=plt.ylim()
x_range = np.linspace(xmin, xmax, 100)
y_range=np.linspace(ymin, ymax, 100)
Y_dec_boundary = weight_vector[0] * x_range + weight_vector[1] * y_range
axes[1, 1].plot(x_range, Y_dec_boundary, color='red')
axes[1, 1].set_title("Decision Boundary")
axes[1, 1].set_xlabel("Feature 1")
axes[1, 1].set_ylabel("Feature 2")
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
disp=ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.title("Confusion Matrix")
plt.show()






