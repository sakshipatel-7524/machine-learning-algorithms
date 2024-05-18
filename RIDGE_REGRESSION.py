import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x, y = make_regression(n_samples=100, n_features=1, bias = 0, noise =5,random_state=40)
# Scale feature x to range -5…..5
x = np.interp(x, (x.min(), x.max()), (-5, 5))
# Scale target y to range 15…..-15
y = np.interp(y, (y.min(), y.max()), (15, -15))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

hyper_parameter=[0.1,0.5,1.0,1.5]
def ridge_regression(x_train,y_train,hparam):
    
    X=np.c_[[1]*len(x_train),x_train]
    XT=np.transpose(X)
    XT_X=np.matmul(XT,X)
    I=np.identity(X.shape[1])
    # print(X.shape[1])
    Regularization_term=(I*hparam)
    XT_X_aI=np.add(XT_X,Regularization_term)
    XT_X_aI_inv=np.linalg.inv(XT_X_aI)
    XT_y=np.matmul(XT,y_train)
    weight_vector=np.matmul(XT_X_aI_inv,XT_y)
    # print("w*=\n",weight_vector)
    return weight_vector
weight_vector_all=[]
for i in hyper_parameter:
    weight_vector=ridge_regression(x_train,y_train,hparam=i)
    weight_vector_all.append(ridge_regression(x_train,y_train,hparam=i))

data={'Hyperparameter':hyper_parameter,'Weight Vector':weight_vector_all}
df=pd.DataFrame(data)
print("Result :")
print(df)

y_te_predicted=np.dot(x_test,weight_vector[1])+weight_vector[0]
print("\n\nPrediction on testing data set=\n",np.transpose(y_te_predicted))
result=mean_squared_error(y_true=y_test,y_pred=y_te_predicted)
print("Root Mean Squared Error on Prediction and Testing data set: ", result)
# print(np.shape(y_test),np.shape(y_predicted))

y_tr_predicted=np.dot(x_train,weight_vector[1])+weight_vector[0]
print("\n\nPrediction on training data set=\n",np.transpose(y_tr_predicted))
result=mean_squared_error(y_true=y_train,y_pred=y_tr_predicted)
print("Root Mean Squared Error on Prediction and Training data set: ", result)

y_predicted=np.dot(x,weight_vector[1])+weight_vector[0]

print("\n\n\nUsing function Ridge() from scikit-learn:")
for i in hyper_parameter:
    rid_regr=linear_model.Ridge(alpha=i)
    rid_regr.fit(x_train,y_train)
    y_pred_inbuilt=preds=rid_regr.predict(x_test)
    print("\nHyper Parameter =",i)
    print('Prediction : \n' ,y_pred_inbuilt)
    result=mean_squared_error(y_true=y_test,y_pred=y_pred_inbuilt)
    print("Root Mean Squared Error on Testing and Prediction: ", result) 


# Plot 1: Overall Data Set
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(12, 10))
ax[0,0].scatter(x,y,label='Overall dataset')
ax[0,0].set_title('Ridge Regression Dataset')
ax[0,0].set_xlabel(r'Feature $x_1$')
ax[0,0].set_ylabel(r'Target $y$')
ax[0,0].legend

# Plot 2: Training and Testing
ax[0, 1].scatter(x_train, y_train, marker='^', label="training data set") 
ax[0, 1].scatter(x_test, y_test, marker='*', label="testing data set")
ax[0, 1].set_title("Splitted Data Set")
ax[0, 1].set_xlabel(r'Feature $x_1$')
ax[0, 1].set_ylabel(r'Target $y$')
ax[0, 1].legend()

# Plot 3: Training and Testing with regression line
ax[1, 0].scatter(x_train, y_train, marker='^', color='green' ,label="training data set") 
ax[1, 0].scatter(x_test, y_test, marker='*', color='blue',label="testing data set")
ax[1, 0].plot(x, y_predicted,'-', color='orange', label='prediction')
ax[1, 0].set_title("Splitted Data Set")
ax[1, 0].set_xlabel(r'Feature $x_1$')
ax[1, 0].set_ylabel(r'Target $y$')
ax[1, 0].legend()

ax[1, 1].scatter(x_test,y_test,label='Test dataset')
ax[1, 1].plot(x_test, y_pred_inbuilt,'-', color='orange', label='prediction')
ax[1, 1].set_title("Ridge Regression inbuilt")
ax[1, 1].set_xlabel(r'Feature $x_1$')
ax[1, 1].set_ylabel(r'Target $y$')
ax[1, 1].legend()



plt.show()

