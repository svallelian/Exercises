###
# Created on Aug 15 2017
#
# "The Elements of Statistical Learning" - Exercise 2.8
#
# Compare linear regression and k-nearest neighbors on handwritten digits (zip code data)
# Data available online at: www-stat.stanford.edu/ElemStatLearn
#
# @author: sarahv
###

import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error


# Load the zip training data
y = np.loadtxt('E:\Research\Courses\Statistical Learning\ZIP data\zip.train', usecols=(0,));
x = np.loadtxt('E:\Research\Courses\Statistical Learning\ZIP data\zip.train', usecols=range(1,257));

# Select the 2's and 3's
indices2 = np.nonzero(y==2);
indices3 = np.nonzero(y==3);
trainy = np.concatenate((y[indices2],y[indices3]),axis=0);
trainx = np.concatenate((np.squeeze(x[indices2,:]), np.squeeze(x[indices3,:])),axis=0);

# Load the test data
y = np.loadtxt('E:\Research\Courses\Statistical Learning\ZIP data\zip.test', usecols=(0,));
x = np.loadtxt('E:\Research\Courses\Statistical Learning\ZIP data\zip.test', usecols=range(1,257));

# Select the 2's and 3's
indices2 = np.nonzero(y==2);
indices3 = np.nonzero(y==3);
testy = np.concatenate((y[indices2],y[indices3]),axis=0);
testx = np.concatenate((np.squeeze(x[indices2,:]), np.squeeze(x[indices3,:])),axis=0);


# Linear regression
linreg = linear_model.LinearRegression();
linreg.fit(trainx, trainy);

# Prediction using training data
trainy_pred = linreg.predict(trainx);
print("Linear Regression MSE, training set: %.6f" % mean_squared_error(trainy, trainy_pred));

# Prediction using test data
testy_pred = linreg.predict(testx);
print("Linear Regression MSE, test set: %.6f" % mean_squared_error(testy, testy_pred));


# KNN for different numbers of neighbors
ks = [1,3,5,7,15];
MSEs = np.zeros((5,2));
for j in range(5):
    KNN = KNeighborsClassifier(n_neighbors=ks[j]);
    KNN.fit(trainx,trainy);
    # Prediction using training data
    trainy_pred = KNN.predict(trainx);
    MSEs[j,0] = mean_squared_error(trainy, trainy_pred);
    # Prediction using test data
    testy_pred = KNN.predict(testx);
    MSEs[j,1] = mean_squared_error(testy, testy_pred);
    
print("KNN MSE, training set:"); print(MSEs[:,0]);
print("KNN MSE, test set:"); print(MSEs[:,1]);