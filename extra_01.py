###
# Created on Oct 03 2017
#
# Classify iris flowers as one of three possible species.
#
# The Iris dataset was used in R.A. Fisher's classic 1936 paper, 
# 'The Use of Multiple Measurements in Taxonomic Problems', and 
# can also be found on the UCI Machine Learning Repository.
#
# @author: sarahv
###


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the iris data from csv file as DataFrame object
iris = pd.read_csv('E:\Research\Courses\Statistical Learning\Iris data\Iris.csv')


##### Data visualization #####

# Estimated densities per feature and pairwise relationships
g = sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i,j].set_visible(False)
plt.show()

# We can see that PetalLengthCm and PetalWidthCm are more useful features for
# distinguishing classes than SepalWidthCm due to the strong separation between
# densities

# Consider new features by combining other features


##### Data processing #####

# Convert feature columns to an array of data
X = np.array(iris[ [col for col in iris.columns if col != 'Id' and col != 'Species'] ])
N, m = np.shape(X)

# Construct classes
# Looking at the data determines the # of species, not hard coded
Y = np.array(iris['Species'])
Yn = np.zeros((N,))
species = list(set(Y)) # This encodes relationship
Nc = len(species)
for i in range(N):
    Yn[i] = species.index(Y[i]) + 1

# Split the data set into a training and test set: 70/30, randomly chosen
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Yn, test_size = 0.3)
Ntrain = len(Xtrain)
Ntest = N - Ntrain


##### Perform classification #####

# Train a k-nearest neighbors classifier using different neighborhood size k
# Think about visualizing the classification results
# What is an appropriate k and whether to weight uniformly vs distance
ks = [1,3,5,7,15]
accuracies = np.zeros((5,2))
for j in range(5):
    KNN = KNeighborsClassifier(n_neighbors=ks[j], weights = 'distance')
    KNN.fit(Xtrain,Ytrain)
    # Prediction using training data
    trainy_pred = KNN.predict(Xtrain)
    accuracies[j,0] = (Ntrain-np.sum(np.count_nonzero(Ytrain-trainy_pred)))/Ntrain
    # Prediction using test data
    testy_pred = KNN.predict(Xtest)
    accuracies[j,1] = (Ntest-np.sum(np.count_nonzero(Ytest-testy_pred)))/Ntest
    
print("KNN accuracy, training set:"); print(accuracies[:,0])
print("KNN accuracy, test set:"); print(accuracies[:,1])