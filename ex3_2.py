###
# Created on Sept 08 2017
#
# "The Elements of Statistical Learning" - Exercise 3.2
#
# Compare two approaches for computing confidence bands
#
# @author: sarahv
###

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as st

##### User-defined functions #####

# Define a data-generating function
# Creates a degree-3 polynomial on the interval [0,5]
# and adds Gaussian random noise to the output
def generateData(numpts,error,beta):
    """ numpts: positive integer, # of data points """
    """ error: nonnegative real, variance of additive error """
    """ beta: 4x1 column vector of true parameters """
    x = np.linspace(0,5,nx) # column vector
    X = np.stack((x**0, x, x**2, x**3),axis=1) # columns correspond to powers
    y = np.dot(X,beta)
    data = np.add(y, np.random.normal(0,error,(nx,)))
    return X, y, data

# These aren't strictly necessary as separate functions but makes 
# the code slightly easier to read    
def convertAtoT(alpha,df):
    """ Find 100*(1-alpha) percentile of Student's t distribution """
    """ alpha: real number in [0,1] """
    """ df: positive real number, degrees of freedom in distribution """
    return st.t.ppf(1-alpha,df)

def convertAtoF(alpha,df1,df2):
    """ Find 100*(1-alpha) percentile of the F-distribution """
    """ p: real number in [0,1] """
    """ df1, df2: positive real numbers, degrees of freedom in distribution"""
    return st.f.ppf(1-alpha,df1,df2)

def generateDiagP(X):
    """ X: Nx4 array of model inputs """
    """ Construct the diagonal of the projection matrix on X """
    s,qt = np.linalg.svd(X)[1:3] # 4x1, 4x4
    s = s**(-2)
    xq = np.dot(X,np.transpose(qt)) # Nx4 array
    xq = xq**(2)
    return np.dot(xq,s)


##### Example #####

# Generate the data
nx = 10 # size of X
pb = 4 # size of beta, this is built-in 
errorvar = 1
beta = np.array([15,-5,4,-0.5])
X,y,data = generateData(nx,errorvar,beta)
x = X[:,1]

# Generate the least-squares solution, hatbeta
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
model.fit(X,data)
hatbeta = np.array(beta)
hatbeta[0] = model.intercept_
hatbeta[1:] = model.coef_[1:]

y_pred = model.predict(X) # this is equal to np.dot(X,hatbeta)
line1, = plt.plot(x,y,'b',label='True y')
line2, = plt.plot(x,data,'o',label='Data')
line3, = plt.plot(x,y_pred,'k',label='Predicted y')
plt.title('Truth, data, and predicted function')
plt.legend(handles=[line1,line2,line3])
plt.show()

RSS = np.sum((y-y_pred)**2)
varest = RSS/(nx-pb) # unbiased estimator for variance of y
DP = generateDiagP(X)
width = (varest*DP)**(0.5)
alpha = 0.05 # confidence is 1 - alpha

# Approach 1: pointwise confidence bounds
# See e.g. Casella and Berger
c1 = convertAtoT(alpha/2,nx-pb)

# Approach 2: simultaneous confidence bounds
# See e.g. Casella and Berger
c2 = convertAtoF(alpha,pb,nx-pb)
c2 = (pb*c2)**(0.5)

line1, = plt.plot(x,data,'o',label='Data')
line2, = plt.plot(x,y_pred,'k',label='Predicted y')
line3, = plt.plot(x,y_pred + c1*width,'r',label='Pointwise confidence bound')
plt.plot(x,y_pred - c1*width,'r')
line4, = plt.plot(x,y_pred + c2*width,'g',label='Simultaneous confidence bound')
plt.plot(x,y_pred - c2*width,'g')
plt.title('Comparing confidence bounds')
plt.legend(handles=[line1,line2,line3,line4])
plt.show()

# Can compare c1 and c2 for different data sets.
# As nx increases, both shrink towards zero.