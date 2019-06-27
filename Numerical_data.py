#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


##https://www.pyimagesearch.com/tag/regression/


# In[ ]:


import pandas as pd
import numpy as np
import math
import statistics
from statistics import mean 
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import locale
from keras.models import Sequential, Model
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, Input
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# In[ ]:


columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
df1=pd.read_csv('C://Users//home//Desktop//IIITD//SML//Project//Houses-dataset-master//Houses Dataset//HousesInfo.txt',header=None, sep=" ",names=columns)


# # One-Hot Encoding

# In[ ]:


def binarizer(df):
    zb=LabelBinarizer().fit(df["zipcode"])
    df2 = pd.DataFrame(zb.transform(df["zipcode"]))
    df3=df.drop(['zipcode'], axis=1)
    df4 = pd.concat([df3, df2], axis=1,sort=False)
    return df4


# # Train-Test Split

# In[ ]:


def split(df):
#     train = df.sample(frac = 0.75)
#     test = df.drop(train.index)
    (train, test) = train_test_split(df, test_size=0.25, random_state=42)
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
    
    return train,test


# # Min-Max Normalization

# In[ ]:


def minmaxNorm(train,test):
    sc = MinMaxScaler()
    train1=pd.DataFrame(sc.fit_transform(train[['bedrooms','bathrooms','area']]))
    train1.columns=['bedrooms','bathrooms','area']
    traindropped=train.drop(['bedrooms','bathrooms','area'],axis=1)
    trainfinal=pd.concat([train1,traindropped],axis=1,sort=False)
    test1=pd.DataFrame(sc.transform(test[['bedrooms','bathrooms','area']]))
    test1.columns=['bedrooms','bathrooms','area']
    testdropped=test.drop(['bedrooms','bathrooms','area'],axis=1)
    testfinal=pd.concat([test1,testdropped],axis=1,sort=False)
    return trainfinal,testfinal


# In[ ]:


def reduce_feat(dfs):
    df = dfs.copy(deep=True)
    zipcodes = list(df['zipcode'].value_counts().keys())
   
    counts = df["zipcode"].value_counts().tolist()
    for (zipcode, count) in zip(zipcodes, counts):

        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    return df


# # K-Fold Cross Validation

# In[ ]:


def kfold(data,datatest, flag):
    tdata_collection = ["" for i in range(5)]
    acc = []
    pp = math.ceil((len(data)/5))
    xs = 0
    for i in range(5):
        test = data[xs:pp+xs]
        train = data.drop(test.index)
        xs = xs + pp
        train.index = range(len(train))
        test.index = range(len(test))
        tdata_collection[i] = train
        if flag==1:
            a = NNregressor(train,test)
        elif flag==2:
            a = supportVectorR(train, test)
        elif flag==3:
            a=RFR(train, test)            
        acc.append(a)
    ind = acc.index(min(acc))
    if flag==1:
        test_acc = NNregressor(tdata_collection[ind],datatest)
    elif flag==2:
        test_acc = supportVectorR(tdata_collection[ind],datatest)
    elif flag==3:
        test_acc = RFR(tdata_collection[ind], datatest)
    return acc


# In[ ]:


df2 = reduce_feat(df1)
df2 = df2.reset_index(drop=True)
print(len(df2))
df = binarizer(df2)
train, test = split(df)
train, test = minmaxNorm(train,test)


# # Neural network

# In[ ]:


from sklearn.neural_network import MLPRegressor
def NNregressor(train, test):
    Ytrain=train['price']/train['price'].max()
    Xtrain=pd.DataFrame(train.drop(['price'],axis=1))
    Ytest=test['price']/train['price'].max()
    Xtest=test.drop(['price'],axis=1)
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    m = mean_absolute_error(Ytest, pred)
#     s = r2_score(Ytest, pred)
#     print("R2_score", s)
    print("mean_absolute_error", m)
    return m


# In[ ]:


accNN = kfold(train,test, 1)


# In[ ]:


print(accNN)


# # Regression Models

# In[ ]:


from sklearn.svm import SVR
def supportVectorR(train,test):
    Ytrain=train['price']/train['price'].max()
    Xtrain=pd.DataFrame(train.drop(['price'],axis=1))
    Ytest=test['price']/train['price'].max()
    Xtest=test.drop(['price'],axis=1)
    clf = SVR(gamma=0.001, C=1.0, epsilon=0.2, kernel='rbf')
    clf.fit(Xtrain.values, Ytrain.values) 
    pred=clf.predict(Xtest)
#     s = r2_score(Ytest, pred)
    m = mean_absolute_error(Ytest, pred)
#     print("R2_score", s)
    print("mean_absolute_error", m)
    return m


# In[ ]:


# pred=supportVectorR(train,test)
accSVR = kfold(train, test, 2)


# In[ ]:


print(accSVR)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
def RFR(train,test):
    Ytrain=train['price']/train['price'].max()
    Xtrain=pd.DataFrame(train.drop(['price'],axis=1))
    Ytest=test['price']/test['price'].max()
    Xtest=test.drop(['price'],axis=1)
    regressor = RandomForestRegressor(random_state=0,n_estimators=10)
    regressor.fit(Xtrain.values, Ytrain.values)
    pred=regressor.predict(Xtest)
    s = r2_score(Ytest, pred)
    m = mean_absolute_error(Ytest, pred)
#     print("R2_score", s)
    print("mean_absolute_error",m)
    return m


# In[ ]:


accRF = kfold(train, test, 3)


# # Plots

# In[ ]:


x_list=[1,2,3,4,5]
plt.xlabel("Number of folds")
plt.ylabel("Mean Absolute Error")
plt.plot(x_list,accNN,label="Neural Network")
plt.plot(x_list,accSVR, label="Support Vector Regression")
plt.plot(x_list,accRF, label="Random Forest")
plt.legend()
plt.show()


# # Residual Plot

# In[ ]:


# Reference: https://media.readthedocs.org/pdf/yellowbrick/stable/yellowbrick.pdf
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
ridge = LinearRegression()
visualizer = ResidualsPlot(ridge)
Ytrain=train['price']/train['price'].max()
Xtrain=pd.DataFrame(train.drop(['price'],axis=1))
Ytest=test['price']/test['price'].max()
Xtest=test.drop(['price'],axis=1)
visualizer.fit(Xtrain, Ytrain)  # Fit the training data to the model
visualizer.score(Xtest, Ytest)  # Evaluate the model on the test data
visualizer.poof()


# In[ ]:





# In[ ]:




