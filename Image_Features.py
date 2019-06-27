#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np
import argparse
import locale
import os
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt


# In[ ]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# In[ ]:


image_shape = (32, 32)


# # Removing Outliers

# In[ ]:


def remove_outliers(dfs):
    
    df = dfs.copy(deep=True)
    zipcodes = list(df['zip_code'].value_counts().keys())
    counts = list(df["zip_code"].value_counts())

    df_new = pd.DataFrame()
    for (zipcode, count) in zip(zipcodes, counts):
        if count > 25:
            ind = df[df['zip_code'] == zipcode].index
            for i in ind:
                df_new = df_new.append(df[i:i+1])
            
    return df_new


# # Reading and Make Collage of Images

# In[ ]:


def read_images(df):
    
    
    path = os.getcwd()
    path = os.path.join(path, "Houses Dataset")
    all_images = []
    for i in df.index.values:
        
        imagepath = os.path.sep.join([path, "{}_*".format(i + 1)])
        four_images = glob.glob(imagepath)
        four_images = sorted(four_images)
        
        temp = []
        collage = np.zeros((64, 64, 3), dtype=np.uint8)
        
        for image in four_images:
            
            img = cv2.imread(image)
            img = cv2.resize(img, image_shape)
            temp.append(img)
            
        collage[0:32, 0:32] = temp[0]
        collage[0:32, 32:64] = temp[1]
        collage[32:64, 32:64] = temp[2]
        collage[32:64, 0:32] = temp[3]
        
        all_images.append(collage)
        
    all_images = np.array(all_images)
    return all_images


# # Convert to Grey-scale

# In[ ]:


def greyscale(arr):
    
    v = []
    for i in range(arr.shape[0]):
    
        gray = cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        v.append(gray.flatten())

    return np.array(v)


# # Support Vector Regression

# In[ ]:


def support_regressor(X_train, y_train, X_test, y_test):
    
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    pred_svr = svr.predict(X_test)
    mae_svr = mean_absolute_error(y_test, pred_svr)
    r2_svr = r2_score(y_test, pred_svr)
    
    return r2_svr, mae_svr


# # XGBoost Regression

# In[ ]:


def XGBoost_regressor(X_train, y_train, X_test, y_test):
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)
    r2_xgb = r2_score(y_test, pred_xgb)
    mae_xgb = mean_absolute_error(y_test, pred_xgb)
    
    return r2_xgb, mae_xgb


# # Linear Regression

# In[ ]:


def linear_regression(X_train, y_train, X_test, y_test):
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, pred_lr)
    mse_lr = mean_absolute_error(y_test, pred_lr)
    
    return r2_lr, mse_lr


# # Kernel Ridge Regression

# In[ ]:


def kernel_ridge_regression(X_train, y_train, X_test, y_test):
    
    clf = KernelRidge(alpha=1.0)
    clf.fit(X_train, y_train)
    pred_krr = clf.predict(X_test)
    r2_krr = r2_score(y_test, pred_krr)
    mse_krr = mean_absolute_error(y_test, pred_krr)
    
    return r2_krr, mse_krr


# # Random Forest Regressor

# In[ ]:


def random_forest_regressor(X_train, y_train, X_test, y_test):
    
    rfr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    rfr.fit(X_train, y_train)
    pred_rfr = rfr.predict(X_test)
    r2_rfr = r2_score(y_test, pred_rfr)
    mae_rfr = mean_absolute_error(y_test, pred_rfr)
    
    return r2_rfr, mae_rfr


# # Convolutional Neural Network

# In[ ]:


def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return BatchNormalization()(Activation(activation='relu')(Conv2D(n_filter, (w_filter, h_filter), padding='same')(inputs)))


# In[ ]:


def build_CNN():
    
    model = Sequential()
    inputs = Input(shape=(64, 64, 3))
    
    x = inputs
    
#     x = Conv2D(16, (3, 3), padding="same", activation='relu', BatchNormalization=-1)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)

    model = Conv2DReluBatchNorm(16, 3, 3, x)
    model = Conv2DReluBatchNorm(32, 3, 3, model)
    model = Conv2DReluBatchNorm(64, 3, 3, model)
    
    model = Flatten()(model)
    model = Dense(16, activation='relu')(model)
    model = BatchNormalization(axis=-1)(model)
    model = Dropout(0.5)(model)
    
    model = Dense(4, activation='relu')(model)
    model = Dense(1, activation='linear')(model)
    
    cnn = Model(inputs, model)

    # return the CNN
    return cnn
    
#     inputs = model.add(Conv2D(16, 3, 3, input_shape=(64, 64, 3), activation='relu', BatchNormalization=-1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model


# # Read Dataset

# In[ ]:


cols = ["num_bedrooms", "num_bathrooms", "area", "zip_code", "price"]
dataset = pd.read_csv("HousesInfo.txt", sep=" ", header=None, names=cols)


# In[ ]:


dframe = remove_outliers(dataset)


# In[ ]:


images = read_images(dframe)
images = images/np.max(images)


# # Train-Test Split

# In[ ]:


X_train_textual, X_test_textual, X_train_images, X_test_images = train_test_split(dframe, images, test_size=0.25, random_state=42)


# In[ ]:


maximum = np.max(X_train_textual["price"])
y_train = X_train_textual["price"]/maximum
y_test = X_test_textual["price"]/maximum


# # Convolutional Neural Network

# In[ ]:


cnn_network = build_CNN()


# In[ ]:


opt = Adam(lr=1e-3, decay=1e-3 / 200)
cnn_network.compile(loss="mean_absolute_percentage_error", optimizer=opt)
cnn_network.summary()


# In[ ]:


cnn_network.fit(X_train_images, y_train, validation_data=(X_test_images, y_test), epochs=100, batch_size=8)


# In[ ]:


preds = cnn_network.predict(X_test_images)


# In[ ]:


r2_cnn = r2_score(y_test, preds)
mae_cnn = mean_absolute_error(y_test, preds)
print("MAE for NN", mae_cnn)
print("R2 for NN", r2_cnn)


# In[ ]:


trainX = greyscale(X_train_images)
testX = greyscale(X_test_images)


# # Regression Models

# In[ ]:


r2_svr, mae_svr = support_regressor(trainX, y_train, testX, y_test)


# In[ ]:


# r2_xgb, mae_xgb = XGBoost_regressor(trainX, y_train, testX, y_test)


# In[ ]:


r2_lr, mae_lr = linear_regression(trainX, y_train, testX, y_test)


# In[ ]:


# r2_krr, mae_krr = kernel_ridge_regression(trainX, y_train, testX, y_test)


# In[ ]:


r2_rfr, mae_rfr = random_forest_regressor(trainX, y_train, testX, y_test)


# In[ ]:


print("R2 for RFR", r2_rfr)
print("MAE for RFR", mae_rfr)


# In[ ]:


print("R2 for SVR", r2_svr)
print("MAE for SVR", mse_svr)


# In[ ]:


print("R2 for LR", r2_lr)
print("MAE for LR", mae_lr)


# In[ ]:


# print("R2 for XGboost", r2_xgb)
# print("MAE for XGboost", mae_xgb)


# In[ ]:


print("R2 for KRR", r2_krr)
print("MAE for KRR", mae_krr)


# # Residual Plot

# In[ ]:


# Reference: https://media.readthedocs.org/pdf/yellowbrick/stable/yellowbrick.pdf
from yellowbrick.regressor import ResidualsPlot

# Instantiate the linear model and visualizer
ridge = KernelRidge(alpha=1.0)
visualizer = ResidualsPlot(ridge)

visualizer.fit(trainX, y_train)  # Fit the training data to the model
visualizer.score(testX, y_test)  # Evaluate the model on the test data
visualizer.poof()

