#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reference: https://www.pyimagesearch.com/


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
from keras.layers import concatenate


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


# In[ ]:


def greyscale(arr):
    
    v = []
    for i in range(arr.shape[0]):
    
        gray = cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        v.append(gray.flatten())

    return np.array(v)


# In[ ]:


def build_MLP(X_train_textual):
    
    model = Sequential()
    model.add(Dense(8,input_dim=(X_train_textual.shape[1]), activation='relu'))
    model.add(Dense(4, activation='relu'))
#     model.add(Dense(1, activation='linear'))
    
    return model


# In[ ]:


def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return BatchNormalization()(Activation(activation='relu')(Conv2D(n_filter, (w_filter, h_filter), padding='same')(inputs)))


# In[ ]:


def build_CNN():
    
    model = Sequential()
    inputs = Input(shape=(64, 64, 3))
    
    x = inputs
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

    return cnn


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
def process_house_attributes(df, train, test):
    
    continuous = ["num_bedrooms", "num_bathrooms", "area"]

    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])
 

    zipBinarizer = LabelBinarizer().fit(df["zip_code"])
    trainCategorical = zipBinarizer.transform(train["zip_code"])
    testCategorical = zipBinarizer.transform(test["zip_code"])
 

    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])
 
    return (trainX, testX)


# In[ ]:


cols = ["num_bedrooms", "num_bathrooms", "area", "zip_code", "price"]
dataset = pd.read_csv("HousesInfo.txt", sep=" ", header=None, names=cols)


# In[ ]:


dframe = remove_outliers(dataset)


# In[ ]:


images = read_images(dframe)
images = images/np.max(images)


# In[ ]:


X_test_textual


# In[ ]:


X_train_textual, X_test_textual, X_train_images, X_test_images = train_test_split(dframe, images, test_size=0.25, random_state=42)


# In[ ]:


X_train_textual, X_test_textual = process_house_attributes(dframe, X_train_textual, X_test_textual)


# In[ ]:


X_train_textual


# In[ ]:


maximum = np.max(X_train_textual[:, 9])
y_train = X_train_textual[:, 9]/maximum
y_test = X_test_textual[:, 9]/maximum


# In[ ]:


cnn_network = build_CNN()


# In[ ]:


opt = Adam(lr=1e-3, decay=1e-3 / 200)
cnn_network.compile(loss="mean_absolute_percentage_error", optimizer=opt)
cnn_network.summary()


# In[ ]:


mlp = build_MLP(X_train_textual)


# In[ ]:


combined = concatenate([mlp.output, cnn_network.output])
y = Dense(4, activation="relu")(combined)
y = Dense(1, activation="linear")(y)
model = Model(inputs=[mlp.input, cnn_network.input], outputs=y)


# In[ ]:


opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


# In[ ]:


model.fit([X_train_textual, X_train_images], y_train,validation_split=0.2, epochs=25, batch_size=8)


# In[ ]:


preds = model.predict([X_test_textual, X_test_images])


# In[ ]:


r2_cnn = r2_score(y_test, preds)
mae_cnn = mean_absolute_error(y_test, preds)
print("MAE for CNN+MLP", mae_cnn)
print("R2 for CNN+MLP", r2_cnn)


# In[ ]:


dframe


# In[ ]:


X_train_textual


# In[ ]:




