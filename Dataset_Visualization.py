#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
df1=pd.read_csv('C://Users//home//Desktop//IIITD//SML//Project//Houses-dataset-master//Houses Dataset//HousesInfo.txt',header=None, sep=" ",names=columns)


# # Feature Information

# In[ ]:


#Statistical details of house dataset
df1[['price','area','bedrooms','bathrooms']].describe().take([1,3, 7])


# # Dataset Visualization

# In[ ]:


#Plot of Bedroom, Bathroom, Kitchen and Frontal Side of Sample House
plt.subplot(2,2,1)
img1 = plt.imread('C://Users//home//Desktop//IIITD//SML//Project//Houses-dataset-master//Houses Dataset//28_bedroom.jpg')
plt.imshow(img1)
plt.subplot(2,2,2)
img2 = plt.imread('C://Users//home//Desktop//IIITD//SML//Project//Houses-dataset-master//Houses Dataset//28_bathroom.jpg')
plt.imshow(img2)
plt.subplot(2,2,3)
img3 = plt.imread('C://Users//home//Desktop//IIITD//SML//Project//Houses-dataset-master//Houses Dataset//28_kitchen.jpg')
plt.imshow(img3)
plt.subplot(2,2,4)
img4 = plt.imread('C://Users//home//Desktop//IIITD//SML//Project//Houses-dataset-master//Houses Dataset//28_frontal.jpg')
plt.imshow(img4)
plt.show()


# # Count of Houses According to Zip-code

# In[ ]:


#Categorical Dataset Visualization
fig, ax = plt.subplots(figsize=(8,8))
X = (np.array(df1["zipcode"].value_counts().keys())).astype(str)
Y = np.array(df1["zipcode"].value_counts())
ax.barh(X, Y)
ax.set_title('Categorical Dataset Visualization', fontsize=12)
ax.set_xlabel('Count of Houses', fontsize=12)
ax.set_ylabel('Zip-code', fontsize=12)


# # Correlation Matrix Plot

# In[ ]:


# Correction Matrix Plot
names = list(df1.columns)
correlations = df1.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[ ]:




