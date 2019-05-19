#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import numpy as np
from sklearn.utils import shuffle


# In[9]:


folder = "/users/home/dlagroup5/wd/Project/code/scene"
folder_class0 = folder + "/0"
folder_class1 = folder + "/1"
training_samples = 0.85
output_folder = folder + "/input"


# In[6]:


def getData(lower, upper):
    X, y = [], []
    for i in range(2):
        class_folder = folder + "/" + str(i)
        total = len([name for name in os.listdir(class_folder)])
        L, U = int(lower * total), int(upper * total)
        for j in range(L, U):
            if(j % 100 == 0):
                print (j)
            img = cv2.imread(class_folder + "/" + str(j + 1) + ".jpg")
            img2 = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
            # print(img2.shape)
            # (144,256,3)
            X.append(img2)
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y
            


# In[4]:


X, y = getData(0, 1)
print("Shuffling...")
X, y = shuffle(X, y)
total = len(y)
div = int(total * training_samples)
print("Dividing into training and testing...")
X_train, y_train = X[:div], y[:div]
X_test, y_test = X[div:], y[div:]


# In[ ]:


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[ ]:


print ("Saving...")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
np.save(output_folder + "/X_train.npy", X_train)
np.save(output_folder + "/y_train.npy", y_train)
np.save(output_folder + "/X_test.npy", X_test)
np.save(output_folder + "/y_test.npy", y_test)
print ("Saved...")

