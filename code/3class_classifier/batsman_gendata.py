#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import os
import numpy as np
from sklearn.utils import shuffle


# In[9]:


folder = "/users/home/dlagroup5/wd/Project/code/3class_classifier/Players_correct/3_classes"
output_folder = "/users/home/dlagroup5/wd/Project/code/3class_classifier/batsman_input"
batsman_folder = folder + "/0"
training_samples = 0.85
X, y = [], []


# In[10]:


def getData(folder, classNo):
    global X, y
    y_len = len(y)
    for frame in os.listdir(folder):
        img = cv2.imread(folder + "/" + frame)
        X.append(img)
        y_len += 1
        y.append([0] * 22)
        y[y_len - 1][classNo] = 1
        


# In[11]:


names = open("playersID_Name.txt", "r").readlines()
for i in range(len(names)):
    name = names[i].split(" ")[1][:-1]
    getData(batsman_folder + "/" + name, i)

    


# In[12]:

print("Converting to np array")
X = np.array(X)
y = np.array(y)
print("Shuffling")
X, y = shuffle(X, y)
total = len(y)
div = int(training_samples * total)
print (total)


# In[13]:

print("Dividing into training and testing data")
X_train = X[:div]
y_train = y[:div]
X_test = X[div:]
y_test = y[div:]

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[14]:


print ("Saving...")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
np.save(output_folder + "/X_train.npy", X_train)
np.save(output_folder + "/y_train.npy", y_train)
np.save(output_folder + "/X_test.npy", X_test)
np.save(output_folder + "/y_test.npy", y_test)
print ("Saved...")


# In[ ]:




