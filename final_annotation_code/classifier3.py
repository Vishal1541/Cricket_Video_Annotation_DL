#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import os
import matplotlib
import cv2
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout, Input
from keras import metrics
from keras.models import model_from_json
from keras.callbacks import Callback
from keras import optimizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# In[10]:


# input_folder = "/users/home/dlagroup5/wd/Project/code/3class_classifier/Players_correct/3_classes"
# input_path = input_folder + "/2/bat90.jpg"
input_path = "../../Output.jpg-objects/person-4.jpg"
op_folder = "/users/home/dlagroup5/wd/Project/code/3class_classifier/3class_check"
weights_path = "/users/home/dlagroup5/wd/Project/code/3class_classifier/3class_results/model_3class.h5"


# In[9]:

# 180, 80
inp = Input(shape=(180,80,3))
cnn = Conv2D(64, kernel_size=5, strides=2, activation="relu")(inp)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=3, strides=1)(cnn)
# cnn = Conv2D(8, kernel_size=1, strides=1, activation="relu")(cnn)
# cnn = BatchNormalization()(cnn)
cnn = Dropout(rate=0.5)(cnn)
x = Flatten()(cnn)

x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(3, activation="softmax", name="3class")(x)

global model

model = Model(inputs=inp, outputs=[x])

model.compile(optimizer="adam",
              loss={"3class": "categorical_crossentropy"},
              metrics=['accuracy'])

# model.summary()
print("3 classifier network (batsman, bowler, other) imported")

# model.load_weights(weights_path)

# img = cv2.imread(input_path)
# img = np.expand_dims(img, axis=0)
# y_pred = model.predict(img)
# y_pred = y_pred >= 0.5
# if(y_pred[0][0] == True):
# 	cv2.imwrite(op_folder + "/batsman.jpg", img[0])
# elif(y_pred[0][1] == True):
# 	cv2.imwrite(op_folder + "/bowler.jpg", img[0])
# else:
# 	cv2.imwrite(op_folder + "/other.jpg", img[0])
# print(y_pred)
