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


input_path = "../../Output.jpg-objects/person-4.jpg"
op_folder = "/users/home/dlagroup5/wd/Project/code/3class_classifier/bowler_check"
weights_path = "/users/home/dlagroup5/wd/Project/code/3class_classifier/bowler_results/model_bowler.h5"



# In[9]:

# 180, 80
inp = Input(shape=(180,80,3))
cnn = Conv2D(64, kernel_size=3, strides=1, activation="relu")(inp)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=2, strides=1)(cnn)
# cnn = Conv2D(8, kernel_size=1, strides=1, activation="relu")(cnn)
# cnn = BatchNormalization()(cnn)
cnn = Dropout(rate=0.5)(cnn)
x = Flatten()(cnn)

x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(22, activation="softmax", name="bowler")(x)

model = Model(inputs=inp, outputs=[x])

model.compile(optimizer="adam",
              loss={"bowler": "categorical_crossentropy"},
              metrics=['accuracy'])

print("bowler network imported")
# model.summary()

# model.load_weights(weights_path)

# img = cv2.imread(input_path)
# img = np.expand_dims(img, axis=0)
# y_pred = model.predict(img)
# # y_pred = y_pred >= 0.5
# print(y_pred)
