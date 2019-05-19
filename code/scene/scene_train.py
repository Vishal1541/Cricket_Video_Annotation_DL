#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import os
import matplotlib
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


input_folder = "/users/home/dlagroup5/wd/Project/code/scene/input"
op_folder = "/users/home/dlagroup5/wd/Project/code/scene/results"


# In[7]:

# ORIGIN IMAGE = 720, 1280
# SCALING = 0.2
# FINAL IMAGE = 144, 256

X_train = np.load(input_folder + "/X_train.npy")
print ("Loaded X_train")
y_train = np.load(input_folder + "/y_train.npy")
print ("Loaded y_train")
X_test = np.load(input_folder + "/X_test.npy")
print ("Loaded X_test")
y_test = np.load(input_folder + "/y_test.npy")
print ("Loaded y_test")
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[9]:

# 144, 256
inp = Input(shape=(144,256,3))
cnn = Conv2D(64, kernel_size=5, strides=2, activation="relu")(inp)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=4, strides=2)(cnn)
# cnn = Conv2D(8, kernel_size=1, strides=1, activation="relu")(cnn)
# cnn = BatchNormalization()(cnn)
cnn = Dropout(rate=0.5)(cnn)
x = Flatten()(cnn)

x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(1, activation="sigmoid", name="scene")(x)

model = Model(inputs=inp, outputs=[x])

model.compile(optimizer="adam",
              loss={"scene": "binary_crossentropy"},
              metrics=['accuracy'])

model.summary()


# In[ ]:


history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=1, batch_size=32)
print ("after train")
if not os.path.exists(op_folder):
    os.makedirs(op_folder)
model.save_weights(op_folder + "/model_hackathon.h5")
print ("Model saved...")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig("train_acc.png", bbox_inches='tight')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig("train_loss.png", bbox_inches='tight')
plt.show()

