import shutil
import numpy as np
import os
import cv2
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

input_folder = "/users/home/dlagroup5/wd/Project/code/frames"
weights_path = "/users/home/dlagroup5/wd/Project/code/scene/results/model_hackathon.h5"
check_folder = "/users/home/dlagroup5/wd/Project/code/scene/check"
frame_no = 70
path = input_folder + "/" + str(frame_no)
zero_class_path = check_folder + "/0"
one_class_path = check_folder + "/1"

# MODEL

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
model.load_weights(weights_path)

count_0, count_1 = 0, 0
shutil.rmtree(zero_class_path)
shutil.rmtree(one_class_path)
os.makedirs(zero_class_path)
os.makedirs(one_class_path)

for frame in os.listdir(path):
	img_ori = cv2.imread(path + "/" + frame)
	img = cv2.resize(img_ori, (0,0), fx=0.2, fy=0.2)
	img = np.expand_dims(img, axis=0)
	y_pred = model.predict(img)
	y_pred = y_pred <= 0.5
	print(y_pred,)
	if(y_pred == True):
		cv2.imwrite(zero_class_path + "/" + str(count_0) + ".jpg", img_ori)
		count_0 += 1
	else:
		cv2.imwrite(one_class_path + "/" + str(count_1) + ".jpg", img_ori)
		count_1 += 1
print("Done")