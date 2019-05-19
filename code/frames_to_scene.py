#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance


# In[2]:


groundtruth = "/users/home/dlagroup5/wd/Project/code/scene/scene0.txt"
ball_folder = "/users/home/dlagroup5/wd/Project/code/frames"
scene0 = "/users/home/dlagroup5/wd/Project/code/scene/0"
scene1 = "/users/home/dlagroup5/wd/Project/code/scene/1"
count_0 = len([name for name in os.listdir(scene0)])
count_1 = len([name for name in os.listdir(scene1)])


# In[3]:


lines = np.loadtxt(groundtruth)
lines = lines.astype(int)
for i in range(len(lines)):
    folder = ball_folder + "/" + str(lines[i][0])
    print (i)
    for images in os.listdir(folder):
        sno = int(images.split(".")[0])
        img = Image.open(folder + "/" + images)
        if(lines[i][1] <= sno and sno <= lines[i][2]):
            count_0 += 1
            img1 = ImageEnhance.Contrast(img)
            img2 = img1.enhance(1.5)
            img3 = img1.enhance(0.8)
            img4 = img1.enhance(1.25)
            img.save(scene0 + "/" + str(count_0) + ".jpg")
            count_0 += 1
            img2.save(scene0 + "/" + str(count_0) + ".jpg")
            count_0 += 1
            img3.save(scene0 + "/" + str(count_0) + ".jpg")
            count_0 += 1
            img4.save(scene0 + "/" + str(count_0) + ".jpg")
        else:
            count_1 += 1
            img.save(scene1 + "/" + str(count_1) + ".jpg")


# In[29]:




