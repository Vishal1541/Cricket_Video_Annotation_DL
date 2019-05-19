#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


folder = "/users/home/dlagroup5/wd/Project/code/frames"
FROM_BALL = 126
TO_BALL = 247
FRAME_RATE = 0.3


# In[3]:


def getFrameUtil(sec, vidcap, frame_no, ball):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(folder + "/" + str(ball) + "/" + str(frame_no) + ".jpg", image)     # save frame as JPG file
    return hasFrames


# In[4]:


def getFrame(vidcap, ball):
    success = True
    frame_no, sec = 0, -FRAME_RATE
    while success:
        sec = sec + FRAME_RATE
        sec = round(sec, 2)
        success = getFrameUtil(sec, vidcap, frame_no, ball)
        frame_no += 1
#         print sec,


# In[5]:


for ball in range(FROM_BALL, TO_BALL + 1):
    vidcap = cv2.VideoCapture(folder + "/" + str(ball) + ".mp4")
    getFrame(vidcap, ball)
    print("Ball ",ball," done")


# In[ ]:




