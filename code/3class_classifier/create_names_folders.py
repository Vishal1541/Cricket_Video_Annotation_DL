#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import numpy as np
import shutil


# In[24]:


op_folder = "./Players_correct"
batsman_folder = op_folder + "/0"
bowler_folder = op_folder + "/1"
# shutil.rmtree(batsman_folder)
# shutil.rmtree(bowler_folder)
if not os.path.exists(batsman_folder):
    os.mkdir(batsman_folder)
if not os.path.exists(bowler_folder):
    os.mkdir(bowler_folder)


# In[26]:


names = open("playersID_Name.txt", "r").readlines()
for i in range(len(names)):
    os.mkdir(batsman_folder + "/" + names[i].split(" ")[1])
    os.mkdir(bowler_folder + "/" + names[i].split(" ")[1])


# In[ ]:




