#!/usr/bin/env python
# coding: utf-8

# In[1]:


from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os


# In[2]:


# folder where each ball video and frames are saved
folder = "/users/home/dlagroup5/wd/Project/code/frames"
VIDEO_PATH = "/users/home/dlagroup5/wd/Project/videos/t1.mp4"

if not os.path.exists(folder):
    os.makedirs(folder)


# In[3]:


ballCounts = 0
for filename in os.listdir(folder):
    ballCounts += 1
ballCounts /= 2
print("Total balls folder: ",ballCounts)
ballCounts += 1

trimVideo = open("trim.txt", "r")
lines = trimVideo.read().split('\n')
# ignoring the last \n line (if any)
L = len(lines)
if(len(lines[L - 1]) == 0):
    lines.remove("")
    L -= 1
print (lines)



# In[4]:


def trim_and_save_video(line, ballCounts):
    h1, m1, s1, h2, m2, s2 = map(int, line.split(" "))
#     Converting Min:Sec to Sec
#     taking margin of 0.75 sec before starting time and after ending time
    start_time = h1 * 60 * 60 + m1 * 60 + s1 - 0.75
    end_time = h2 * 60 * 60 + m2 * 60 + s2 + 0.75
    os.makedirs(folder + "/" + ballCounts)
    ffmpeg_extract_subclip(VIDEO_PATH, start_time, end_time, targetname=folder + "/" + ballCounts + ".mp4")
    


# In[5]:


for i in range(L):
    trim_and_save_video(lines[i], str(ballCounts))
    ballCounts += 1
    print("Trimmed ", i)

