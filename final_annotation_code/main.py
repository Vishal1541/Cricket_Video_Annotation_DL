import numpy as np
import cv2
import scene
import classifier3
import batsman
import bowler
from imageai.Detection import ObjectDetection
import os
import shutil

scene_weights_path = "/users/home/dlagroup5/wd/Project/code/scene/results/model_hackathon.h5"
class3_weights_path = "/users/home/dlagroup5/wd/Project/code/3class_classifier/3class_results/model_3class.h5"
batsman_weights_path = "/users/home/dlagroup5/wd/Project/code/3class_classifier/batsman_results/model_batsman.h5"
bowler_weights_path = "/users/home/dlagroup5/wd/Project/code/3class_classifier/bowler_results/model_bowler.h5"
INPUT_VIDEO = "input/video.mp4"
FRAME_RATE = 0.03
output_image = "Output.jpg"
names = open("../code/playersID.txt").readlines()

def init_detection():
	global detector, custom_objects
	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath("../resnet50_coco_best_v2.0.1.h5")
	detector.loadModel()
	custom_objects = detector.CustomObjects(person=True)

def load_weights():
	print("loading scene weights...")
	scene.model.load_weights(scene_weights_path)
	print("scene weights loaded...")

	print("loading 3 class weights...")
	classifier3.model.load_weights(class3_weights_path)
	print("3 class weights loaded...")

	print("loading batsman weights...")
	batsman.model.load_weights(batsman_weights_path)
	print("batsman weights loaded...")

	print("loading bowler weights...")
	bowler.model.load_weights(bowler_weights_path)
	print("bowler weights loaded...")

def getFrameUtil(sec, vidcap, frame_no):
	vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,image = vidcap.read()
	image_small = 0
	if(hasFrames):
		image_small = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
	return hasFrames, image, image_small

def getFrame(vidcap):
	success = True
	frame_no, sec = 0, -FRAME_RATE
	frames, frames_small = [], []
	while success:
		sec = sec + FRAME_RATE
		sec = round(sec, 2)
		success, img, img_small = getFrameUtil(sec, vidcap, frame_no)
		frame_no += 1
		if(success == True):
			frames.append(img)
			frames_small.append(img_small)
	frames = np.array(frames)
	frames_small = np.array(frames_small)
	return frames, frames_small

def convert_frames_to_video(frame_array, pathOut,fps, size):

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

init_detection()
load_weights()
vidcap = cv2.VideoCapture(INPUT_VIDEO)
frames, frames_small = getFrame(vidcap)
# print(frames_small.shape)
TOTAL_FRAMES = len(frames_small)
print("video loaded...")
print("Total frames = ", TOTAL_FRAMES)

print(frames_small.shape)
y_pred = scene.model.predict(frames_small)
y_pred = y_pred <= 0.5		#	Scene 0 = true
# print(y_pred)

shutil.rmtree("frames")
os.mkdir("frames")
shutil.rmtree("frames2")
os.mkdir("frames2")
shutil.rmtree("output")
os.mkdir("output")

scene0_total = 0
for i in range(TOTAL_FRAMES):
	if(y_pred[i][0] == True):
		scene0_total += 1
		cv2.imwrite("./frames/" + str(i) + ".jpg", frames[i])

print("Total scene0 = ", scene0_total)
frame_array = []
scene_no = 0
H, W, L = frames[0].shape
size = (W, H)
for i in range(TOTAL_FRAMES):
	if(y_pred[i][0] == True):
		scene_no += 1
		print("\t\t\t\t SCENE NO: ", scene_no, " / " , scene0_total)
		detections, eachObjectPath = detector.detectCustomObjectsFromImage(custom_objects = custom_objects, input_image="./frames/" + str(i) + ".jpg", minimum_percentage_probability=30, extract_detected_objects=True)
		if(detections == "hello"): continue
		print(detections)
		batsman_idx, bowler_idx = -1, -1
		obj_arr = []
		for j in range(len(detections)):
			obj = cv2.imread("-objects/person-" + str(j + 1) + ".jpg")
			obj = np.expand_dims(obj, axis=0)
			obj_arr.append(obj)
			y_3class = classifier3.model.predict(obj)
			y_3class = np.argmax(y_3class)
			if(y_3class == 0): batsman_idx = j
			elif(y_3class == 1): bowler_idx = j
			print (y_3class)
		print("BatsIDX: ", batsman_idx, type(batsman_idx))
		print("BowlIDX: ", bowler_idx, type(bowler_idx))
		if(batsman_idx != -1):
			batsman_img = obj_arr[batsman_idx]
			y_batsman = batsman.model.predict(batsman_img)
			y_batsman = np.argmax(y_batsman)
			P = detections[batsman_idx]["box_points"]
			img = cv2.rectangle(frames[i], (P[0], P[1]), (P[2], P[3]), (0,0,255), 2)
			img = cv2.putText(img, "Batsman: " + names[y_batsman].split(" ")[1][:-1], (P[0], P[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA) 
			print("Batsman = ", names[y_batsman])
		if(bowler_idx != -1):
			bowler_img = obj_arr[bowler_idx]
			y_bowler = bowler.model.predict(bowler_img)
			y_bowler = np.argmax(y_bowler)
			P = detections[bowler_idx]["box_points"]
			img = cv2.rectangle(frames[i], (P[0], P[1]), (P[2], P[3]), (0,0,255), 2)
			img = cv2.putText(img, "Bowler: " + names[y_bowler].split(" ")[1][:-1], (P[0], P[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA) 
			print("Bowler = ", names[y_bowler])
		# cv2.imwrite("frames2/" + str(i) + ".jpg", img)
		img = cv2.putText(img, "Camera View 1: Front View", (W - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), lineType=cv2.LINE_AA) 
		frame_array.append(img)
	else:
		img = cv2.putText(frames[i], "Camera View 2: Other View", (W - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), lineType=cv2.LINE_AA) 
		frame_array.append(img)

convert_frames_to_video(frame_array, "output/video.avi", 30, size)
