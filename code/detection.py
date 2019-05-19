from imageai.Detection import ObjectDetection
import os

# ball_no = 4
# frame_no = 2
# input_image = "./frames/" + ball_no + "/" + frame_no
# output_image = "OutputImage.png"

# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
# detector.loadModel()
# custom_objects = detector.CustomObjects(person=True)


# detections, eachObjectPath = detector.detectCustomObjectsFromImage(custom_objects = custom_objects, input_image=input_image, output_image_path=output_image, minimum_percentage_probability=30, extract_detected_objects=True)

# for eachObject in detections:
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     # print("Object's image saved in " + eachObjectPath)


# i = 1

# directory = output_image+"-objects"

# for filename in os.listdir(directory): 
# 	os.rename(directory+"/"+filename,directory+"/"+str(i)+".png")
# 	i += 1