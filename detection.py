from ultralytics import YOLO
import cv2
import torch
import numpy as np
from collections import defaultdict
from pydantic import BaseModel
from detection_keypoint import DetectKeypoint
from load import *

print()

model = YOLO("yolov8m-seg.pt")
pose = YOLO("yolov8m-pose.pt")
gpu_availability = torch.backends.mps.is_available()
track_history = defaultdict(lambda: [])
detection_keypoint = DetectKeypoint()
keypoint_name = ["Nose","Left Eye","Right Eye","Left Ear","Right Ear","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]

def detection(image):
    if gpu_availability:
        print("The GPU is available, Allocating Resources...")
        results = model(image,device="mps")
    else:
        print("The GPU is not available, the object tracking ability is limited and the program slows down")
        results = model(image)
    boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, boundingBoxes):
        if str(cls) == "0":
            (x, y, x2, y2) = bbox
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 225), 2)
    cv2.imshow("Image from Hololens", image)

def track(image):
    if gpu_availability:
        results = model.track(image,device="mps", persist=True)
    else:
        results = model.track(image, persist=True)
    
    boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    if len(boundingBoxes) == 0:
        return
    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
    #trackingids = np.array(results[0].boxes.id.int().cpu().tolist())
    segmentation_contours_idx = []
    for seg in results[0].masks.xy:
        segment = np.array(seg, dtype=np.int32)
        segmentation_contours_idx.append(segment)
    for cls, bbox, seg in zip(classes, boundingBoxes, segmentation_contours_idx):
        if str(cls) == "0":
            (x, y, x2, y2) = bbox
            #track = track_history[trackingid]
            #track.append((float(x+x2)/2,float(y+y2)/2))
            #if len(track) > 30:
            #    track.pop(0)
            #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #cv2.polylines(image, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            cv2.polylines(image, [seg], isClosed=False, color=(0, 0, 255), thickness=10)
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 225), 2)
    cv2.imshow("Tracking image from Hololens", image)
    cv2.waitKey(300)
    input()

def poseEstimation(image, timestamp):
    results = detection_keypoint(image)
    keypoints = detection_keypoint.get_xy_keypoint(results)
    boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    
    for i in range(0,int(len(keypoints)/2)):
        x = int(keypoints[i*2])
        y = int(keypoints[i*2+1])
        color = (255,0,0)
        if i < 5:
            color = (0,255,0)
        elif i < 11:
            color = (0,0,255)
        cv2.circle(image, (x,y), 3, color, -1)
        cv2.imshow("Tracking image from Hololens", image)
        if keypoint_name[i] == "Left Shoulder":
            coordinate_observed = getPointCloudCoordinate(timestamp, x, y)
        
        print(keypoint_name[i] + ": (" + str(x) + "," + str(y) + "), 3D Depth: " + str(getPointCloudCoordinate(timestamp, x, y)) + ", 2D Depth: " + str(getDepth(timestamp, x, y)))
        cv2.waitKey(10) 
    #input()
    return coordinate_observed