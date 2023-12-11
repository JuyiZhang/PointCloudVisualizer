from ultralytics import YOLO
import cv2
import torch
import numpy as np
from collections import defaultdict
from pydantic import BaseModel
from detection_keypoint import DetectKeypoint
from load import *

model = YOLO("yolov8m-seg.pt")
pose = YOLO("yolov8m-pose.pt")

class DetectionManager:
    def __init__(self) -> None:
        self.keypoint_name = ["Nose","Left Eye","Right Eye","Left Ear","Right Ear","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
        self.gpu_availability = torch.backends.mps.is_available()
        if self.gpu_availability:
            print("The M-Series GPU is available, Allocating Resources...")
        else:
            print("The M-Series GPU is not available, falling back to CPU/Using default torch device")
        self.track_history = defaultdict(lambda: [])
        self.detection_keypoint = DetectKeypoint()
        
    def detection(self, image):
        if self.gpu_availability:
            results = model(image,device="mps")
        else:
            results = model(image)
        boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        for cls, bbox in zip(classes, boundingBoxes):
            if str(cls) == "0":
                (x, y, x2, y2) = bbox
                cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.imshow("Image from Hololens", image)
    
    def track(self, image):
        if self.gpu_availability:
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
    
    def pose_eval(self, image):
        model.track(image, device="mps", persist=True)
    
    def poseEstimation(self, image, timestamp, session=None, observer_coord=np.array([0,0,0])):
        results = self.detection_keypoint(image)
        keypoints = self.detection_keypoint.get_xy_keypoint(results)
        boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        keypoint_location = {}
        keypoint_location_2d = {}
        keypoint_location_validity = {}
        for i in range(0,int(len(keypoints)/2)):
            x = int(keypoints[i*2])
            y = int(keypoints[i*2+1])
            color = (255,0,0)
            if i < 5:
                color = (0,255,0)
            elif i < 11:
                color = (0,0,255)
            cv2.circle(image, (x,y), 3, color, -1)
            threed_coordinate = getPointCloudCoordinate(timestamp, x, y, session)
            validity = self.get_validity(threed_coordinate,x,y,getDepth(timestamp, x, y, session), observer_coord)
            keypoint_location.__setitem__(self.keypoint_name[i], threed_coordinate)
            keypoint_location_2d.__setitem__(self.keypoint_name[i], np.array([x,y]))
            keypoint_location_validity.__setitem__(self.keypoint_name[i], validity)
            if validity:
                print(self.keypoint_name[i] + ": (" + str(x) + "," + str(y) + "), 3D Depth: " + str(getPointCloudCoordinate(timestamp, x, y, session)) + ", 2D Depth: " + str(getDepth(timestamp, x, y, session)) + ", Validity:" + str(validity))
        #input()
        body_vector = keypoint_location_2d["Left Shoulder"] - keypoint_location_2d["Right Shoulder"]
        unit_body_vector = body_vector/np.linalg.norm(body_vector)
        x_unit_vector = np.array([1,0])
        dot_x_body = np.dot(x_unit_vector,unit_body_vector)
        orientation_observed = np.arccos(dot_x_body)
        if keypoint_location_validity["Left Shoulder"] and keypoint_location_validity["Right Shoulder"]:
            if np.linalg.norm(keypoint_location["Left Shoulder"] - keypoint_location["Right Shoulder"]) < 0.6:
                coordinate_observed = (keypoint_location["Left Shoulder"] + keypoint_location["Right Shoulder"])/2
            else:
                coordinate_observed = keypoint_location["Right Shoulder"]
        elif keypoint_location_validity["Left Shoulder"]:
            coordinate_observed = keypoint_location["Left Shoulder"]
        elif keypoint_location_validity["Right Shoulder"]:
            coordinate_observed = keypoint_location["Right Shoulder"]
        cv2.imshow("Tracking image from Hololens", image)
        return coordinate_observed, orientation_observed
    
    def get_validity(self, threed_coordinate, x, y, depth, observer_coord):
        depth_norm = np.linalg.norm(threed_coordinate - observer_coord)
        if depth_norm < 0.3 or (x == 0 and y == 0): #or np.abs(depth_norm - float(depth)/1000) > 0.8:
            return False
        else:
            return True
    
    def get_coordinate(self, coordinate3d_data, validity):
        return 0
    
    




