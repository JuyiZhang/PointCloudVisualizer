from ultralytics import YOLO
import cv2
import torch
import numpy as np
from collections import defaultdict
from pydantic import BaseModel
from detection_keypoint import DetectKeypoint
from load import *
import random

model = YOLO("yolov8m-seg.pt")
pose = YOLO("yolov8m-pose.pt")

class DetectionManager:
    
    track_history = []
    
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
            results = model.track(image, device="mps", persist=True)
        else:
            results = model.track(image, persist=True)
    
        boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        if len(boundingBoxes) == 0:
            return
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        if (results[0].boxes.id != None):
            tracking_ids = np.array(results[0].boxes.id.cpu().tolist())
        segmentation_contours_idx = []
        for seg in results[0].masks.xy:
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)
        image_overlay = image.copy()
        segment_person = []
        for cls, bbox, seg in zip(classes, boundingBoxes, segmentation_contours_idx):
            if str(cls) == "0":
                segment_person.append(seg)
                (x, y, x2, y2) = bbox
                color = self.generate_random_color()
                # Generate tracking result
                """track = self.track_history[id]
                track.append((float(x+x2)/2,float(y+y2)/2))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                
                cv2.polylines(image, [points], isClosed=False, color=color, thickness=10)"""

                # Draw segmented result
                cv2.polylines(image, [seg], isClosed=False, color=color, thickness=1)
                cv2.fillPoly(image, [seg], color=color)
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x2, y2), color, 2)
                
        cv2.addWeighted(image_overlay, 0.4, image, 0.6, 0)
        return segment_person
        
    def pose_eval(self, image):
        model.track(image, device="mps", persist=True)
    
    def poseEstimation(self, image, timestamp, session=None, observer_coord=np.array([0,0,0])):
        results = self.detection_keypoint(image)
        person_count = self.detection_keypoint.get_person_count(results)
        if (person_count == 0):
            return None, None
        keypoints_list = self.detection_keypoint.get_xy_keypoint(results)
        coordinate_list = []
        img_seg = self.track(image)
        for keypoints in keypoints_list:
            keypoint_location = {}
            keypoint_location_2d = {}
            keypoint_location_validity = {}
            color = self.generate_random_color()
            for i in range(0,int(len(keypoints)/2)):
                x = int(keypoints[i*2])
                y = int(keypoints[i*2+1])
                """color = (255,0,0)
                if i < 5:
                    color = (0,255,0)
                elif i < 11:
                    color = (0,0,255)"""
                cv2.circle(image, (x,y), 3, color, -1)
                threed_coordinate = getPointCloudCoordinate(timestamp, x, y, session)
                validity = self.get_validity(img_seg, threed_coordinate, x, y, observer_coord, self.keypoint_name[i])
                keypoint_location.__setitem__(self.keypoint_name[i], threed_coordinate)
                keypoint_location_2d.__setitem__(self.keypoint_name[i], np.array([x,y]))
                keypoint_location_validity.__setitem__(self.keypoint_name[i], validity)
                if validity:
                    print(self.keypoint_name[i] + ": (" + str(x) + "," + str(y) + "), 3D Depth: " + str(getPointCloudCoordinate(timestamp, x, y, session)) + ", 2D Depth: " + str(getDepth(timestamp, x, y, session)) + ", Validity:" + str(validity))
            #input()
            coordinate_observed = self.get_coordinate(keypoint_location, keypoint_location_validity)
            
            if len(coordinate_observed) != 0:
                body_vector = keypoint_location_2d["Right Shoulder"] - keypoint_location_2d["Left Shoulder"]
                unit_body_vector = body_vector/np.linalg.norm(body_vector)
                x_unit_vector = np.array([1,0])
                dot_x_body = np.dot(x_unit_vector,unit_body_vector)
                orientation_observed = np.arccos(dot_x_body)
                if (body_vector[1] < 0):
                    orientation_observed += 3.14159
                coordinate_observed = np.append(coordinate_observed, orientation_observed)
            coordinate_list.append(coordinate_observed)
        cv2.imshow("Tracking image from Hololens", image)
        return self.remove_duplicate(coordinate_list)
    
    def get_validity(self, image_segment, threed_coordinate, x, y, observer_coord, point_name = ""):
        
        depth_norm = np.linalg.norm(threed_coordinate - observer_coord)
        if depth_norm < 0.3 or (x == 0 and y == 0): #or np.abs(depth_norm - float(depth)/1000) > 0.8:
            return False
        else:
            for seg in image_segment:
                #print("The segmentation data is: ")
                #print(seg)
                #print(cv2.pointPolygonTest(seg, (x,y), True))
                if cv2.pointPolygonTest(seg, (x,y), False) > 0:
                    #print("Point " + point_name + " passed segmentation test!")
                    return True
            #print("Point " + point_name + " failed to pass segmentation test!")
            return False
    
    def get_coordinate(self, coordinate3d_data, validity):
        validity_priority = ["Shoulder","Ear", "Hip","Elbow","Wrist","Knee","Ankle"]
        for priority_str in validity_priority:
            if validity["Left "+priority_str] and validity["Right "+priority_str]:
                return (coordinate3d_data["Left "+priority_str] + coordinate3d_data["Left "+priority_str])/2
            elif validity["Left "+priority_str]:
                return coordinate3d_data["Left "+priority_str]
            elif validity["Right "+priority_str]:
                return coordinate3d_data["Right "+priority_str]
        return np.array([])
            
    def generate_random_color(self):
        a = random.randint(100,255)
        b = random.randint(100,255)
        c = random.randint(100,255)
        return (a, b, c)
    
    def remove_duplicate(self, coordinate_list):
        final_output_list = []
        for coordinate_combine in coordinate_list:
            coord = coordinate_combine[0:3]
            duplicate_found = False
            if len(final_output_list) == 0:
                final_output_list.append(coordinate_combine)
                continue
            if len(coord) == 0:
                continue
            else:
                for out_coordinate_combine in final_output_list:
                    print(coord)
                    out_coord = out_coordinate_combine[0:3]
                    print(out_coord)
                    if (np.linalg.norm(out_coord - coord) < 0.1):
                        print("Duplicate found, not putting into list")
                        duplicate_found = True
            if not(duplicate_found):
                print("Duplicate not found, safe to add to list")
                final_output_list.append(coordinate_combine)
        return final_output_list

    
    




