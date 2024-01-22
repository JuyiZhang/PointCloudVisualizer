from ultralytics import YOLO
import cv2
import torch
import numpy as np
from collections import defaultdict
from DetectionKeypoint import DetectKeypoint
from FrameManager import Frame
import random
import Debug
from multiprocessing import Queue
import Utility


model = YOLO("yolov8m-seg.pt")
pose = YOLO("yolov8m-pose.pt")

track_history = []
    
keypoint_name = ["Nose","Left Eye","Right Eye","Left Ear","Right Ear","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
gpu_availability = torch.backends.mps.is_available()
if gpu_availability:
    Debug.Log("The M-Series GPU is available, Allocating Resources...")
else:
    Debug.Log("The M-Series GPU is not available, falling back to CPU/Using default torch device")
track_history = defaultdict(lambda: [])
detection_keypoint = DetectKeypoint()
    
def track_person(image):
    if gpu_availability:
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
    segment_person = []
    for cls, bbox, seg in zip(classes, boundingBoxes, segmentation_contours_idx):
        if str(cls) == "0":
            segment_person.append([seg, bbox])
    return segment_person

def pose_estimation(frame: Frame) -> (list, any):
    Debug.Log("Perform Detection")
    image = frame.get_ab_image()
    post_process_image = image.copy()
    img_seg = track_person(post_process_image)
    if (len(img_seg) == 0):
        Debug.Log("No Person Found, Skip", category="Detection")
        return None, None
    results = detection_keypoint(image)
    keypoints_list = detection_keypoint.get_xy_keypoint(results)
    bounding_box_list, id_list = detection_keypoint.get_bounding_box(results)
    if (keypoints_list == None):
        return None, None
    coordinate_list = []
    for keypoints, bbox, id in zip(keypoints_list, bounding_box_list, id_list):
        keypoint_location = {}
        keypoint_location_2d = {}
        keypoint_location_validity = {}
        color = Utility.generate_random_color()
        for i in range(0,int(len(keypoints)/2)):
            x = int(keypoints[i*2])
            y = int(keypoints[i*2+1])

            cv2.circle(post_process_image, (x,y), 3, color, -1)
            threed_coordinate = frame.get_coordinate_of(x, y)
            validity = get_validity(img_seg, threed_coordinate, x, y, frame.get_position())
            keypoint_location.__setitem__(keypoint_name[i], threed_coordinate)
            keypoint_location_2d.__setitem__(keypoint_name[i], np.array([x,y]))
            keypoint_location_validity.__setitem__(keypoint_name[i], validity)
            if validity:
                Debug.Log(keypoint_name[i] + ": (" + str(x) + "," + str(y) + "), 3D Depth: " + str(threed_coordinate) + ", 2D Depth: " + str(frame.get_depth_of(x,y)) + ", Validity:" + str(validity), category="Detection Miscellaneous")
        #input()
        coordinate_observed = get_coordinate(keypoint_location, keypoint_location_validity)
        
        if len(coordinate_observed) != 0:
            orientation_observed = orientation_calculation(keypoint_location_2d["Right Shoulder"], keypoint_location_2d["Left Shoulder"])
            coordinate_observed = np.append(coordinate_observed, orientation_observed)
        coordinate_list.append([coordinate_observed, bbox, id])
    #cv2.imshow("Tracking image from Hololens", image)
    return remove_duplicate(coordinate_list), post_process_image

def orientation_calculation(x, y):
    body_vector =  x - y
    unit_body_vector = body_vector/np.linalg.norm(body_vector)
    x_unit_vector = np.array([1,0])
    dot_x_body = np.dot(x_unit_vector,unit_body_vector)
    orientation_observed = np.arccos(dot_x_body)
    if (body_vector[1] < 0):
        orientation_observed += 3.14159
    return orientation_observed

def get_validity(image_segment, threed_coordinate, x, y, observer_coord):
    
    depth_norm = np.linalg.norm(threed_coordinate - observer_coord)
    if depth_norm < 0.3 or (x == 0 and y == 0):
        return False
    else:
        for seg in image_segment:
            if cv2.pointPolygonTest(seg[0], (x,y), False) > 0:
                return True
        return False

def get_coordinate(coordinate3d_data, validity):
    validity_priority = ["Shoulder","Ear", "Hip","Elbow","Wrist","Knee","Ankle"]
    for priority_str in validity_priority:
        if validity["Left "+priority_str] and validity["Right "+priority_str]:
            return (coordinate3d_data["Left "+priority_str] + coordinate3d_data["Left "+priority_str])/2
        elif validity["Left "+priority_str]:
            return coordinate3d_data["Left "+priority_str]
        elif validity["Right "+priority_str]:
            return coordinate3d_data["Right "+priority_str]
    return np.array([])

def remove_duplicate(coordinate_list):
    final_output_list = []
    for coordinate_combine in coordinate_list:
        coord = coordinate_combine[0][0:3]
        duplicate_found = False
        if len(final_output_list) == 0:
            final_output_list.append(coordinate_combine)
            continue
        if len(coord) == 0:
            continue
        else:
            for out_coordinate_combine in final_output_list:
                Debug.Log(coord, category="Detection remove duplicate")
                out_coord = out_coordinate_combine[0:3]
                Debug.Log(out_coord, category="Detection remove duplicate")
                if (np.linalg.norm(out_coord - coord) < 0.1):
                    Debug.Log("Duplicate found, not putting into list", category="Detection remove duplicate")
                    duplicate_found = True
        if not(duplicate_found):
            Debug.Log("Duplicate not found, safe to add to list", category="Detection remove duplicate")
            final_output_list.append(coordinate_combine)
    return final_output_list







