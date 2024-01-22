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

track_history = []
    
keypoint_name = ["Nose","Left Eye","Right Eye","Left Ear","Right Ear","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
m1_gpu_availability = torch.backends.mps.is_available()
nv_gpu_availability = torch.backends.cudnn.is_available()
if m1_gpu_availability:
    Debug.Log("The M-Series GPU is available, Allocating Resources...")
elif nv_gpu_availability:
    Debug.Log("The Nvidia Cudnn is available, Allocating Resources...")
else:
    Debug.Log("The GPU is not available, falling back to CPU/Using default torch device")
track_history = defaultdict(lambda: [])
detection_keypoint = DetectKeypoint()
    
def track_person(image):
    if m1_gpu_availability:
        results = model.track(image, device="mps", persist=True)
    elif nv_gpu_availability:
        results = model.track(image, persist=True)
    else:
        results = model.track(image, persist=True)
    boundingBoxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    if len(boundingBoxes) == 0:
        return
    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
    segmentation_contours_idx = []
    for seg in results[0].masks.xy:
        segment = np.array(seg, dtype=np.int32)
        segmentation_contours_idx.append(segment)
    segment_person = []
    if (results[0].boxes.id == None):
        for cls, bbox, seg in zip(classes, boundingBoxes, segmentation_contours_idx):
            if str(cls) == "0":
                segment_person.append([seg, bbox, 65535+random.randint(0,65535)])
    else:
        tracking_ids = np.array(results[0].boxes.id.cpu().tolist())
        for cls, bbox, seg, id in zip(classes, boundingBoxes, segmentation_contours_idx, tracking_ids):
            if str(cls) == "0":
                segment_person.append([seg, bbox, id])
    return segment_person

def pose_estimation(frame: Frame) -> (list, any):
    Debug.Log("Perform Detection")
    image = frame.get_ab_image()
    post_process_image = image.copy()
    img_seg = track_person(post_process_image)
    if img_seg is None:
        Debug.Log("Invalid track, skip", category="Detection")
        return None, None
    if (len(img_seg) == 0):
        Debug.Log("No Person Found, Skip", category="Detection")
        return None, None
    results = detection_keypoint(image)
    keypoints_list = detection_keypoint.get_xy_keypoint(results)
    bounding_box_list = detection_keypoint.get_bounding_box(results)
    if (keypoints_list == None):
        return None, None
    coordinate_list = []
    for keypoints, bbox in zip(keypoints_list, bounding_box_list):
        keypoint_location = {}
        keypoint_location_2d = {}
        keypoint_location_validity = {}
        color = Utility.generate_random_color()
        person_segment, person_id = correlate_bounding_box(bbox, img_seg)
        cv2.polylines(post_process_image, [person_segment], isClosed=False, color=color, thickness=1)
        for i in range(0,int(len(keypoints)/2)):
            x = int(keypoints[i*2])
            y = int(keypoints[i*2+1])
            
            cv2.circle(post_process_image, (x,y), 3, color, -1)
            threed_coordinate = frame.get_coordinate_of(x, y)
            validity = get_validity(person_segment, threed_coordinate, x, y, frame.get_position())
            keypoint_location.__setitem__(keypoint_name[i], threed_coordinate)
            keypoint_location_2d.__setitem__(keypoint_name[i], np.array([x,y]))
            keypoint_location_validity.__setitem__(keypoint_name[i], validity)
            Debug.Log(keypoint_name[i] + ": (" + str(x) + "," + str(y) + "), 3D Depth: " + str(threed_coordinate) + ", 2D Depth: " + str(frame.get_depth_of(x,y)) + ", Validity:" + str(validity), category="Detection Miscellaneous")
        #input()
        coordinate_observed = get_coordinate(keypoint_location, keypoint_location_validity)
        
        if len(coordinate_observed) != 0:
            orientation_observed = orientation_calculation(keypoint_location_2d["Right Shoulder"], keypoint_location_2d["Left Shoulder"])
            coordinate_observed = np.append(coordinate_observed, orientation_observed)
        coordinate_list.append([coordinate_observed, person_id])
    cv2.imwrite("cache/" + str(frame.timestamp) + ".png", post_process_image)
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

# Input bounding box target, find the correlated segment and id
def correlate_bounding_box(bounding_box_target, bounding_box_id_seg_array):
    Debug.Log("Input Bounding Box: " + bounding_box_target.__str__())
    for i in range(0, len(bounding_box_id_seg_array)):
        bounding_box_to_inspect = bounding_box_id_seg_array[i][1]
        if np.linalg.norm(bounding_box_target - bounding_box_to_inspect) < 10: # we regard this as same rectangle
            Debug.Log("Bounding Box found: " + bounding_box_to_inspect.__str__())
            return bounding_box_id_seg_array[i][0], bounding_box_id_seg_array[i][2]
    return None, None

def get_validity(image_segment, threed_coordinate, x, y, observer_coord):
    
    depth_norm = np.linalg.norm(threed_coordinate - observer_coord)
    if depth_norm < 0.3 or (x == 0 and y == 0):
        return False
    else:
        if image_segment is None:
            Debug.Log("No valid segment is assigned, regard as invalid", category="Detection Miscellaneous")
            return False
        if cv2.pointPolygonTest(image_segment, (x,y), False) > 0:
            Debug.Log("Passed Polygon Test", category="Detection Miscellaneous")
            return True
        Debug.Log("Failed Polygon Test", category="Detection Miscellaneous")
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







