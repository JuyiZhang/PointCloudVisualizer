import time
import threading
from multiprocessing import Pool
import os

import numpy as np

import Debug

from FrameManager import Frame, SessionData
import DetectionManager
from PointCloudManagerAsync import PointCloudManager

class Session():
    
    detection_flag = False # Define if a new frame is ready for detection
    detection_in_process = False # Define if a detection is in process
    observed_list = [] # The list of observed body
    devices_frame = dict() # The ip_address/pose pair
    main_device = "" # The main device for display collected data
    detected_image = np.array([]) # The image after post processing
    master_frame_updated = False
        
    def __init__(self, save_data: bool = True, image_type: str = "long") -> None:
        self.timestamp = time.time()
        self.session_folder = "data/Session_" + str(int(time.time()))
        self.session_data = SessionData(image_type, save_data, self.session_folder)
        if not os.path.isdir(self.session_folder):
            os.mkdir(self.session_folder)
        self.save = save_data
        self.image_type = image_type
        self.point_cloud_manager = PointCloudManager(timestamp=self.timestamp)
        self.current_frame = None
        threading.Thread(target=self.frame_detection, daemon=True).start()
        #Process(target=self.frame_add_point_cloud, daemon=True).start()
        
    def new_frame(self, data, device):
        if (device == self.main_device):
            self.current_frame = Frame(data, device, self.session_data)
            self.master_frame_updated = True
            self.devices_frame[device] = self.current_frame.get_pose()
            self.detection_flag = True
        else:
            self.devices_frame[device] = Frame(data, device, self.session_data).get_pose()
        #self.point_cloud_manager.new_point_cloud_data(self.current_frame.point_cloud)
        
            #Process(target=DetectionManager.pose_estimation, args=(current_frame_queue, frame_result)).start()
            #threading.Thread(target=self.obtain_frame_data, args=frame_result).start()
        #self.frame_task_queue_point_cloud.append(self.current_frame)
    
    # Setting new main device and return the old main device address
    def set_main_device(self, device) -> str: 
        old_main_device = self.main_device
        self.main_device = device
        return old_main_device
    
    def frame_detection(self):
        while(True):
            if self.detection_flag and not(self.detection_in_process):
                self.detection_in_process = True
                self.observed_list, self.detected_image = DetectionManager.pose_estimation(self.current_frame)
                if self.observed_list is not None:
                    for item in self.observed_list:
                        Debug.Log("The detected person coordinate in frame " + str(self.current_frame.timestamp) + " is " + item.__str__(), category="Detection")
                self.detection_in_process = False
                self.detection_flag = False
    
    """def frame_add_point_cloud(self):
        while True:
            if len(self.frame_task_queue_point_cloud) > 0:
                point_cloud_frame = self.frame_task_queue_point_cloud.pop(0)
                Debug.Log("Begin Processing Frame " + str(id(point_cloud_frame)), category="Session")
                if len(self.frame_task_queue_point_cloud) < 10: # If there are too many frame waiting to be recognized, we simply skip to prevent occupation of memory
                    self.point_cloud_manager.new_point_cloud_data(point_cloud_frame.point_cloud)"""
    
    def get_point_cloud_data(self):
        return self.point_cloud_manager.get_point_cloud_data()
    
    def get_observed_list(self):
        return self.observed_list

    def get_master_pose(self):
        if self.current_frame is not None:
            return self.current_frame.get_pose()
        else:
            return np.array([])                
                
    def get_all_pose(self):
        return self.devices_frame
    
    def get_post_processed_image(self):
        if self.detected_image is not None:
            return self.detected_image
        else:
            return self.get_ab_image()
        
    def get_ab_image(self):
        if self.current_frame is not None:
            return self.current_frame.get_ab_image()
        else:
            return None
        