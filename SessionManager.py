import time
import threading
from multiprocessing import Pool
import os

import numpy as np

import Debug

from FrameManager import Frame
import DetectionManager
from FrameFileManager import get_file_name
#from PointCloudManagerAsync import PointCloudManager

class Session:
    
    detection_flag = False # Define if a new frame is ready for detection
    detection_in_process = False # Define if a detection is in process
    observed_list = [] # The list of observed body
    devices_frame = dict() # The ip_address/pose pair
    main_device = "" # The main device for display collected data
    detected_image = np.array([]) # The image after post processing
    master_frame_updated = False
        
    def __init__(self, timestamp, save_data: bool = True, image_type: str = "long", use_secondary_tracking = True) -> None:
        self.timestamp = timestamp
        self.session_folder = "data/Session_" + str(int(timestamp))
        if not os.path.isdir(self.session_folder):
            os.mkdir(self.session_folder)
        self.save = save_data
        self.image_type = image_type
        #self.point_cloud_manager = PointCloudManager(timestamp=self.timestamp)
        self.current_frame = None
        self.use_secondary_tracking = use_secondary_tracking
        threading.Thread(target=self.frame_detection, daemon=True).start()
        #Process(target=self.frame_add_point_cloud, daemon=True).start()
        
    def new_frame(self, timestamp, device):
        if (self.main_device == ""):
            self.main_device = device
        if (device == self.main_device):
            Debug.Log("Updating frame for main device", category="Frame")
            self.current_frame = Frame(timestamp, device, self.session_folder)
            self.master_frame_updated = True
            self.devices_frame[device] = self.current_frame.get_pose()
            self.detection_flag = True
        else:
            Debug.Log("Updating frame for secondary device", category="Frame")
            self.devices_frame[device] = Frame(timestamp, device, self.session_folder).get_pose()
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
            if self.detection_flag and not(self.detection_in_process) and self.use_secondary_tracking:
                self.detection_in_process = True
                frame_observed_list, self.detected_image = DetectionManager.pose_estimation(self.current_frame)
                if frame_observed_list is not None:
                    # Go over all the person detected
                    for observed_frame in frame_observed_list:
                        if observed_frame[1] is None:
                            continue
                        frame_found = False
                        # Go over the existing list to see if there are id that can be updated cooresponding to the currently set person detected
                        for i in range(0,len(self.observed_list)):
                            if(self.observed_list[i][1] == observed_frame[1]):
                                self.observed_list[i][0] = observed_frame[0]
                                frame_found = True
                                break
                        if not(frame_found):
                            self.observed_list.append(observed_frame)
                for item in self.observed_list:
                    Debug.Log("The detected person coordinate in time " + str(self.current_frame.timestamp) + " is " + item.__str__(), category="Detection")
                self.detection_in_process = False
                self.detection_flag = False
    
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
    
    def set_use_secondary_tracking(self, use_secondary_tracking = True):
        self.use_secondary_tracking = use_secondary_tracking