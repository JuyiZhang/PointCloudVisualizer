import cv2
import numpy as np

import struct
import os

import Debug
from FrameFileManager import get_file_name

type_def = {'abimage': '_ab_image.png', 'pose': '_master_pose.sci', 'depth': '_depth_map.sci', 'point_cloud': '_point_cloud.sci', 'folder': ''}

# Each frame instance represent a frame received
class Frame:
    
    # Variables of Frame
    timestamp: int # The timestamp of the frame
    pose: np.ndarray
    point_cloud: np.ndarray
    depth_map: np.ndarray
    device: str # The ip address of incoming device
    session_folder: str
    
    def __init__(self, timestamp: int, device: str, session_folder: str) -> None:
        Debug.Log("A New Frame Is Added", category="Frame")
        self.device = device
        self.timestamp = timestamp
        self.session_folder = session_folder
        self.point_cloud = np.load(self.get_file_name("point_cloud"))
        self.depth_map = np.load(self.get_file_name("depth"))
        self.pose = np.load(self.get_file_name("pose"))
        
    # Retrieving depth in terms of 2D coordinate
    def get_depth_of(self, x, y):
        return self.depth_map[y][x]
    
    # Retrieving 3D coordinate in terms of 2D coordinate
    def get_coordinate_of(self, x, y):
        return self.point_cloud[y*320+x]
    
    # Retrieving the pose of the frame
    def get_pose(self) -> list:
        return self.pose.tolist()
    
    # Retrieving the position of frame
    def get_position(self) -> list:
        return self.pose[0].tolist()
    
    # Retrieving abimage for image recognition
    def get_ab_image(self):
        return cv2.imread(self.get_file_name("abimage"))
    
    def get_file_name(self, type) -> str:
        return get_file_name(type, self.session_folder, self.device, self.timestamp) + ("" if type == "abimage" else ".npy")
        

    
    