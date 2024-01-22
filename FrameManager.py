import cv2
import numpy as np

import struct
import os

import Debug

type_def = {'abimage': '_ab_image.png', 'pose': '_master_pose.sci', 'depth': '_depth_map.sci', 'point_cloud': '_point_cloud.sci', 'folder': ''}

class SessionData:
    
    def __init__(self, image_type: str = 'long', save: bool = True, session_folder: str = '') -> None:
        self.image_type = image_type
        self.save = save
        self.session_folder = session_folder

# Each frame instance represent a frame received
class Frame:
    
    # Variables of Frame
    session: SessionData # The session managing the frame
    timestamp: int # The timestamp of the frame
    depth_map: np.ndarray # The depth map captured
    ab_img: np.ndarray # The absolute brightness image captured
    point_cloud: np.ndarray # The point cloud data captured
    master_pose: np.ndarray # The pose of the master device
    detected_image: any # The post-processed image after detection
    detection_finished = False # Indicate if the frame has been detected
    point_cloud_add_finished = False # Indicate if the frame is added to the overall point cloud
    device = "" # The ip address of incoming device
    
    def __init__(self, data, device, session: SessionData) -> None:
        Debug.Log("A New Frame Is Added", category="Frame")
        self.session = session
        self.device = device
        self.data_process(data)
    
    # Process the newly passed data
    def data_process(self, data):
        Debug.Log("Processing Data...", category="Frame")
        
        # Processing header and get data length
        self.timestamp = struct.unpack(">q", data[1:9])[0]
        depth_length = struct.unpack(">i", data[9:13])[0]
        ab_length = struct.unpack(">i", data[13:17])[0]
        pointcloud_length = struct.unpack(">i", data[17:21])[0]
        if (depth_length != 184320):
            Debug.Log("Depth Length: " + str(depth_length), category="stat")
            
        
        image_shape = (288,320) if self.session.image_type == "long" else (512,512)
        
        # Processing data for depth map
        data_pointer_begin = 21
        data_pointer_end = data_pointer_begin + depth_length
        self.depth_map = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.uint16).reshape(image_shape)
        
        # Processing data for ab image
        data_pointer_begin = data_pointer_end
        data_pointer_end = data_pointer_begin + ab_length
        self.ab_img = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.uint8).reshape(image_shape)
        self.ab_image_process(72)
        
        # Processing data for point cloud
        data_pointer_begin = data_pointer_end
        data_pointer_end = data_pointer_begin + pointcloud_length - 48
        self.point_cloud = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.float32).reshape((-1,3))
        
        # Processing data for master pose
        data_pointer_begin = data_pointer_end
        data_pointer_end = data_pointer_begin + 48
        self.master_pose = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.float32).reshape((-1,3))
        
        # Save data after finish processing
        self.data_save()
    
    # Saving the data to disk for future use only if session save is activated
    def data_save(self):
        if (self.session.save):
            Debug.Log("Saving Data...", category="Frame")
            save_folder = self.get_file_name('folder')
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            np.save(self.get_file_name("depth"), self.depth_map)
            np.save(self.get_file_name("point_cloud"), self.point_cloud)
            np.save(self.get_file_name("pose"), self.master_pose)
            cv2.imwrite(self.get_file_name("abimage"), self.ab_img)
        
    # Retrieving depth in terms of 2D coordinate
    def get_depth_of(self, x, y):
        return self.depth_map[y][x]
    
    # Retrieving 3D coordinate in terms of 2D coordinate
    def get_coordinate_of(self, x, y):
        image_width = 320 if self.session.image_type == "long" else 512
        return self.point_cloud[y*image_width+x]
    
    # Retrieving the pose of the frame
    def get_pose(self):
        return self.master_pose
    
    # Retrieving the position of frame
    def get_position(self):
        return self.master_pose[0]
    
    # Retrieving abimage for image recognition
    def get_ab_image(self):
        return self.ab_img
    
    # Getting the name of the image, select from 'abimage', 'pose', 'depth', and 'point_cloud"
    def get_file_name(self, type: str) -> str: 
        return self.session.session_folder + "/" + self.device + "/" + str(int(self.timestamp/1000000)) + ("" if type == "folder" else "/" + str(self.timestamp) + type_def[type])

    # Processing the collected ab image and convert it to yolo-compatible format
    def ab_image_process(self, medium=128):
        Debug.Log("Begin Processing Ab Image", category="Frame")
        image_width = self.ab_img.shape[0]
        image_height = self.ab_img.shape[1]
        min = np.min(self.ab_img)
        max = np.max(self.ab_img)
        for i in range(image_width):
            for j in range(image_height):
                raw_pxl_value = self.ab_img[i][j]
                processed_pxl_value = (raw_pxl_value-min)*256/(max-min)
                if (processed_pxl_value < medium):
                    processed_pxl_value = 128/medium*processed_pxl_value
                if (processed_pxl_value > medium):
                    processed_pxl_value = 128/(256-medium)*(processed_pxl_value - medium)+128
                
        self.ab_img = cv2.cvtColor(self.ab_img, cv2.COLOR_GRAY2RGB)
        Debug.Log("Ab Image Process Complete", category="Frame")
        

    
    