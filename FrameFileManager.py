import os
import cv2
import struct
import numpy as np

import Debug

"""from numba import jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning"""

from multiprocessing import Process
"""
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)"""

type_def = {'abimage': '_ab_image.png', 'pose': '_pose.sci', 'depth': '_depth_map.sci', 'point_cloud': '_point_cloud.sci', 'folder': ''}

def data_process(data, device, device_timestamp, session_folder):
    Process(target=data_process_async, args=(data, device, session_folder, device_timestamp), daemon=True).start()

# Process the newly passed data
def data_process_async(data, device: str, session_folder: str, device_timestamp):
    
    Debug.Log("Processing Data...", category="Frame")
    
    # Processing header and get data length
    timestamp = struct.unpack(">q", data[1:9])[0]
    depth_length = struct.unpack(">i", data[9:13])[0]
    ab_length = struct.unpack(">i", data[13:17])[0]
    pointcloud_length = struct.unpack(">i", data[17:21])[0]
    
    if (depth_length != 184320):
        Debug.Log("Depth Length: " + str(depth_length), category="stat")

    image_shape = (288,320)
    
    # Processing data for depth map
    data_pointer_begin = 21
    data_pointer_end = data_pointer_begin + depth_length
    depth_map = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.uint16).reshape(image_shape)
    
    # Processing data for ab image
    data_pointer_begin = data_pointer_end
    data_pointer_end = data_pointer_begin + ab_length
    ab_img = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.uint8).reshape(image_shape)
    cv2.convertScaleAbs(ab_img, ab_img, 0.7, 64)
    
    # Processing data for point cloud
    data_pointer_begin = data_pointer_end
    data_pointer_end = data_pointer_begin + pointcloud_length - 48
    point_cloud = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.float32).reshape((-1,3))
    
    # Processing data for master pose
    data_pointer_begin = data_pointer_end
    data_pointer_end = data_pointer_begin + 48
    pose = np.frombuffer(data[data_pointer_begin:data_pointer_end], np.float32).reshape((-1,3))
    
    # Save data after finish processing
    Debug.Log("Saving Data...", category="Frame")
    save_folder = get_file_name('folder', session_folder, device, timestamp)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    np.save(get_file_name("depth", session_folder, device, timestamp), depth_map)
    np.save(get_file_name("point_cloud", session_folder, device, timestamp), point_cloud)
    np.save(get_file_name("pose", session_folder, device, timestamp), pose)
    cv2.imwrite(get_file_name("abimage", session_folder, device, timestamp), ab_img)
        
    device_timestamp[device] = timestamp
        

# Getting the name of the image, select from 'abimage', 'pose', 'depth', and 'point_cloud"
def get_file_name(type: str, session_folder: str, device: str, timestamp: int) -> str: 
    return session_folder + "/" + device + "/" + str(int(timestamp/1000000)) + ("" if type == "folder" else "/" + str(timestamp) + type_def[type])

