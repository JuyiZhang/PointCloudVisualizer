import struct
import time
import numpy as np
import cv2
from DetectionManager import DetectionManager
import os

from load import getImageName
from debug_var import *

image_height = 288
image_width = 320

def image_post_process_from_file(fileName, timestamp, medium=32, min=0, max=256, session=None, observer_coord=np.array([0,0,0])):
    image = np.load(fileName + ".npy")
    detection = DetectionManager()
    cv2.imwrite(fileName + ".png", image_post_process(image, medium, min, max))
    return detection.poseEstimation(cv2.imread(fileName + ".png"), timestamp, session, observer_coord)

def image_post_process(image, medium=128, min=0, max=256): # As we are using B&W image, an np array can be more efficient
    image_width = image.shape[0]
    image_height = image.shape[1]
    for i in range(image_width):
        for j in range(image_height):
            if image[i][j] < medium*2:
                image[i][j] *= 128/medium
            else:
                image[i][j] = 255
    return image

def data_post_process(data, save_folder, connection):
    timestamp = struct.unpack(">q", data[1:9])[0]
    depth_length = struct.unpack(">i", data[9:13])[0]
    ab_length = struct.unpack(">i", data[13:17])[0]
    pointcloud_length = struct.unpack(">i", data[17:21])[0]
    currentDataLength = depth_length + ab_length + pointcloud_length + 21
    depthMap_img_np = np.frombuffer(data[21:21+depth_length], np.uint16).reshape((image_height,image_width))
    ab_img_np = np.frombuffer(data[21+depth_length:21+depth_length+ab_length], np.uint8).reshape((image_height,image_width))
    pointcloud_np = np.frombuffer(data[21+depth_length+ab_length: currentDataLength-48], np.float32).reshape((-1,3))
    coordinate = np.frombuffer(data[currentDataLength-48: currentDataLength], np.float32).reshape((-1,3))
    #track(ab_img_np)
    #timestamp = str(int(time.time()))
    if not os.path.exists(save_folder + "/" + str(int(timestamp/1000000))):
        os.mkdir(save_folder + "/" + str(int(timestamp/1000000)))
    cv2.imwrite(save_folder + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_Abimage.png", ab_img_np)
    np.save(save_folder + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_Abimage.sci", ab_img_np)
    np.save(save_folder + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_Depth_Map.sci", depthMap_img_np)
    np.save(save_folder + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_Point_Cloud_Data.sci", pointcloud_np)
    np.save(save_folder + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_Coordinate_Data.sci", coordinate)
    timestamp_nu = timestamp
    #coord_observed, orient_observed = image_post_process_from_file(getImageName(timestamp_nu,"A",session_ts),timestamp_nu, session=session_ts, observer_coord=coordinate[0])
    connection.send(np.array((0,0,1,0,0,0)).astype(np.float32).tobytes())
    #print("Sending coordinate of" + coord_observed)
    currentTimestamp = int(time.time_ns()/1000000)
    #print("Data Size: " + str(len(data)))
    #print("Current Timestamp:" + str(currentTimestamp))
    #print("Timestamp at receive:" + str(timestamp))
    #print("Delay: " + str((currentTimestamp-timestamp)/1000))
#image_post_process_from_file("data_long/1697120292500Abimage.sci.npy")


    