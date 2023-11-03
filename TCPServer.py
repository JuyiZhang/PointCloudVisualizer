import socket
import struct
import sys
import os
import numpy as np
import cv2
import time
import pickle as pkl
from detection import *
import traceback
from load import *
from postprocessing import *
from http_operation import *
import atexit


def tcp_server():
    serverHost = '' # localhost
    serverPort = 9090
    add_host(socket.gethostbyname(socket.gethostname()))
    save_folder = 'data_long'
    print("Initializing Object Detection...")
    #track(cv2.imread("test/test.png"))
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # Create a socket
    sSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind server to port
    try:
        sSock.bind((serverHost, serverPort))
        print('Server bind to port '+str(serverPort))
    except socket.error as msg:
        print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        return

    sSock.listen(10)
    print('Start listening...')
    sSock.settimeout(3.0)
    while True:
        try:
            conn, addr = sSock.accept() # Blocking, wait for incoming connection
            break
        except Exception:
            continue

    print('Connected with ' + addr[0] + ':' + str(addr[1]))

    finalData = bytearray(0)
    currentHeader = ""
    currentDataLength = 0

    while True:
        # Receiving from client
        try:
            data = conn.recv(1024*1024*4+100)
            
            if len(data)==0:
                continue
            if currentHeader != "": # Means there are data that is not yet received
                finalData += bytearray(data)
                
                if currentDataLength > len(finalData):
                    continue
                else:
                    if currentHeader == "c":
                        currentHeader = ""
                        data_post_process(finalData, save_folder)
                        currentDataLength = 0
                        finalData.clear()    
                    continue
                    
            header = data[0:1].decode('utf-8')
            print('--------------------------\nHeader: ' + header)
            


            if header == 'c':
                currentDataLength = 0
                finalData.clear()
                # Retrieve Depth Sensor Data
                finalData += bytearray(data)
                currentHeader = 'c'
                depth_length = struct.unpack(">i", data[9:13])[0]
                ab_length = struct.unpack(">i", data[13:17])[0]
                pointcloud_length = struct.unpack(">i", data[17:21])[0]
                currentDataLength = depth_length + ab_length + pointcloud_length + 21
                if len(data) < currentDataLength: #Insufficient data, continue receive
                    continue
                else:
                    currentHeader = ""
                    data_post_process(finalData, save_folder)
                    currentDataLength = 0
                    finalData.clear()
                
            if header == 'f':
                # save spatial camera images
                data_length = struct.unpack(">i", data[1:5])[0]
                ts_left, ts_right = struct.unpack(">qq", data[5:21])

                N = int(data_length/2)
                LF_img_np = np.frombuffer(data[21:21+N], np.uint8).reshape((480,640))
                RF_img_np = np.frombuffer(data[21+N:21+2*N], np.uint8).reshape((480,640))
                cv2.imwrite(save_folder + str(ts_left)+'_LF.tiff', LF_img_np)
                cv2.imwrite(save_folder + str(ts_right)+'_RF.tiff', RF_img_np)
                print('Image with ts %d and %d is saved' % (ts_left, ts_right))

        except:
            print(traceback.print_exc())
            userinput = input("Error on receiving data, press e to exit, or any other key to continue")
            if userinput == 'e' or userinput == 'E':
                break
            else:
                continue
                
    print('Closing socket...')
    sSock.close()

def onReceive():
    return 0

def onExit():
    del_host()
    
atexit.register(onExit)

if __name__ == "__main__":
    tcp_server()
