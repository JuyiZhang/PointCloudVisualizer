from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
import simplepbr
from load import *
from postprocessing import *
from debug_var import *
import numpy as np
import time
import threading
from TCPServer import tcp_server
from point_cloud_manager import *


class Slugrace3D(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        base.setBackgroundColor(0.113,0.12109375,0.13671875)
        simplepbr.init()
        props = WindowProperties()
        props.setTitle('3D Visualization for Tracking')
        props.setOrigin(100,300)
        props.setSize(1200,675)
        base.win.requestProperties(props)
        self.master = loader.loadModel("assets/human_representation.gltf")
        coordinate = np.array()
        pcd_mgr = PointCloudManager()
        coord_list = []
        threading.Thread(target=tcp_server, args=(coordinate, pcd_mgr, coord_list))
        coordinate = getUserCoordinate(timestamp_nu, session_ts)
        print(coordinate)
        x_p = coordinate[0][0]
        y_p = coordinate[0][1]
        z_p = coordinate[0][2]
        y_r = eulerAngleConversion(coordinate[1][1])#np.arctan(coordinate[1][1]/coordinate[1][0]) #Verify!
        pcd = np.load("data_long/Session_"+str(session_ts)+"/Point_Cloud.sci.npy")
        for point in pcd.reshape(-1,3):
            pointmodel = loader.loadModel("assets/point.stl")
            pointmodel.setPos(point[2]*10, point[0]*10, point[1]*10)
            pointmodel.setScale(15)
            pointmodel.setColor(0.37,0.597,1.0)
            pointmodel.reparentTo(render)
        self.master.setPos(-z_p*10,x_p*10,y_p*10)
        self.master.setHpr(y_r,0,0)
        self.master.reparentTo(render)
        timeopbegin = int(time.time_ns())
        coord_list = image_post_process_from_file(getImageName(timestamp_nu,"A",session_ts),timestamp_nu, session=session_ts, observer_coord=coordinate[0])
        timeop = int(time.time_ns()) - timeopbegin
        print("Total time cost equals " + str(timeop/1000000))
        if (coord_list != None):
            for coord_observed in coord_list:
                print(coord_observed)
                if (len(coord_observed) == 0):
                    continue
                observed_person = loader.loadModel("assets/human_representation.gltf")
                x_op = 0
                z_op = 0
                y_or = 0
                if (coord_observed[0] != None):
                    x_op = coord_observed[0]
                    z_op = coord_observed[2]
                if (len(coord_observed) == 4):
                    y_or = coord_observed[3]
                observed_person.setPos(-z_op*10,x_op*10, 0)
                observed_person.setHpr(y_or/3.14*180,0,0)
                observed_person.reparentTo(render)
        base.disableMouse()
        self.camera.setPos(30,-10,160)
        self.camera.setHpr(-8,-90,0)
        
        #loadDepthData(timestamp)
        

app = Slugrace3D()
app.run()