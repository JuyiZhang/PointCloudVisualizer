from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
import simplepbr
from load import *
from postprocessing import *
from debug_var import *
import numpy as np



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
        coordinate = getUserCoordinate(timestamp_nu, session_ts)
        print(coordinate)
        x_p = coordinate[0][0]
        y_p = coordinate[0][1]
        z_p = coordinate[0][2]
        x_r = eulerAngleConversion(coordinate[1][0])
        y_r = eulerAngleConversion(coordinate[1][1])
        z_r = eulerAngleConversion(coordinate[1][2])
        pcd = np.load("data_long/Session_"+str(session_ts)+"/Point_Cloud.sci.npy")
        for point in pcd.reshape(-1,3):
            pointmodel = loader.loadModel("assets/point.stl")
            pointmodel.setPos(point[2]*10, point[0]*10, point[1]*10)
            pointmodel.setScale(15)
            pointmodel.setColor(0.37,0.597,1.0)
            pointmodel.reparentTo(render)
        self.master.setPos(-z_p*10,x_p*10,y_p*10)
        self.master.setHpr(y_r+210,0,0)
        self.master.reparentTo(render)
        coord_observed, orient_observed = image_post_process_from_file(getImageName(timestamp_nu,"A",session_ts),timestamp_nu, session=session_ts, observer_coord=coordinate[0])
        print(coord_observed, orient_observed)
        self.observed = loader.loadModel("assets/human_representation.gltf")
        x_op = coord_observed[0]
        y_op = coord_observed[1]
        z_op = coord_observed[2]
        y_or = orient_observed
        self.observed.setPos(-z_op*10,x_op*10, 0)
        self.observed.setHpr(y_or/3.14*180+90,0,0)
        self.observed.reparentTo(render)
        base.disableMouse()
        self.camera.setPos(30,-10,160)
        self.camera.setHpr(-8,-90,0)
        
        #loadDepthData(timestamp)
        

app = Slugrace3D()
app.run()