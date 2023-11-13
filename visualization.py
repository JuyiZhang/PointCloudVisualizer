from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
import simplepbr
from load import *
from postprocessing import *

timestamp = 1698064389229

class Slugrace3D(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        simplepbr.init()
        props = WindowProperties()
        props.setTitle('3D Visualization for Tracking')
        props.setOrigin(100,300)
        props.setSize(1200,675)
        base.win.requestProperties(props)
        self.master = loader.loadModel("assets/human_representation.gltf")
        coordinate = getUserCoordinate(timestamp)
        print(coordinate[1])
        x_p = coordinate[0][0]
        y_p = coordinate[0][1]
        z_p = coordinate[0][2]
        x_r = eulerAngleConversion(coordinate[1][0])
        y_r = eulerAngleConversion(coordinate[1][1])
        z_r = eulerAngleConversion(coordinate[1][2])
        self.master.setPos(z_p,x_p,y_p)
        self.master.setHpr(y_r,0,0)
        self.master.reparentTo(render)
        coord_observed, orient_observed = image_post_process_from_file(getImageName(timestamp,"A"),timestamp)
        print(coord_observed, orient_observed)
        self.observed = loader.loadModel("assets/human_representation.gltf")
        x_op = coord_observed[0]
        y_op = coord_observed[1]
        z_op = coord_observed[2]
        y_or = orient_observed
        self.observed.setPos(z_op*10,x_op*10, 0)
        self.observed.setHpr(y_or/3.14*180,0,0)
        self.observed.reparentTo(render)
        base.disableMouse()
        self.camera.setPos(0,0,40)
        self.camera.setHpr(0,-90,0)
        loadDepthData(timestamp)
        

app = Slugrace3D()
app.run()