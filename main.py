# Panda 3D Import
from direct.showbase.ShowBase import ShowBase
from direct.showbase.Loader import Loader
from direct.task import Task
from direct.gui.DirectGui import *
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import WindowProperties, NodePath, Texture, CardMaker
import simplepbr

# Accessing TCP server and Point Cloud Manager
import threading
import atexit

import numpy as np

import Debug, Utility
from TCPServer import TCPServer
from SessionManager import Session

class MuTA(ShowBase):

    devices_model = dict()
    devices_color = dict()
    post_processed_image: Texture
    ab_image: Texture
    buttons_connected_device = dict()

    def __init__(self):
        ShowBase.__init__(self)
        
        # Create variable for update
        # Session manages the capture session
        # TCP Server manages the connection
        
        self.updated = False
        self.session = Session(True)
        self.tcp_server = TCPServer(self.session)
        
        # Add a task to update the environment every frame
        self.taskMgr.add(self.update, "EnvironmentUpdate")
        
        self.disable_mouse()
        self.camera.setPos(30,-10,160)
        self.camera.setHpr(-8,-90,0)
        
        self.init_ui()
        
    def update(self, task):
        for device_address in self.tcp_server.get_connection_list():
            # If the device is not yet initiated by the system, assign a random color to the device and create the model to the device
            if not(device_address in self.devices_color.keys()):
                device_color = Utility.generate_random_color()
                self.devices_color[device_address] = device_color
                self.devices_model[device_address] = self.create_model("human_representation", color_rgb=device_color)
                self.new_address_button(device_address, device_color)
                
        """
        coordinate_dict = self.session.get_all_pose()
        observed_list = self.session.get_observed_list()
        if len(coordinate_dict.keys()) != 0:
            for device_address in coordinate_dict.keys():
                coordinate = coordinate_dict[device_address]
                Debug.Log("Proceed positioning self object with value " + coordinate.__str__(), category="Frame")
                self.set_model_pose(self.devices_model[device_address], coordinate[0], coordinate[1])
        if observed_list is not None:
            if len(observed_list) != 0:
                Debug.Log("Found body in space, creating instance")
            
        #self.render_point_cloud(self.session.get_point_cloud_data())
        self.render_observed_user(observed_list)"""    
        return Task.cont
    
    def update_ui(self):
        pp_image = self.session.get_post_processed_image()
        ab_image = self.session.get_ab_image()
        if pp_image != None:
            self.post_processed_image
        
    def set_master_device(self, address):
        print(address)
        old_main_device_address = self.session.set_main_device(address)
        if old_main_device_address != '':
            old_main_device_button = self.buttons_connected_device[old_main_device_address]
            self.button_set_state(old_main_device_button)
        new_main_device_button = self.buttons_connected_device[address]
        self.button_set_state(new_main_device_button, state='active')
        Debug.Log("Main Device Set To " + address)
        
    def button_set_state(self, button, state='pending'):
        if state == 'pending':
            frameColor = (0,0,0,0)
        elif state == 'active':
            frameColor = (0.8,0.8,0.8,0.4)
        button['frameColor'] = frameColor
    
    def new_address_button(self, device_address, color):
        position = -len(self.buttons_connected_device.values())*0.08-0.1
        self.buttons_connected_device.__setitem__(
            device_address, 
            DirectButton(
                self.buttons_connected_device['parent'], 
                command=self.set_master_device, 
                extraArgs=[device_address], 
                text=device_address,
                text_fg=(color[0], color[1], color[2], 1),
                frameColor=(0,0,0,0),
                pos=(0,0,position), 
                scale=0.05
                )
            )
        if len(self.buttons_connected_device) == 2:
            self.set_master_device(device_address)
    
    def render_point_cloud(self, pointCloudData):
        for point in pointCloudData.reshape(-1,3):
            self.create_model("point", "stl", position=point, scale = 15, positional_scale_factor = 10, color_rgb = [0.37,0.597,1.0])
            
    def create_model(self, model_name, model_type = "gltf", position = np.array([]), color_rgb = None, rotation_y = 0, scale = 1, positional_scale_factor = 1, rev_x = False) -> NodePath:
        model = self.loader.loadModel("assets/" + model_name + "." + model_type)
        if len(position) == 3:
            pos_scaled_x = position[2] if rev_x else -position[2]
            model.setPos(pos_scaled_x * positional_scale_factor, position[0] * positional_scale_factor, position[1] * positional_scale_factor)
        if (rotation_y != 0):
            model.setHpr(rotation_y, 0, 0)
        if (scale != 0):
            model.setScale(scale)
        if color_rgb != None:
            model.setColor(color_rgb[0], color_rgb[1], color_rgb[2])  
        model.reparentTo(self.render)
        return model
    
    def set_model_pose(self, model, position = None, forward_vector = None):
        if len(position) == 3:
            model.setPos(-position[2], position[0], position[1])
        if len(forward_vector) == 3:
            rotate_angle_raw = Utility.to_degree(np.arctan2(forward_vector[0],forward_vector[2]))
            if (forward_vector[2]>0):
                rotate_angle_raw += 180    
            
            model.setHpr(Utility.euler_simplify(rotate_angle_raw), 0, 0)
        
    def render_observed_user(self, coord_list):
        if (coord_list != None and len(coord_list) > 0):
            for coord_observed in coord_list:
                if (len(coord_observed) == 0):
                    continue
                observed_position = []
                observed_rotation = []
                if (coord_observed[0] != None):
                    observed_position = [-coord_observed[2], coord_observed[0], 0]
                if (len(coord_observed) == 4):
                    observed_rotation = [0, coord_observed[3], 0]
                self.create_model("human_representation", "gltf", observed_position, observed_rotation)
    
    def setup(self):
        self.base.setBackgroundColor(0.113,0.12109375,0.13671875)
        simplepbr.init()
        props = WindowProperties()
        props.setTitle('3D Visualization for Tracking')
        props.setOrigin(100,300)
        props.setSize(1200,675)
        self.base.win.requestProperties(props)
        
    def init_ui(self):
        
        # Render the image captured by HoloLens
        image_maker = CardMaker('image_maker')
        ab_image_card = self.a2dTopRight.attachNewNode(image_maker.generate())
        pp_image_card = self.a2dTopRight.attachNewNode(image_maker.generate())
        
        ab_image_card.setScale(0.45)
        pp_image_card.setScale(0.45)
        ab_image_card.setPos(-0.5,0,-0.5)
        pp_image_card.setPos(-0.5,0,-1)
        
        self.ab_image = self.loader.loadTexture("assets/ab_image_sample.png")
        self.post_processed_image = self.loader.loadTexture("assets/pp_image_sample.png")
        
        ab_image_card.setTexture(self.ab_image)
        pp_image_card.setTexture(self.post_processed_image)
        
        self.buttons_connected_device.__setitem__("parent", self.a2dTopCenter)
        OnscreenText("Select Main Address", pos=(0,-0.1), parent=self.a2dTopCenter)
        
    def destroy(self):
        pass #Remember to call close connection on TCP Server class
        



app = MuTA()
atexit.register(app.destroy)
app.run()