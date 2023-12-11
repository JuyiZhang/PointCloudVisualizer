import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PointCloudManager:
    
    PointCloudData = np.array([0,0,0])
    PointCloudDataSimplified = np.array([0,0,0])
    O3DPointCloud = o3d.geometry.PointCloud()
    
    def __init__(self, session_dir = '', timestamp = 0) -> None:
        if session_dir != '':
            self.PointCloudData = np.load(session_dir + "/Point_Cloud.sci.npy").reshape(-1,3)
            self.PointCloudDataSimplified = self.PointCloudData
            self.O3DPointCloud.points = o3d.utility.Vector3dVector(self.PointCloudData.astype(np.float64))
            self.directory = session_dir
        elif timestamp != 0:
            self.directory = "data_long/Session_"+str(timestamp)
    
    def new_point_cloud_data(self, pointCloud):
        removed_point_cloud = self.remove_ceiling(pointCloud.reshape(-1,3))
        removed_point_cloud[:,2] = -removed_point_cloud[:,2]
        #print(str(removed_point_cloud.shape)+", "+str(self.PointCloudDataSimplified.shape))
        self.PointCloudData = np.append(self.PointCloudDataSimplified, removed_point_cloud).reshape(-1,3)
        #print(self.PointCloudData.shape)
        self.post_process()
    
    def post_process(self):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(self.PointCloudData.astype(np.float64))
        o3d_pc_simplified = o3d_pc.voxel_down_sample(voxel_size=0.07)
        #o3d_pc_simplified, r = o3d_pc_simplified.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
        #o3d_pc_simplified.estimate_normals()
        self.O3DPointCloud = o3d_pc_simplified
        #self.view_point_cloud()
        self.PointCloudDataSimplified = np.asarray(o3d_pc_simplified.points).reshape(-1,3)
        self.save_point_cloud_data()
        
    def get_point_cloud_data(self):
        return self.PointCloudDataSimplified
    
    def save_point_cloud_data(self):
        np.save(self.directory + "/Point_Cloud.sci", self.PointCloudDataSimplified)
        
    def remove_ceiling(self, point_cloud_data):
        coordinate_to_remove = []
        for i in range(0,point_cloud_data.shape[0]):
            z_data = point_cloud_data[i][1]
            if z_data > 1 or z_data < -1:
                coordinate_to_remove.append(i)
        return np.delete(point_cloud_data, coordinate_to_remove, 0)
    
    def view_point_cloud(self):
        o3d.visualization.draw_geometries([self.O3DPointCloud])