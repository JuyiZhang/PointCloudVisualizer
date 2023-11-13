import open3d as o3d
import numpy as np
from load  import *

timestamp_nu = 1698064389229



def simplify_point_cloud(timestamp):
    point_cloud_data = np.load(getImageName(timestamp,"P") + ".npy")
    point_cloud_data[:,2] = -point_cloud_data[:,2]
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(point_cloud_data.astype(np.float64))
    #o3d_pc.estimate_normals()
    o3d_pc_simplified = o3d_pc.voxel_down_sample(voxel_size=0.01)
    o3d_pc_simplified.estimate_normals()
    
    pc_mesh, pc_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pc_simplified)
    #print(pc_mesh)
    o3d.visualization.draw_geometries([o3d_pc])
    
simplify_point_cloud(timestamp_nu)
