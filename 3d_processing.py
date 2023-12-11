import open3d as o3d
import numpy as np
from load  import *
from debug_var import *
from point_cloud_manager import PointCloudManager

pc = PointCloudManager(timestamp=session_ts)

def simplify_point_cloud(point_cloud_data):
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(point_cloud_data.astype(np.float64))
    #o3d_pc.estimate_normals()
    o3d_pc_simplified = o3d_pc.voxel_down_sample(voxel_size=0.05)
    o3d_pc_simplified.estimate_normals()
    return o3d_pc_simplified
    
    pc_mesh, pc_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pc_simplified)
    #print(pc_mesh)
    o3d.visualization.draw_geometries([o3d_pc])

def point_cloud_post_process(point_cloud_data):
    o3d_pc = o3d.geometry.PointCloud()
    
    o3d_pc.points = o3d.utility.Vector3dVector(remove_ceiling(point_cloud_data).astype(np.float64))
    #o3d_pc.estimate_normals()
    o3d_pc_simplified = o3d_pc.voxel_down_sample(voxel_size=0.05)
    o3d_pc_simplified, num = o3d_pc_simplified.remove_statistical_outlier(nb_neighbors=80, std_ratio=0.5)
    
    o3d_pc_simplified.estimate_normals()
    
    return o3d_pc_simplified

def remove_ceiling(point_cloud_data):
    max_z_data = 0
    coordinate_to_remove = []
    for i in range(0,point_cloud_data.shape[0]):
        z_data = point_cloud_data[i][1]
        if z_data > 1 or z_data < -2:
            coordinate_to_remove.append(i)
    return np.delete(point_cloud_data, coordinate_to_remove, 0)
            
            

def reconstruct_point_cloud_session(path):
    timestamp_minsec_array = []
    timestamp_array = []
    for directory in os.listdir(path):
        if os.path.isfile(path + "/" + directory):
            continue
        timestamp_hr = int(directory)
        for dir in os.listdir(path + "/" + directory):
            min_sec = int(str(dir).split('_')[0])
            try: 
                timestamp_minsec_array.index(min_sec)
            except:
                timestamp_minsec_array.append(min_sec)
                timestamp_array.append(min_sec + timestamp_hr * 60000)
    timestamp_array.sort()

    pcd_combined = o3d.geometry.PointCloud()
    point_cloud_data = np.array([0,0,0])
    for timestamp in timestamp_array:
        point_cloud_data = np.load(getImageName(timestamp,"P", session_ts) + ".npy")
        pc.new_point_cloud_data(point_cloud_data)
        
    """point_cloud_data = point_cloud_data.reshape(-1,3)
    point_cloud_data[:,2] = -point_cloud_data[:,2]
    pcd_simplified = point_cloud_post_process(point_cloud_data)
    o3d.io.write_point_cloud("data_long/Session_" + str(session_ts) + "/PointCloud.ply", pcd_simplified)
    ball_size = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    pcd_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_simplified, o3d.utility.DoubleVector(ball_size))
    #mesh = read_mesh("assets/human_representation.gltf")
    """
    o3d.visualization.draw_geometries([pc.O3DPointCloud], mesh_show_back_face = True, mesh_show_wireframe = True)

def read_mesh(path):
    return o3d.io.read_triangle_mesh(path)
        

reconstruct_point_cloud_session("data_long/Session_"+str(session_ts))
#simplify_point_cloud(timestamp_nu)
