import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image 
from scipy.spatial.transform import Rotation as R

dirList = sorted(os.listdir("data_long"))
depthImage = None
abImage = None
fileName = ""
data_folder = "data_long"

"""for file in dirList:

    if file.endswith("tiff"):
        image = cv2.imread("data/"+file)
        hist, bins = np.histogram(image.flatten(),256,[0,256])
        #cumulative sum of histogram
        cdf = hist.cumsum()
        #Remove everything equal to 0
        cdf_m = np.ma.masked_equal(cdf,0)
        
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

        cdf = np.ma.filled(cdf_m,0).astype('uint8')

        processedImg = cdf[image]
        if file.endswith("Abimage.tiff"):
            cv2.imshow("AbImage", processedImg)
            fileName = "data/"+file+"_processed.png"
            cv2.imwrite(fileName,processedImg)
            
        else:
            cv2.imshow("DepthMap", processedImg)
            cv2.imwrite("data/"+file+"_processed.png",processedImg)

    if file.endswith("npy"):
        print(file)
        loadPointCloud("data/"+file)"""

def loadDepthImage(timestamp, ab=True): 
    if (ab):
        imageName = getImageName(timestamp, "A")
    else:
        imageName = getImageName(timestamp, "D")
    image = cv2.imread(imageName)

def getDepth(timestamp, x, y):
    data = np.load(getImageName(timestamp,"D")+".npy")
    return data[x][y]

def getPointCloudCoordinate(timestamp, x, y):
    data = np.load(getImageName(timestamp,"P") + ".npy")
    return data[y*288+x]

def loadDepthData(timestamp):
    data = np.load(getImageName(timestamp,"D") + ".npy")
    #color = cv2.imread(getImageName(timestamp, "A") + ".npy")[:,:,1].reshape(-1,1)
    x = np.arange(0,320,1) * -1
    y = np.arange(0,288,1)
    x,y = np.meshgrid(x,y)
    z = data * -1
    fig = plt.figure()
    #abImage = read_png(fileName)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    ax.view_init(elev=140, azim = 90, roll = -30)
    #ax.plot_surface(xim, yim, zim, rstride = 50, cstride = 50, facecolors = abImage)
    color_map = plt.get_cmap('plasma')
    # Plot the point cloud data
    scatter = ax.scatter(x, y, z, s=1, c=z, cmap=color_map)
    plt.colorbar(scatter)

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

def loadPointCloud(timestamp, x_c=0, y_c=0, z_c=0):
    file = getImageName(timestamp,"P") + ".npy"
    data = np.load(file)
    x_c = eulerAngleConversion(x_c)
    y_c = eulerAngleConversion(y_c)
    z_c = eulerAngleConversion(z_c)
    print(x_c)
    print(y_c)
    print(z_c)

    r = R.from_rotvec([x_c,y_c,z_c], degrees=True)
    data = np.matmul(data,r.as_matrix())

    x = data[:, 0] * -1
    y = data[:, 1]
    z = data[:, 2]


    fig = plt.figure()
    #abImage = read_png(fileName)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    ax.view_init(elev=-90, azim = 90, roll = 0)
    #ax.plot_surface(xim, yim, zim, rstride = 50, cstride = 50, facecolors = abImage)
    color_map = plt.get_cmap('spring')
    # Plot the point cloud data
    scatter = ax.scatter(x, y, z, s=1, c=z, cmap=color_map)
    plt.colorbar(scatter)

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

def getUserCoordinate(timestamp):
    return np.load((getImageName(timestamp,"C")) + ".npy")

#Utilities Functions

def eulerAngleConversion(angle):
    angle %= 360  
    if (angle > 180):
        return angle - 360
    else:
        return angle

def getImageName(timestamp, type):
    imageName = data_folder + "/" + str(int(timestamp/60000)) + "/" + str(timestamp % 60000)
    if type == 'A':
        imageName += "_Abimage.sci"
    elif type == 'C':
        imageName += "_Coordinate_Data.sci"
    elif type == 'D':
        imageName += "_Depth_Map.sci"
    elif type == 'P':
        imageName += "_Point_Cloud_Data.sci"
    return imageName

"""cloud = pv.PolyData(data)
mesh = cloud.delaunay_2d()
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='white')
plotter.show_axes()
plotter.show()

# Save the mesh to a file
#mesh.save('mesh.vtk')

# Visualize the mesh with matplotlib
pv.plot(mesh)
plt.show()
print(data.shape)"""
