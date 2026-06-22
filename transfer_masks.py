# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:55:22 2023

@author: estumpe
"""

import os
import glob
import open3d as o3d
import imageio
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from collections import defaultdict # https://stackoverflow.com/questions/23297569/python-key-error-0-cant-find-dict-error-in-code
from sklearn.neighbors import NearestNeighbors

def _pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int32)
    y = np.linspace(0, height - 1, height).astype(np.int32)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

def project_depth_to_pcl(depth_img,trafo):
    
    depth_indices       = _pixel_coord_np(depth_img.shape[1], depth_img.shape[0]).T
    depth_values        = depth_img[depth_indices[:,1],depth_indices[:,0]]
       
    depth_uv_und        = depth_indices.copy()
    depth_uv_und[:,0:2] = cv2.undistortPointsIter(np.expand_dims(depth_uv_und[:,0:2].astype(np.float64),0),trafo["mtxL"],trafo["distL"],R = None,P = trafo["mtxL"],criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
    
    point_vector        = np.dot(np.linalg.inv(trafo["mtxL"]),depth_uv_und.T).T
    pcl                 = point_vector*np.expand_dims(depth_values,-1)
        
    return pcl,depth_indices


def extract_pcl_ROI(pcl,b_dict):
    
    x_bound = (np.logical_and(pcl[:,0]>=b_dict["xmin"], pcl[:,0]<=b_dict["xmax"])) # get all points which are within the bounding box xmin and xmax
    y_bound = (np.logical_and(pcl[:,1]>=b_dict["ymin"], pcl[:,1]<=b_dict["ymax"])) # get all points which are within the bounding box ymin and ymax
    z_bound = (np.logical_and(pcl[:,2]>=b_dict["zmin"], pcl[:,2]<=b_dict["zmax"])) # in the pointcloud array get all points which are within the specified width d --> list with True for these points

    res_bound = z_bound*x_bound*y_bound
    
    pcl[~res_bound]= np.array([0,0,0])
    
    return pcl


def pcl_to_mesh(pcl, h, w, minAngle=15):
    
    cam_coords      = pcl.T
    
    indices = o3d.utility.Vector3iVector()

    for i in range(0, h-1):
        for j in range(0, w-1):
            verts = [
                cam_coords[:, w*i+j],
                cam_coords[:, w*(i+1)+j],
                cam_coords[:, w*i+(j+1)],
            ]
            if [0,0,0] in map(list, verts):
                pass
            else:
                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)])
        
            verts = [
                cam_coords[:, w*i+(j+1)],
                cam_coords[:, w*(i+1)+j],
                cam_coords[:, w*(i+1)+(j+1)],
            ]
            if [0,0,0] in map(list, verts):

                pass
            else:
                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    indices.append([w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)])
                
    points = o3d.utility.Vector3dVector(cam_coords.transpose())

    mesh = o3d.geometry.TriangleMesh(points, indices)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()   

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    
    return mesh,scene


def compute_frustum_mesh(pcl,boundary_edges_np,b_dict):

    # create dictionary with leaf and ground border coordinates

    pi_dict         = {}
    points_list     = []
    vert_list = o3d.utility.Vector3iVector()
    
    for i in range(boundary_edges_np.shape[0]):
        for j in range(2):
            ind = boundary_edges_np[i,j]
            if(str(ind) in pi_dict):
                pass
            else:
                coords              = pcl[ind]
                ground_coords       = (coords/coords[2])*b_dict["zmax"]
                curr_index          = len(points_list) 
                
                pi_dict[str(ind)]                   = {}
                pi_dict[str(ind)]["leaf"]           = coords
                pi_dict[str(ind)]["ground"]         = ground_coords
                pi_dict[str(ind)]["leaf_index"]     = curr_index
                pi_dict[str(ind)]["ground_index"]   = curr_index+1
                
                points_list.append(coords)
                points_list.append(ground_coords)
                
        # create vert indices
        vertsL = [
            pi_dict[str(boundary_edges_np[i,0])]["leaf_index"],
            pi_dict[str(boundary_edges_np[i,1])]["leaf_index"],
            pi_dict[str(boundary_edges_np[i,0])]["ground_index"]
        ]
        
        
        vertsR = [
            pi_dict[str(boundary_edges_np[i,0])]["ground_index"],
            pi_dict[str(boundary_edges_np[i,1])]["ground_index"],
            pi_dict[str(boundary_edges_np[i,1])]["leaf_index"]
        ]
        
        vert_list.append(vertsL)
        vert_list.append(vertsR)
       
    points_arr = np.asarray(points_list)     
            
    points = o3d.utility.Vector3dVector(points_arr)
    
    mesh = o3d.geometry.TriangleMesh(points, vert_list)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()              
    mesh.paint_uniform_color([1, 0.706, 0])   
     
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    return mesh,scene,pi_dict 

def load_calibration_params(file,basis=False,upscale=1):
    
    trafo = {}
    # load calibration parameters
    trafo["Rot"]    = np.load(os.path.join(file,"Rot"+'.npy'))
    trafo["Trns"]   = np.load(os.path.join(file,"Trns"+'.npy'))
    trafo["mtxL"]   = np.load(os.path.join(file,"mtxL"+'.npy'))
    trafo["mtxR"]   = np.load(os.path.join(file,"mtxR"+'.npy'))
    trafo["distL"]  = np.load(os.path.join(file,"distL"+'.npy'))
    trafo["distR"]  = np.load(os.path.join(file,"distR"+'.npy'))
    trafo["scale"]  = np.load(os.path.join(file,"Scale"+'.npy'))

    # Transformation Matrix
    transfo_mat = np.zeros((4,4))
    transfo_mat[0:3,0:3] = trafo["Rot"]
    transfo_mat[0:3,3] = -trafo["Trns"][:,0]
    transfo_mat[3,3] = 1
    
    trafo["transfo_mat"]  = transfo_mat
    
    global scale
    
    trafo["scale"]  = scale   
    
    trafo["transfo_mat"][0:3,3] = trafo["transfo_mat"][0:3,3]*-1
      
    if basis:
        trafo["transfo_mat"] = np.eye(4,4)
        trafo["mtxR"]   = trafo["mtxL"].copy()
        trafo["distR"]  = trafo["distL"].copy()
    
    if upscale != 1:
        
        trafo["mtxR"]*= upscale 
        # value in lower right corner of Rot matrix has to stay at 1
        trafo["mtxR"][2,2] = 1
        
    return trafo


def add_img_modality(modality_config,img_type,img_format,img_dir,trafo_dir,up_factor = 1,base_modality = False):

    m_dict = {}
    
    m_dict["img_type"]  = img_type
    m_dict["img_files"] = sorted(glob.glob(os.path.join(img_dir,"*"+img_format)))
    m_dict["trafo"]     = load_calibration_params(trafo_dir,basis = base_modality, upscale = up_factor)
    m_dict["up_factor"] = up_factor
    m_dict["basenames"] = [os.path.basename(f).split(".")[0] for f in m_dict["img_files"]]
    
    modality_config[img_type] = m_dict
    
    print(f"added {img_type} modality to settings")
    
    return modality_config
    
def create_depth_modality(img_type,img_format,img_dir,trafo_dir):
    
    d_dict = {}
    
    d_dict["img_type"]  = img_type
    d_dict["img_files"] = sorted(glob.glob(os.path.join(img_dir,"*"+img_format)))
    d_dict["trafo"]     = load_calibration_params(trafo_dir)
    d_dict["basenames"] = [os.path.basename(f).split(".")[0] for f in d_dict["img_files"]]
    
    print(f"added {img_type} modality to settings")
    
    return d_dict    
 

def generate_rays(trafo,uv):

    ray_vector = np.dot(np.linalg.inv(trafo["mtxR"]),uv.T).T
    norm       = np.expand_dims(np.linalg.norm(ray_vector,axis = 1),axis=-1)
    ray_vector = ray_vector/norm
    ray_vector = np.c_[ray_vector,np.ones(ray_vector.shape[0])]
    
    origin          = np.c_[np.zeros((ray_vector.shape[0],3)),np.ones(ray_vector.shape[0])]
    
    origin_k0       = np.dot(np.linalg.inv(trafo["transfo_mat"]),origin.T).T
    origin_k0       = np.add(origin_k0,0.000001) # add something for slight offset to prevent rays from casting into mesh nodes
    
    ray_vector_k0   = np.dot(np.linalg.inv(trafo["transfo_mat"]),ray_vector.T).T
    ray_vector_k0   = ray_vector_k0 - origin_k0
    
    rays            = np.c_[origin_k0[:,0:3],ray_vector_k0[:,0:3]].astype(np.float32)
    
    return rays,ray_vector

def compute_frustum_mask(scene_hit,u_scene_hit):
   
    u_diff  =  u_scene_hit - scene_hit 
    
    sc_inf = (scene_hit!=np.inf)
    u_inf  = (u_scene_hit!=np.inf)
    closer = (np.sign(u_diff)==-1)
    
    unc_ray_mask    = (sc_inf*u_inf*closer)
    obj_mesh_mask   = sc_inf
    unc_mesh_mask   = u_inf*~obj_mesh_mask
    
    return unc_ray_mask,obj_mesh_mask,unc_mesh_mask
    


def compute_3D_points_ks0(ray_vector,trafo,scene_hit,scene_indices):
    
    xyz_ksx         = ray_vector.copy()
    xyz_ksx[:,0:3]  = np.multiply(ray_vector[:,0:3],np.expand_dims(scene_hit,-1)) # scaled to correct size
    
    xyz_ks0         = np.dot(np.linalg.inv(trafo["transfo_mat"]),xyz_ksx.T).T
    xyz_ks0         = xyz_ks0[scene_indices]
    
    return xyz_ks0

def ray_backcast(xyz_ks0,trafo,scene,u_scene):
        
    origin      = np.c_[np.zeros((xyz_ks0.shape[0],3)),np.ones(xyz_ks0.shape[0])]
    origin_k0   = np.dot(np.linalg.inv(trafo["transfo_mat"]),origin.T).T  
        
    rays        = np.c_[origin_k0[:,0:3], xyz_ks0[:,0:3]-origin_k0[:,0:3]].astype(np.float32)   
      
    scene_hit   = (scene.cast_rays(rays))["t_hit"].numpy()
    u_scene_hit = (u_scene.cast_rays(rays))["t_hit"].numpy()
        
    return scene_hit,u_scene_hit

def compute_img_coords(xyz_ks0,img,trafo,uv_target):
     
    xyz_ks_source = np.dot(trafo["transfo_mat"],xyz_ks0.T).T
    
    uv_source   = np.round(np.dot(trafo["mtxR"],xyz_ks_source.T[0:3,:]/xyz_ks_source.T[2,:]).T).astype(np.int32)
    
    h_bound     = (np.logical_and(uv_source[:,1]>=0,
                              uv_source[:,1]<img.shape[0]))
                
    w_bound     = (np.logical_and(uv_source[:,0]>=0,
                              uv_source[:,0]<img.shape[1]))
    
    b_mask          = h_bound*w_bound
    b_mask_indices  = np.argwhere(b_mask)[:,0]
    
    uv_source_out   = uv_source[b_mask_indices]
    xyz_out         = xyz_ks0[b_mask_indices]
    uv_target_out   = uv_target[b_mask_indices]
    
    return uv_target_out,xyz_out,uv_source_out


def target_3D_registration(target,trafo,target_uv,scene,u_scene=None):
    
    reg_in = {}
    rays,target_ray_vector = generate_rays(trafo,target_uv)
    
    target_scene_hit    = (scene.cast_rays(rays))["t_hit"].numpy()
    
    if u_scene != None:
        
        target_u_scene_hit  = (u_scene.cast_rays(rays))["t_hit"].numpy() 
      
        unc_ray_mask,obj_mesh_mask,unc_mesh_mask = compute_frustum_mask(target_scene_hit,target_u_scene_hit)

        reg_in["unc_ray_mask"]  = target_uv[unc_ray_mask]
        reg_in["obj_mesh_mask"] = target_uv[obj_mesh_mask]
        reg_in["unc_mesh_mask"] = target_uv[unc_mesh_mask]    
    
    target_scene_indices  = np.argwhere(target_scene_hit!=np.inf)[:,0]
    
    target_xyz_ks0  = compute_3D_points_ks0(target_ray_vector,modality_config[target]["trafo"],target_scene_hit,target_scene_indices)
     
    reg_in["uv_target"]     = target_uv[target_scene_indices]
    reg_in["xyz"]           = target_xyz_ks0
    
    return reg_in



def target_source_registration(target,source,modalities,modality_config,scene,u_scene):
    source_scene_hit,source_u_scene_hit = ray_backcast(reg_in["xyz"],modality_config[source]["trafo"],scene,u_scene)
    
    uv_target_out,xyz_out,uv_source_out = compute_img_coords(reg_in["xyz"],modalities[source]["img"],modality_config[source]["trafo"],reg_in["uv_target"])
        
    reg_out = {}
    reg_out["uv_target"] = uv_target_out
    reg_out["xyz"]       = xyz_out
    reg_out["uv_source"] = uv_source_out
    
    return reg_out


def sort_coordinates(centroid_arr):

    sort_indices = np.argsort(centroid_arr[:,1])
    
    sorted_centroids = centroid_arr[sort_indices]
    
    x           = sorted_centroids[:,1]
    x_diff      = x - np.roll(x,1)
    x_diff[0]   = 0 # first index is invalid (gets value from bottom of vector) 
        
    x_indices   = np.argsort(x_diff)[-7:]     
    x_indices   = np.sort(x_indices)    
    
    groups = []
    
    for i in range(x_indices.shape[0]+1):
        
        if(i == 0):
            groups.append(sorted_centroids[0:x_indices[i]])
    
        elif(i < x_indices.shape[0]):
            groups.append(sorted_centroids[x_indices[i-1]:x_indices[i]])
        
        else:
            groups.append(sorted_centroids[x_indices[i-1]:])
            
    y_sort_groups = []
        
    for g in groups:
        
        sort_indices = np.argsort(g[:,0])
        
        y_sort_groups.append(g[sort_indices])
        
    accum_arr = np.concatenate(y_sort_groups, axis = 0)
    
    return accum_arr


def xgap_coordinates(accum_arr,order_list):
    
    new_coordinates = []
    
    c = 0
    
    for i,o in enumerate(order_list):
        o05 = int(o/2)
        
        for j in range(o05-1):
            new_coord = (accum_arr[c] + accum_arr[c+1])/2 
            new_coordinates.append(new_coord)
            
            c+=1
        
        c+=1    
        
        for j in range(o05-1):
            new_coord = (accum_arr[c] + accum_arr[c+1])/2 
            new_coordinates.append(new_coord)
            
            c+=1
        
        c+=1
         
        new_coordinate_arr = np.asarray(new_coordinates)
        
    return new_coordinate_arr


def ygap_coordinates(accum_arr,num,forward,skip):
     
    new_coordinates2 = []
    
    
    c = 0
    
    for i in range(len(num)):
        
        for j in range(num[i]):
                        
            new_coord = (accum_arr[c] + accum_arr[c+forward[i]])/2 
            new_coordinates2.append(new_coord)
            
            c+=1
        
        for k in range(skip[i]):
            
            c+= 1
      
    new_coordinate_arr = np.asarray(new_coordinates2)  
      
    return new_coordinate_arr


def correct_via_homography(accum_arr):
    
    ideal_coords = np.array([
                         [0,0], #1st row
                         [0,2],
                         [0,4],
                         [0,6],
                         
                         [0,11],
                         [0,13],
                         [0,15],
                         [0,17],
                         
                         [2,0], #2nd row
                         [2,2],
                         [2,4],
                         [2,6],
                         
                         [2,11],
                         [2,13],
                         [2,15],
                         [2,17],
                         
                         [4,0], #3rd row
                         [4,2],

                         [4,15],
                         [4,17], 
                         
                         [6,0], #4th row
                         [6,2],

                         [6,15],
                         [6,17],    
                         
                         [11,0], #5th row
                         [11,2],

                         [11,15],
                         [11,17],  
                         
                         [13,0], #6th row
                         [13,2],

                         [13,15],
                         [13,17],  
                         
                         [15,0], #7th row
                         [15,2],
                         [15,4],
                         [15,6],
                         
                         [15,11],
                         [15,13],
                         [15,15],
                         [15,17],       
                         
                         [17,0], #8th row
                         [17,2],
                         [17,4],
                         [17,6],
                         
                         [17,11], 
                         [17,13],
                         [17,15],
                         [17,17],                            
                                                  
                         ]) 
    
    
    # Create an empty array (image) with the shape (18, 18)
    plot_arr = np.zeros((18, 18))
    
    # Set the positions in plot_arr to 1 where coordinates are given by ideal_coords
    for coord in ideal_coords:
        y, x = coord  # Get y and x components from the coordinate
        plot_arr[y, x] = 1  # Set the value at that position to 1
    
    # Plot the image
    # plt.figure(figsize=(6, 6))  # Set figure size for better visibility
    # plt.imshow(plot_arr, cmap='gray', interpolation='none')  # Use grayscale for binary image
    # plt.title("Geometric Pattern Visualization")
    # plt.xlabel("X (Horizontal)")
    # plt.ylabel("Y (Vertical)")
    # plt.grid(False)  # Turn off the grid to focus on the pattern
    # plt.show()
    
    yx_accum_arr = np.flip(accum_arr,axis = 1).astype(np.float32)
    print(yx_accum_arr.shape)
    
    ideal_coords = ideal_coords.astype(np.float32)
    print(ideal_coords.shape)
    
    
    # Calculate the reverse perspective transformation matrix using cv2.findHomography
    # This time, we find the matrix that maps ideal_coords to yx_accum_arr
    h_reverse, status = cv2.findHomography(ideal_coords, yx_accum_arr, cv2.RANSAC)
    
    # Transform the ideal points to the detected points' coordinate system using the reverse matrix
    projected_points = cv2.perspectiveTransform(ideal_coords.reshape(-1, 1, 2), h_reverse).reshape(-1, 2)
    
    # # Plot the original ideal points and the projected points for comparison
    # plt.figure(figsize=(10, 5))
    
    # # Ideal points before transformation
    # plt.subplot(1, 2, 1)
    # plt.scatter(ideal_coords[:, 1], ideal_coords[:, 0], color='red', label='Ideal Points')
    # plt.scatter(yx_accum_arr[:, 1], yx_accum_arr[:, 0], color='blue', marker='x', label='Detected Points')
    # plt.title('Original Points')
    # plt.xlabel('X (Horizontal)')
    # plt.ylabel('Y (Vertical)')
    # plt.legend()
    # plt.gca().invert_yaxis()  # Invert y-axis for image coordinates
    
    # # Ideal points after projection (mapped to detected points)
    # plt.subplot(1, 2, 2)
    # plt.scatter(projected_points[:, 1], projected_points[:, 0], color='green', label='Projected Ideal Points')
    # plt.scatter(yx_accum_arr[:, 1], yx_accum_arr[:, 0], color='blue', marker='x', label='Detected Points')
    # plt.title('Ideal Points Projected onto Detected Points')
    # plt.xlabel('X (Horizontal)')
    # plt.ylabel('Y (Vertical)')
    # plt.legend()
    # plt.gca().invert_yaxis()  # Invert y-axis for image coordinates
    
    # plt.tight_layout()
    # plt.show()
    
    # Output the reverse transformation matrix
    print("Reverse Perspective Transformation Matrix (h_reverse):\n", h_reverse)
    
    return np.flip(projected_points,axis = 1)

def opposite_lut(gap_coords,gap_list):
    
    h           = len(gap_list)
    w           = np.max(np.array(gap_list))
    
    address_arr     = np.ones((h,w))*np.nan 
    coord_address   = np.zeros(gap_coords.shape).astype(np.uint8)
    
    c = 0
    for y,g in enumerate(gap_list):
        
        rest = w - g
        
        g05 = int(g/2)
        
        for x in range(g05):
            
            address_arr[y,x] = c
            coord_address[c] = np.array([y,x])
            
            c+=1
            
        for x in range(g05+rest,g+rest):
            
            address_arr[y,x] = c
            coord_address[c] = np.array([y,x])
            c+=1
    
    gap_lut = np.zeros(gap_coords.shape[0])
    
    for i in range(gap_lut.shape[0]):
        
        address     = coord_address[i]
        gap_lut[i] = int(address_arr[address_arr.shape[0]-1-address[0],address[1]])
        
    gap_lut = gap_lut.astype(np.uint8)
        
    return gap_lut




def extract_3D_points(depth,ROOT_DIR,calibration,gap_coords,trafo):
        
    uv          = np.ones((gap_coords.shape[0],3))
    uv[:,0:2]   = gap_coords
    
    # compute ray vector
    # ray_vector  = np.dot(np.linalg.inv(trafo["mtxL"]),uv.T).T 
    ray_vector  = np.dot(np.linalg.inv(trafo["mtxR"]),uv.T).T  #double check
    norm        = np.expand_dims(np.linalg.norm(ray_vector,axis = 1),axis=-1)
    ray_vector  = ray_vector/norm
    ray_vector  = np.c_[ray_vector,np.ones(ray_vector.shape[0])]
    
    # compute ray intersection
    origin          = np.c_[np.zeros((ray_vector.shape[0],3)),np.ones(ray_vector.shape[0])]
    origin_k0       = np.dot(np.linalg.inv(np.eye(4,4)),origin.T).T
    origin_k0       = np.add(origin_k0,0.000001) # add something for slight offset to prevent rays from casting into mesh nodes
    
    ray_vector_k0   = np.dot(np.linalg.inv(np.eye(4,4)),ray_vector.T).T
    ray_vector_k0   = ray_vector_k0 - origin_k0
    
    rays            = np.c_[origin_k0[:,0:3],ray_vector_k0[:,0:3]].astype(np.float32)
        
    scene_hit       = (depth["scene"].cast_rays(rays))["t_hit"].numpy()
       
    scene_indices   = np.argwhere(scene_hit!=np.inf)[:,0]
    
    xyz_ksx         = ray_vector
    xyz_ksx[:,0:3]  = np.multiply(ray_vector[:,0:3],np.expand_dims(scene_hit,-1)) # scaled to correct size
    
    xyz_ks0         = np.dot(np.linalg.inv(np.eye(4,4)),xyz_ksx.T).T
    xyz_ks0         = xyz_ks0[scene_indices]
    
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_ks0[:,0:3])
    
    return xyz_ks0,depth


def compute_optimal_rigid_transformation(source_points, target_points):
    # Ensure input arrays have the correct shape
    assert source_points.shape == target_points.shape, "Source and target must have the same shape"
    assert source_points.shape[1] == 3, "Each point must have three coordinates"

    # Compute centroids of both point clouds
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Compute the covariance matrix
    covariance_matrix = np.dot(centered_target.T, centered_source)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Compute the optimal rotation matrix
    R = np.dot(U, Vt)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    # Compute the optimal translation
    t = centroid_target - np.dot(R, centroid_source)

    # Construct the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix

def compute_plane_normal(points):
    """
    Computes the normal vector of a best-fit plane for a set of 3D points using SVD.
    
    Args:
        points (np.ndarray): Array of shape (N, 3) representing N 3D points.
    
    Returns:
        np.ndarray: The normal vector of the best-fit plane.
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Center the points around the centroid
    centered_points = points - centroid
    
    # Perform Singular Value Decomposition (SVD) on the centered points
    _, _, vh = np.linalg.svd(centered_points)
    
    # The normal vector is the last row of vh (corresponding to the smallest singular value)
    normal_vector = vh[-1]
    
    return normal_vector

def compute_3D_points(img_uv,trafo,depth):
     
    # compute ray vector
    # ray_vector  = np.dot(np.linalg.inv(trafo["mtxL"]),img_uv).T 
    ray_vector  = np.dot(np.linalg.inv(trafo["mtxR"]),img_uv).T 
    norm        = np.expand_dims(np.linalg.norm(ray_vector,axis = 1),axis=-1)
    ray_vector  = ray_vector/norm
    ray_vector  = np.c_[ray_vector,np.ones(ray_vector.shape[0])]
    
    # compute ray intersection
    origin          = np.c_[np.zeros((ray_vector.shape[0],3)),np.ones(ray_vector.shape[0])]
    origin_k0       = np.dot(np.linalg.inv(np.eye(4,4)),origin.T).T
    origin_k0       = np.add(origin_k0,0.000001) # add something for slight offset to prevent rays from casting into mesh nodes
    
    ray_vector_k0   = np.dot(np.linalg.inv(np.eye(4,4)),ray_vector.T).T
    ray_vector_k0   = ray_vector_k0 - origin_k0
    
    rays            = np.c_[origin_k0[:,0:3],ray_vector_k0[:,0:3]].astype(np.float32)
        
    scene_hit       = (depth["scene"].cast_rays(rays))["t_hit"].numpy()
       
    scene_indices   = np.argwhere(scene_hit!=np.inf)[:,0]
    
    xyz_ksx         = ray_vector
    xyz_ksx[:,0:3]  = np.multiply(ray_vector[:,0:3],np.expand_dims(scene_hit,-1)) # scaled to correct size
    
    xyz_ks0         = np.dot(np.linalg.inv(np.eye(4,4)),xyz_ksx.T).T
    xyz_ks0         = xyz_ks0[scene_indices]
    
    return xyz_ks0,scene_indices



def load_images(modality_config,index):
    
    modalities = {}
    
    for k in modality_config.keys():
        
        m = {}
        
        print(k)
        print(index)
        
        file        = modality_config[k]["img_files"][index]
        basename    = os.path.basename(file)
        
        if (basename.endswith(".png")):
            img = imageio.imread(file)
        elif(basename.endswith(".npy")):
            img = np.load(file)
         
        if modality_config[k]["up_factor"] !=1:
            
            m["img_orig"] = img.copy()
  
            h           = int(img.shape[0]*modality_config[k]["up_factor"])
            w           = int(img.shape[1]*modality_config[k]["up_factor"])
    
            img         = cv2.resize(img,(w,h)) 
        # undistort image    
        img_distorted         = cv2.undistort(img, modality_config[k]["trafo"]["mtxR"], modality_config[k]["trafo"]["distR"])
            
        m["img"]    = img_distorted
        m["img_raw"]= img
        m["name"]   = basename
        
        modalities[k] = m
    
    return modalities  
    

def knn_outliers(points, k=5, th=2):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)  # k+1 because it includes the point itself
    distances, _ = nbrs.kneighbors(points)
    median_dist  = np.median(distances,axis=0)[1:3] # take the 2 first neighbors ()
    
    error = np.sum(np.abs(distances[:,1:3] - median_dist),axis=1)
    sorted_errors = np.argsort(error)
    error_values  = error[sorted_errors[-th:]]
    next_error    = error[sorted_errors[-(th+1)]]
    print(f"error values are {error_values}")
    print(f"next largest error is {next_error}")
    
    return sorted_errors[-th:]



def compute_marker_coords(img,trafo,upscale):
    
    img_normed = img.copy()
    img_normed[img_normed>17000] = 17000
        
    img_gray = img_normed.copy()
    img_gray = ((img_normed - np.min(img_normed))/(np.max(img_normed)-np.min(img_normed)))*255
    img_gray = img_gray.astype(np.uint8)
        
    # mean adaptive thresholding
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    
    thresh = (((thresh/255)*-1)+1).astype(np.uint8)

    # Define a kernel for the morphological operation
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply erosion
    eroded_image = cv2.erode(thresh, kernel, iterations=1)
    
    # Apply dilation
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    
    thresh = dilated_image.copy()
        
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh)
            
    smin = 8
    smax = 18 # von 16
    ratio_thresh = 0.65 # verändert von 0.6
     
    new_label_im = np.zeros(thresh.shape)
    
    valid_indices = []
    centroid_list = []
    indices_list  = []
    ratio         = []
    
    c = 1
    for i in range(num_labels):
        
        # check for different characteristics
        min_x_bound = stats[i][2] > smin
        max_x_bound = stats[i][2] < smax
        min_y_bound = stats[i][3] > smin
        max_y_bound = stats[i][3] < smax
        
        long_side   = np.max((stats[i][2],stats[i][3]))
        area_ratio  = stats[i][4]/(long_side*long_side)
        
        ratio_bound = area_ratio>ratio_thresh
        
        
        if min_x_bound*max_x_bound*min_y_bound*max_y_bound*ratio_bound:
            centroid_list.append(centroids[i])
            new_label_im[labels_im==i] = c
            indices_list.append(np.argwhere(labels_im==i))
            ratio.append(area_ratio)
            valid_indices.append(i)
            
            c+=1
        
    valid_indices = np.array(valid_indices)
                   
    centroid_arr = np.asarray(centroid_list)
    
    centroid_arr*= upscale
    
    c_undistort = cv2.undistortPointsIter(np.expand_dims(centroid_arr.astype(np.float64),0),trafo["mtxR"],trafo["distR"],R = None,P = trafo["mtxR"],criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
    
    points      =  np.flip(c_undistort,axis = 1).astype(np.float32)
    
    # plt.figure()
    # plt.scatter(points[:, 1], points[:, 0], color='red', label='Ideal Points')
    
    if(points.shape[0]>48):
        outlier_indices = knn_outliers(points, k=5, th=points.shape[0]-48)
        centroid_arr = np.delete(centroid_arr, outlier_indices, axis=0)
        
        valid_indices = np.delete(valid_indices, outlier_indices)
             
    elif(points.shape[0]<48):
        print(f"found to few segments: {points.shape[0]}")
      
    # get all coordinates in order
    accum_arr = sort_coordinates(centroid_arr)
    
    # undistort all coords
    accum_arr_undistort = cv2.undistortPointsIter(np.expand_dims(accum_arr.astype(np.float64),0),trafo["mtxR"],trafo["distR"],R = None,P = trafo["mtxR"],criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
     
    # correct the coordinates via homography of ideal coordinate position
    accum_arr_corrected = correct_via_homography(accum_arr_undistort)
    
    # compute points between the squares
    x_gap_coords = xgap_coordinates(accum_arr_corrected,[8,8,4,4,4,4,8,8])
    
    num = [8,2,6,6,10] # num
    forward = [8,8,4,4,8] # forward
    skip = [0,4,4,0,0] # skip
    
    y_gap_coords = ygap_coordinates(accum_arr_corrected,num,forward,skip)
    
    # create look up tables to get access to the opposite position
    x_gap_lut    = opposite_lut(x_gap_coords,[6,6,2,2,2,2,6,6])
    y_gap_lut    = opposite_lut(y_gap_coords,[8,4,4,4,4,8]) 
    
    # combine the coordinates    
    gap_coords  = np.concatenate([x_gap_coords,y_gap_coords],axis = 0)
    gap_lut     = np.concatenate([x_gap_lut,y_gap_lut + x_gap_lut.shape[0]],axis = 0)
    
    return gap_coords,gap_lut    
    

def transform_coordinates(c_in,c_out,c_out_t):
    
    c_in_set = {tuple(row) for row in c_in}
    
    # Check if each row in c_out is in c_in_set
    matches = np.array([1 if tuple(row) in c_in_set else 0 for row in c_out])
    
    # Apply found matches to c_out_t 
    out  = c_out_t[matches==1]
    
    print(f"a total of {np.count_nonzero(matches)} matches was found")
    
    return out


def check_consistency(depth_config,modality_config):

    reference_modality  = "depth" 
    reference_basenames = [
        os.path.basename(f).split(".")[0]
        for f in depth_config["img_files"]
    ]   

    for i, m in enumerate(modality_config):
    
        basenames = [
            os.path.basename(f).split(".")[0]
            for f in modality_config[m]["img_files"]
        ]
        
        missing = set(reference_basenames) - set(basenames)
        extra = set(basenames) - set(reference_basenames)
    
        order_mismatches = [
            (idx, ref_name, current_name)
            for idx, (ref_name, current_name) in enumerate(
                zip(reference_basenames, basenames)
            )
            if ref_name != current_name
        ]
    
        if missing or extra or order_mismatches or len(reference_basenames) != len(basenames):
            print(f"Image mismatch between '{reference_modality}' and '{m}'")
    
            if missing:
                print(f"\nMissing in '{m}':")
                for item in sorted(missing):
                    print(f"  - {item}")
    
            if extra:
                print(f"\nExtra in '{m}':")
                for item in sorted(extra):
                    print(f"  - {item}")
    
            if order_mismatches:
                print(f"\nOrder mismatches:")
                for idx, ref_name, current_name in order_mismatches:
                    print(
                        f"  index {idx}: "
                        f"{reference_modality}='{ref_name}' vs {m}='{current_name}'"
                    )
    
            if len(reference_basenames) != len(basenames):
                print(
                    f"\nLength mismatch: "
                    f"{reference_modality} has {len(reference_basenames)} images, "
                    f"{m} has {len(basenames)} images"
                )
    
            raise ValueError(
                f"Image files are dissimilar or out of order between "
                f"'{reference_modality}' and '{m}'"
            )
            
    print("files are consistent")


def check_mask_consistency(mask_basenames,bot_top_lut,basenames):
    
    for m in mask_basenames:
        
        if m not in bot_top_lut:
            raise ValueError(f"mask {m} not found in bot_top_lut")
            
        if m not in basenames:
            raise ValueError(f"mask {m} not found in modality basenames")
            
        top_basename = bot_top_lut[m]
        
        if top_basename not in basenames:
            raise ValueError(f"top reference {top_basename} from mask {m} not found in modality basenames")
            
    print("all masks and filenames consistent")

bot_top_lut = {
    "exp00_setID0000":"exp00_setID0001",
    "exp00_setID0002":"exp00_setID0003",
    "exp00_setID0004":"exp00_setID0005",
    "exp00_setID0006":"exp00_setID0007",
    "exp00_setID0008":"exp00_setID0009",
    "exp01_setID0000":"exp01_setID0001",
    "exp01_setID0002":"exp01_setID0003",
    "exp01_setID0004":"exp01_setID0005",
    "exp01_setID0006":"exp01_setID0007",
    "exp01_setID0008":"exp01_setID0009",
    "exp01_setID0010":"exp01_setID0011",
    "exp01_setID0012":"exp01_setID0013",
    "exp02_setID0000":"exp02_setID0001",
    "exp02_setID0002":"exp02_setID0003",
    "exp02_setID0004":"exp02_setID0005",
    "exp02_setID0006":"exp02_setID0007",
    "exp02_setID0008":"exp02_setID0009",
    "exp02_setID0010":"exp02_setID0011",
    "exp02_setID0012":"exp02_setID0013",
    "exp02_setID0014":"exp02_setID0015",
    "exp02_setID0016":"exp02_setID0017",
    "exp02_setID0018":"exp02_setID0019",
    "exp02_setID0020":"exp02_setID0021",
    "exp02_setID0022":"exp02_setID0023",
    "exp03_setID0018":"exp03_setID0019",
    "exp03_setID0020":"exp03_setID0021",
    "exp03_setID0022":"exp03_setID0023",
    "exp03_setID0024":"exp03_setID0025",
    "exp03_setID0026":"exp03_setID0027",
    "exp03_setID0028":"exp03_setID0029",
    "exp03_setID0030":"exp03_setID0031",
    "exp03_setID0032":"exp03_setID0033",
    "exp03_setID0034":"exp03_setID0035",
    "exp03_setID0036":"exp03_setID0037",
    "exp03_setID0038":"exp03_setID0039",
    "exp03_setID0040":"exp03_setID0041"
    }


ROOT_DIR = os.getcwd()

calibration  = "calibration_params"

scale = 1/(12.9) #1/sidelength of checkerboard square (mm)

modality_config = {}
# add infrared modality
modality_config = add_img_modality(modality_config,
                                   "infrared",
                                   ".npy",
                                   os.path.join(ROOT_DIR,"dataset","infrared"),
                                   os.path.join(ROOT_DIR,"calibration_params","rgbd"),
                                   up_factor = 4,
                                   base_modality = True)

# add RGB modality
modality_config = add_img_modality(modality_config,
                                   "rgb",
                                   ".png",
                                   os.path.join(ROOT_DIR,"dataset","rgb"),
                                   os.path.join(ROOT_DIR,"calibration_params","rgbd"),
                                   up_factor = 1,
                                   base_modality = False)

# add hs modality
modality_config = add_img_modality(modality_config,
                                   "hs",
                                   ".png",
                                   os.path.join(ROOT_DIR,"dataset","hs"),
                                   os.path.join(ROOT_DIR,"calibration_params","hs"),
                                   up_factor = 1,
                                   base_modality = False)

# load depth config
depth_config = create_depth_modality("depth",
                                     ".npy",
                                     os.path.join(ROOT_DIR,"dataset","depth"),
                                     os.path.join(ROOT_DIR,"calibration_params","rgbd"))

mask_files      = sorted(glob.glob(os.path.join(ROOT_DIR,"dataset","hs_mask_annot_bot","*.png")))
mask_basenames  = [os.path.basename(f).split(".")[0] for f in mask_files] 

check_consistency(depth_config,modality_config)
check_mask_consistency(mask_basenames,bot_top_lut,modality_config["infrared"]["basenames"])

target = "infrared"
source = "hs"

# limit xyz coordinates to reduce computation effort
bbox_dict       ={"xmin":-400*scale,
                  "xmax":260*scale,
                  "ymin":-500*scale,
                  "ymax":180*scale,
                  "zmin":200*scale,
                  "zmax":900*scale}

TRF_OUT_DIR = os.path.join(ROOT_DIR,"transformation_results","images")
os.makedirs(TRF_OUT_DIR,exist_ok = True)

MASK_OUT_DIR = os.path.join(ROOT_DIR,"transformation_results","masks")
os.makedirs(MASK_OUT_DIR,exist_ok = True)

for i,bot_basename in enumerate(mask_basenames):
    
    top_basename        = bot_top_lut[bot_basename]
    modality_bot_index  = modality_config[target]["basenames"].index(bot_basename)
    modality_top_index  = modality_config[target]["basenames"].index(top_basename)
        
    modality_list   = []
    depth_list      = []
    
    print(modality_bot_index)
    print(modality_top_index)
    
    for index in (modality_bot_index,modality_top_index):
        
        modalities          = load_images(modality_config,index)
        depth_img           = np.load(depth_config["img_files"][index])
        
        # create point cloud, object mesh and rendering scene
        pcl,depth_indices   = project_depth_to_pcl(depth_img,depth_config["trafo"])
        pcl[:,0:3]          = pcl[:,0:3]*depth_config["trafo"]["scale"]
        pcl                 = extract_pcl_ROI(pcl,bbox_dict)
        mesh,scene          = pcl_to_mesh(pcl,depth_img.shape[0],depth_img.shape[1], minAngle=15)
        
        mesh_edges          = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
        u_mesh,u_scene,pi_dict = compute_frustum_mesh(pcl,mesh_edges,bbox_dict)
        
        target_uv = _pixel_coord_np(modalities[target]["img"].shape[1], modalities[target]["img"].shape[0]).T
        
        reg_in  = target_3D_registration(target,modality_config[target]["trafo"],target_uv,scene,u_scene)
        reg_out = target_source_registration(target,source,modalities,modality_config,scene,u_scene) 
                
        mod = {}
        mod["modalities"]   = modalities
        mod["reg_in"]       = reg_in
        mod["reg_out"]      = reg_out
        
       
        depth_dict = {}
        depth_dict["pcl"] = pcl
        depth_dict["depth_indices"] = depth_indices
        depth_dict["mesh"] = mesh
        depth_dict["scene"] = scene
        
        modality_list.append(mod)
        depth_list.append(depth_dict)
        
    plt.close('all') 
     
    # bottom side registration
    bot_target   = modality_list[0]["modalities"][target]["img"] #img1infra
    bot_source   = np.zeros((bot_target.shape[0],bot_target.shape[1],3)).astype(np.uint8) #img1hs
    t_uv            = modality_list[0]["reg_out"]["uv_target"]
    s_uv            = modality_list[0]["reg_out"]["uv_source"]   
    bot_source[t_uv[:,1],t_uv[:,0]] = modality_list[0]["modalities"][source]["img"][s_uv[:,1],s_uv[:,0]] 

    # top side registration
    top_target      = modality_list[1]["modalities"][target]["img"] #img2infra
    top_source      = np.zeros((top_target.shape[0],top_target.shape[1],3)).astype(np.uint8) #img2hs
    t_uv            = modality_list[1]["reg_out"]["uv_target"]
    s_uv            = modality_list[1]["reg_out"]["uv_source"]  
    top_source[t_uv[:,1],t_uv[:,0]] = modality_list[1]["modalities"][source]["img"][s_uv[:,1],s_uv[:,0]]     
      
    # detect marker coordinates        
    bot_target_orig  = modality_list[0]["modalities"][target]["img_orig"]
    top_target_orig  = modality_list[1]["modalities"][target]["img_orig"]  
    
    trafo = modality_config[target]["trafo"]
    
    bt_gap_coords,bt_gap_lut = compute_marker_coords(bot_target_orig,trafo,modality_config[target]["up_factor"])
    tt_gap_coords,tt_gap_lut = compute_marker_coords(top_target_orig,trafo,modality_config[target]["up_factor"])
         
    # MESH_DIR = os.path.join(ROOT_DIR,"2026_tests",str(modality_bot_index))
    # os.makedirs(MESH_DIR, exist_ok = True)
              
    img1_3D,depth1 = extract_3D_points(depth_list[0],ROOT_DIR,calibration,bt_gap_coords,trafo)
    img2_3D,depth2 = extract_3D_points(depth_list[1],ROOT_DIR,calibration,tt_gap_coords,trafo)
       
    
    img2_3D_opp = np.ones(img2_3D.shape)
    
    for j in range(img2_3D.shape[0]):
        
        img2_3D_opp[j,:] = img2_3D[tt_gap_lut[j],:]
    
    transformation_matrix = compute_optimal_rigid_transformation(img1_3D[:,0:3],img2_3D_opp[:,0:3])
    
    
    img1_3D_trafo = np.dot(transformation_matrix,img1_3D.T).T
    

    square_diff     = np.square(img2_3D_opp[:,0:3]-img1_3D_trafo[:,0:3])
    diff_per_row    = np.sqrt(np.sum(square_diff,axis=1))
    rmse            = np.sqrt(np.sum(np.square(diff_per_row))/diff_per_row.shape[0])
    
    print(f"rmse: {rmse}")
    
    # Example usage
    # points = np.random.rand(64, 3)  # Replace this with your 64x3 array of points
    
    normal = compute_plane_normal(img2_3D_opp[:,0:3])
    print("Normal Vector of the Best-fit Plane:", normal)
    
    #transform the mesh as a sanity check
    
    mesh1 = depth1["mesh"]
    
    mesh1_transformed = mesh1.transform(transformation_matrix)
    
    #o3d.io.write_triangle_mesh(os.path.join(MESH_DIR,"mesh_transformed.ply"), mesh1_transformed)
    
    
    img1_uv = _pixel_coord_np(bot_target.shape[1], bot_target.shape[0])
    
    img1_xyz,img1_indices = compute_3D_points(img1_uv,trafo,depth1)
    
    img1_xyz_ks2 = np.dot(transformation_matrix,img1_xyz.T).T
    
    #sanity check transformed points
    
    #Create an Open3D PointCloud object
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(img1_xyz_ks2[:,0:3])
    #Save the point cloud as an XYZ file
    #o3d.io.write_point_cloud(os.path.join(MESH_DIR,"img_points_transformed.xyz"), point_cloud, write_ascii=True)
    

    normal_vector           = np.ones((1,4))
    normal_vector[:,0:3]    = normal
    normal_vector           = np.repeat(normal_vector, img1_xyz_ks2.shape[0] ,axis=0)    
    
    normal_vector[:,0:3] /= np.linalg.norm(normal_vector[:,0:3], axis=1, keepdims=True)
        
    rays                = np.c_[img1_xyz_ks2[:,0:3],normal_vector[:,0:3]].astype(np.float32)
    
    scene_hit           = (depth2["scene"].cast_rays(rays))["t_hit"].numpy()
    
       
    hit_mask            = scene_hit!=np.inf
    scene_indices_pos   = np.argwhere(scene_hit!=np.inf)[:,0]
    
    xyz_ksx         = normal_vector.copy()
    xyz_ksx[:,0:3]  = np.multiply(normal_vector[:,0:3],np.expand_dims(scene_hit,-1)) # scaled to correct size
    
    xyz_ks0         = xyz_ksx.copy()
    xyz_ks0[:,0:3]  = img1_xyz_ks2[:,0:3] + xyz_ks0[:,0:3]
     
    xyz_ks0         = xyz_ks0[scene_indices_pos]
    
    
    #%%
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_ks0[:,0:3])
    
    rays_reverse    = np.c_[img1_xyz_ks2[:,0:3],-normal_vector[:,0:3]].astype(np.float32)
    
    scene_hit_reverse   = (depth2["scene"].cast_rays(rays_reverse))["t_hit"].numpy()
    
    hit_mask        = scene_hit!=np.inf
    scene_indices_neg   = np.argwhere(scene_hit_reverse!=np.inf)[:,0]
    
    xyz_ksx         = - normal_vector.copy()
    xyz_ksx[:,0:3]  = np.multiply(xyz_ksx[:,0:3],np.expand_dims(scene_hit_reverse,-1)) # scaled to correct size
    
    xyz_ks0         = xyz_ksx.copy()
    xyz_ks0[:,0:3]  = img1_xyz_ks2[:,0:3] + xyz_ks0[:,0:3]
    
    xyz_ks0         = xyz_ks0[scene_indices_neg]
    
    # Create an Open3D PointCloud object
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(xyz_ks0[:,0:3])
    # o3d.io.write_point_cloud(os.path.join(MESH_DIR,"new_mesh2_intersec_reverse.xyz"), point_cloud, write_ascii=True)
    

    set1 = set(scene_indices_pos)
    set2 = set(scene_indices_neg)
    set3 = set1.union(set2)
    
    # backprojection into image space
    xyz_ksj     = np.dot(np.eye(4,4),xyz_ks0.T).T
    
    uv_ksj      = np.round(np.dot(trafo["mtxR"],xyz_ksj.T[0:3,:]/xyz_ksj.T[2,:]).T).astype(np.int32)
    
    
    h_bound     = (np.logical_and(uv_ksj[:,1]>=0,
                              uv_ksj[:,1]<modality_list[0]["modalities"]["infrared"]["img_raw"].shape[0]))
                
    w_bound     = (np.logical_and(uv_ksj[:,0]>=0,
                              uv_ksj[:,0]<modality_list[0]["modalities"]["infrared"]["img_raw"].shape[1]))
    
    b_mask           = h_bound*w_bound
    b_mask_indices   = np.argwhere(b_mask)[:,0]
    
    uv_ksj_filt         = uv_ksj[b_mask_indices]
    
    #positive direction
    
    rays            = np.c_[img1_xyz_ks2[:,0:3],normal_vector[:,0:3]].astype(np.float32)
    
    scene_hit           = (depth2["scene"].cast_rays(rays))["t_hit"].numpy()
       
    hit_mask            = scene_hit!=np.inf
    scene_indices_pos   = np.argwhere(scene_hit!=np.inf)[:,0]
    
    xyz_ksx         = normal_vector.copy()
    xyz_ksx[:,0:3]  = np.multiply(normal_vector[:,0:3],np.expand_dims(scene_hit,-1)) # scaled to correct size
    
    xyz_ks0         = xyz_ksx.copy()
    xyz_ks0[:,0:3]  = img1_xyz_ks2[:,0:3] + xyz_ks0[:,0:3]
    
    xyz_ks0_pos         = xyz_ks0[scene_indices_pos].copy()
    
    # backprojection into image space
    xyz_ksj     = np.dot(np.eye(4,4),xyz_ks0_pos.T).T
    
    uv_ksj      = np.round(np.dot(trafo["mtxR"],xyz_ksj.T[0:3,:]/xyz_ksj.T[2,:]).T).astype(np.int32)
    
    
    h_bound     = (np.logical_and(uv_ksj[:,1]>=0,
                              uv_ksj[:,1]<modality_list[0]["modalities"]["infrared"]["img_raw"].shape[0]))
                
    w_bound     = (np.logical_and(uv_ksj[:,0]>=0,
                              uv_ksj[:,0]<modality_list[0]["modalities"]["infrared"]["img_raw"].shape[1]))
    
    b_mask           = h_bound*w_bound
    b_mask_indices   = np.argwhere(b_mask)[:,0]
    
    uv_ksj_filt_pos  = uv_ksj[b_mask_indices]
    
    rel_img1_uv_pos  = img1_uv.T[img1_indices][scene_indices_pos][b_mask_indices]
    
    
    # negative direction
    rays_reverse    = np.c_[img1_xyz_ks2[:,0:3],-normal_vector[:,0:3]].astype(np.float32)
    
    scene_hit_reverse   = (depth2["scene"].cast_rays(rays_reverse))["t_hit"].numpy()
    
    hit_mask        = scene_hit!=np.inf
    scene_indices_neg   = np.argwhere(scene_hit_reverse!=np.inf)[:,0]
    
    xyz_ksx         = - normal_vector.copy()
    xyz_ksx[:,0:3]  = np.multiply(xyz_ksx[:,0:3],np.expand_dims(scene_hit_reverse,-1)) # scaled to correct size
    
    xyz_ks0         = xyz_ksx.copy()
    xyz_ks0[:,0:3]  = img1_xyz_ks2[:,0:3] + xyz_ks0[:,0:3]
    
    xyz_ks0_neg         = xyz_ks0[scene_indices_neg].copy()
    
    # backprojection into image space
    xyz_ksj     = np.dot(np.eye(4,4),xyz_ks0_neg.T).T
    
    uv_ksj      = np.round(np.dot(trafo["mtxR"],xyz_ksj.T[0:3,:]/xyz_ksj.T[2,:]).T).astype(np.int32)
    
    
    h_bound     = (np.logical_and(uv_ksj[:,1]>=0,
                              uv_ksj[:,1]<modality_list[0]["modalities"]["infrared"]["img_raw"].shape[0]))
                
    w_bound     = (np.logical_and(uv_ksj[:,0]>=0,
                              uv_ksj[:,0]<modality_list[0]["modalities"]["infrared"]["img_raw"].shape[1]))
    
    b_mask           = h_bound*w_bound
    b_mask_indices   = np.argwhere(b_mask)[:,0]
    
    uv_ksj_filt_neg  = uv_ksj[b_mask_indices]
    
    rel_img1_uv_neg  = img1_uv.T[img1_indices][scene_indices_neg][b_mask_indices]
       
    hs_proj_neg = np.zeros(bot_source.shape)
    hs_proj_neg[rel_img1_uv_neg[:,1],rel_img1_uv_neg[:,0]] = top_source[uv_ksj_filt_neg[:,1],uv_ksj_filt_neg[:,0]]
    
    hs_proj_pos = np.zeros(bot_source.shape)
    hs_proj_pos[rel_img1_uv_pos[:,1],rel_img1_uv_pos[:,0]] = top_source[uv_ksj_filt_pos[:,1],uv_ksj_filt_pos[:,0]]

    hs_proj = np.maximum(hs_proj_pos,hs_proj_neg)
    
    # convert hyperspectral back from infrared-depth frame to hyperspectral frame
    hs_top = np.zeros((512,512,3)).astype(np.uint8)
    
    hs_uv = modality_list[0]["reg_out"]["uv_source"][:,0:2]
    if_uv = modality_list[0]["reg_out"]["uv_target"][:,0:2]
    
    hs_top[hs_uv[:,1],hs_uv[:,0]] = hs_proj[if_uv[:,1],if_uv[:,0]]  
    
    xs, ys          = np.meshgrid(np.arange(512), np.arange(512))
    hs_uv_reg       = np.stack((xs.ravel(), ys.ravel()), axis=1)
    hs_uv_undist    = cv2.undistortPointsIter(np.expand_dims(hs_uv_reg[:,0:2].astype(np.float64),0),modality_config["hs"]["trafo"]["mtxR"],modality_config["hs"]["trafo"]["distR"],R = None,P = modality_config["hs"]["trafo"]["mtxR"],criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
    hs_uv_undist    = np.rint(hs_uv_undist).astype(int)
    hs_uv_undist[(hs_uv_undist < 0) | (hs_uv_undist > 511)] = 0
    
    hs_out = np.zeros((512,512,3)).astype(np.uint8)
    
    hs_out[hs_uv_reg[:,1],hs_uv_reg[:,0]] = hs_top[hs_uv_undist[:,1],hs_uv_undist[:,0]]   
    
    top_out_name  = os.path.basename(modality_config["hs"]["img_files"][modality_top_index])
    bot_out_name  = os.path.basename(modality_config["hs"]["img_files"][modality_bot_index])
    
    top_out_path = os.path.join(TRF_OUT_DIR,top_out_name)
    bot_out_path = os.path.join(TRF_OUT_DIR,bot_out_name)
    
    imageio.imwrite(top_out_path,hs_out)
    imageio.imwrite(bot_out_path,modality_list[0]["modalities"]["hs"]["img_raw"])



    # Mask transformation
    mask = imageio.imread(mask_files[i])/255
    
    
    hs_bot_uv_dist = np.flip(np.argwhere(mask == 1),axis=1)
    
    # to undistort
    hs_bot_uv      = np.rint(cv2.undistortPointsIter(np.expand_dims(hs_bot_uv_dist.astype(np.float64),0),
                                                      modality_config["hs"]["trafo"]["mtxR"],
                                                      modality_config["hs"]["trafo"]["distR"],
                                                      R = None,
                                                      P = modality_config["hs"]["trafo"]["mtxR"],
                                                      criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
                                                      ).astype(int)
 
    hs_bot_uv_infra = transform_coordinates(hs_bot_uv,
                                             modality_list[0]["reg_out"]["uv_source"][:,0:2],
                                             modality_list[0]["reg_out"]["uv_target"][:,0:2])
    
    
    hs_top_uv_infra = transform_coordinates(hs_bot_uv_infra,
                                            np.concatenate([rel_img1_uv_neg,rel_img1_uv_pos],axis = 0)[:,0:2],
                                            np.concatenate([uv_ksj_filt_neg,uv_ksj_filt_pos],axis = 0)[:,0:2])
    
    
    hs_top_uv       = transform_coordinates(hs_top_uv_infra,
                                            modality_list[1]["reg_out"]["uv_target"][:,0:2],
                                            modality_list[1]["reg_out"]["uv_source"][:,0:2])
    
    hs_coords_dist = (_pixel_coord_np(mask.shape[1], mask.shape[0]).T)[:,0:2]
    hs_coords = cv2.undistortPointsIter(np.expand_dims(hs_coords_dist[:,0:2].astype(np.float64),0),modality_config["hs"]["trafo"]["mtxR"],modality_config["hs"]["trafo"]["distR"],R = None,P = modality_config["hs"]["trafo"]["mtxR"],criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
    hs_coords = np.rint(hs_coords).astype(int)
    
    hs_top_uv_dist  = transform_coordinates(hs_top_uv,
                                            hs_coords,
                                            hs_coords_dist)
    
    top_mask = np.zeros(mask.shape)
    
    top_mask[hs_top_uv_dist[:,1],hs_top_uv_dist[:,0]] = 255
    
    
    mask_out_name = os.path.basename(modality_config["hs"]["img_files"][modality_top_index])
    
    imageio.imsave(os.path.join(MASK_OUT_DIR,mask_out_name),top_mask.astype(np.uint8))
    
    
    
    
    
 
