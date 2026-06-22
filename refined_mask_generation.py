# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:39:27 2025

@author: ericx
"""




# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:39:27 2025

@author: ericx
"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2

def gaussian_kernel(ks, sigma):
    # Create coordinate grid centered at (0,0)
    ax = np.arange(-ks // 2 + 1., ks // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Apply the Gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    # Normalize so the sum of all values is 1
    return kernel / np.sum(kernel)


def gaussian_kernel_peak1(ks, sigma):
    ax = np.arange(-ks // 2 + 1., ks // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return k / k.max()  # peak = 1

def dilate_until(img,ks,min_pixels):
    
    if np.sum(img/255) > min_pixels:
        return img
    
    else:
        
        dil_img = img
        kernel = np.ones((ks, ks), np.uint8)
         
        for i in range(10):
            
            dil_img = cv2.dilate(dil_img, kernel, iterations=1)
            
            if np.sum(dil_img/255) > min_pixels:
                return dil_img
            
    return img

def remove_exclusive_files(files1, files2):
     
    basenames1 = [os.path.basename(f).split(".")[0] for f in files1]
    basenames2 = [os.path.basename(f).split(".")[0] for f in files2]
    
    set1 = set(basenames1)
    set2 = set(basenames2)
    
    intersec_list = list(set1.intersection(set2)) 
    
    
    out_files1 = []
    out_files2 = []
    
    for item in intersec_list:
        
        index1 = basenames1.index(item)
        index2 = basenames2.index(item)
        
        out_files1.append(files1[index1])
        out_files2.append(files2[index2])
    
    return out_files1,out_files2
        

# choose the model between dilated label refinement and peak label refinement
mode = "peak" #"peak" "dilation"    
# mode = "dilation" #"peak" "dilation"  

ROOT_DIR = os.getcwd()

OUT_DIR = os.path.join(ROOT_DIR,"peak_refinement_masks",mode)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# gt masks that have been transformed from the annotated bottom side to the top side
top_transfer_files      = sorted(glob.glob(os.path.join(ROOT_DIR,"dataset","hs_mask_transfer_direct","*.png")))

# top pseud-labels from patch-CNN model trained only on bottom data
top_pseudo_label_files  = sorted(glob.glob(os.path.join(ROOT_DIR,"dataset","hs_pseudo_top","*.npy")))

top_transfer_files,top_pseudo_label_files = remove_exclusive_files(top_transfer_files,top_pseudo_label_files)


gauss_sigma = 3
tolerance   = 0.6  # Fraction of the difference between max and median
min_pixels  = 20 
dilate      = True
window_size = 21
roi = [205,333,176,304] # relevant hyperspectral window
         
for trnsf,pseudof in zip(top_transfer_files,top_pseudo_label_files):
    
    trns   = imageio.imread(trnsf)
    pseudo = np.load(pseudof)
    
    pseudo_orig = pseudo.copy()
    
    pseudo  = pseudo[roi[0]:roi[1], roi[2]:roi[3]]
    trns  = trns[roi[0]:roi[1], roi[2]:roi[3]]
    
    output = cv2.connectedComponentsWithStats(trns,connectivity=8)

    rad = int(window_size/2)
         
    new_mask = np.zeros(trns.shape)
    
    for index in range(output[3].shape[0]-1):
             
        coords = np.rint(output[3][index+1]).astype(int)
        height, width = trns.shape[:2]
        
        # Desired ROI size
        roi_size = 2*rad + 1
        
        # Create zero-padded array (this is the output ROI)
        pseudo_roi  = np.zeros((roi_size, roi_size), dtype=pseudo.dtype)
        trns_roi    = pseudo_roi.copy()
        
        # Coordinates in the source image
        ys = coords[1] - rad
        ye = coords[1] + rad + 1
        xs = coords[0] - rad
        xe = coords[0] + rad + 1
        
        # Compute overlapping region within image bounds
        src_y1 = max(0, ys)
        src_y2 = min(height, ye)
        src_x1 = max(0, xs)
        src_x2 = min(width, xe)
        
        # Compute corresponding region in the padded excerpt
        dst_y1 = src_y1 - ys
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = src_x1 - xs
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        
        # Copy the overlapping region into the padded ROI
        pseudo_roi[dst_y1:dst_y2, dst_x1:dst_x2] = pseudo[src_y1:src_y2, src_x1:src_x2]
        trns_roi[dst_y1:dst_y2, dst_x1:dst_x2] = trns[src_y1:src_y2, src_x1:src_x2]      
        
        
        if mode == "peak":
        
        
            # apply gaussian reweighting
            kernel = gaussian_kernel_peak1(window_size,gauss_sigma)  
            pseudo_roi*=kernel    
        
            #Calculate key statistics
            max_val = np.max(pseudo_roi[dst_y1:dst_y2, dst_x1:dst_x2])
            median_val = np.median(pseudo_roi[dst_y1:dst_y2, dst_x1:dst_x2])
        
            # Define a tolerance factor (adjust as needed)
            
            # Compute threshold
            threshold = median_val + tolerance * (max_val - median_val)
            
            # Threshold the image
            dominantly_high = (pseudo_roi >= threshold).astype(np.uint8) * 255
            
            out = dominantly_high
            
        elif mode == "dilation":
            
            out = trns_roi.astype(np.uint8)
            
        a = cv2.connectedComponentsWithStats(out,connectivity=8)   
        b = a[2][1:,4]
        c = np.argmax(b)+1    
        
        single_ele_mask = np.zeros((a[1].shape[0],a[1].shape[1])).astype(np.uint8)
        single_ele_mask[a[1]==c] = 255  
             
        if dilate:
             single_ele_mask = dilate_until(single_ele_mask,3,min_pixels)
 
        new_mask[src_y1:src_y2, src_x1:src_x2] += single_ele_mask[dst_y1:dst_y2, dst_x1:dst_x2]
        
        
    print(f"saving: {os.path.basename(trnsf)}")
        



    full_mask = np.zeros(pseudo_orig.shape)
    full_mask[roi[0]:roi[1], roi[2]:roi[3]] = new_mask
    full_mask = full_mask.astype(np.uint8)

    out_path = os.path.join(OUT_DIR,os.path.basename(trnsf))
    imageio.imwrite(out_path,full_mask)
    
    print(f"saving: {os.path.basename(trnsf)}")












