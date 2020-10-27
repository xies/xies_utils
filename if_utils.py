#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:10:51 2019

@author: mimi
"""

import numpy as np

def gate_cells(df,x_name,y_name):
    import numpy as np
    import matplotlib.path as mplPath
    import matplotlib.pyplot as plt
    from roipoly import RoiPoly
    """∑
    Gate cells
    
    """
    plt.figure()
    sb.scatterplot(data = df, x=x_name, y=y_name)
    gate = RoiPoly(color='r') # Draw polygon gate
    gate_p = mplPath.Path( np.vstack((gate.x, gate.y)).T )
    gateI = gate_p.contains_points( np.vstack((df[x_name],df[y_name])).T )
    return df[gateI]
    

def subtract_nonmask_background(img,bg_mask,erosion_radius=5,mean=np.mean):
    from skimage import morphology
    """
    
    Use the 'ON' pixels from a binary mask of BACKGROUND to estimate background
    intensity. Uses an disk of radius 5 (default) to 'erode' first. Background
    intensity is defined as the mean of the leftover background pixels.
    
    """
    import numpy as np
    disk = morphology.selem.disk(erosion_radius) # for bg subtraction on RB channel
    bg_pixels = morphology.erosion(bg_mask,disk)
    bg = np.median(img[bg_pixels])
    img_sub = img.copy()
    img_sub = img_sub - bg
    img_sub[img_sub < 0] = 0
    return img_sub


    
def detect_border_object(labels):
    import numpy as np
    """
    
    Givenb a bwlabeled image, return the labels of the objects that touch the image border (1px wide)
    Background object should be labeled 0
    
    """
    border = np.zeros(labels.shape)
    border[0,:] = 1
    border[-1,:] = 1
    border[:,0] = 1
    border[:,-1] = 1
    touch_border = np.unique(labels[border == 1])
    
    return touch_border[touch_border > 0] # Get rid of 'background object'
    
def subtract_cytoplasmic_ring(img,nuclear_mask,inner_r=3,outer_r=5):
    from skimage import morphology
    import numpy as np
    
    """
    Generate a 'ring' of background pixels around the nuclei in an image.
    inner_r -- inner ring radius, # of pixels dilated from mask (default = 3)
    outer_r -- outer ring radius (default = 5)
    
    """
    
    kernel = morphology.disk(inner_r)
    inner_mask = morphology.dilation(nuclear_mask,kernel)
    kernel = morphology.disk(outer_r)
    outer_mask = morphology.dilation(nuclear_mask,kernel)
    ring = np.logical_xor(outer_mask,inner_mask)
    
    bg = np.mean(img[ring])
    img_sub = img.copy() - bg
    img_sub[img_sub < 0] = 0
    
    return img_sub


