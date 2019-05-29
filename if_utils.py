#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:10:51 2019

@author: mimi
"""


def gate_cells(df,x_name,y_name):
    import matplotlib.path as mplPath
    import matplotlib.pyplot as plt
    from roipoly import RoiPoly
    """
    Gate cells
    
    """
    plt.figure()
    sb.scatterplot(data = df, x=x_name, y=y_name)
    gate = RoiPoly(color='r') # Draw polygon gate
    gate_p = mplPath.Path( np.vstack((gate.x, gate.y)).T )
    gateI = gate_p.contains_points( np.vstack((df[x_name],df[y_name])).T )
    return df[gateI]
    

def subtract_nonmask_background(img,bg_mask,erosion_radius=5):
    from skimage import morphology
    """
    
    Use the 'ON' pixels from a binary mask of BACKGROUND to estimate background
    intensity. Uses an disk of radius 5 (default) to 'erode' first. Background
    intensity is defined as the mean of the leftover background pixels.
    
    """
    
    disk = morphology.selem.disk(erosion_radius) # for bg subtraction on RB channel
    bg_pixels = morphology.erosion(bg_mask,disk)
    bg = img[bg_pixels].mean()
    img_sub = img.copy()
    img_sub = img_sub - bg
    img_sub[img_sub < 0] = 0
    return img_sub
    