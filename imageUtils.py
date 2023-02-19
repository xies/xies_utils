#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:30:54 2022

@author: xies
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage import draw, filters

def draw_labels_on_image(coords,labels,im_shape,font_size=10,fill='white'):
    
    if fill == 'random':
        import random
        fill = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]

    
    # im = np.zeros(im_shape)
    image = Image.new('L',im_shape)
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/System/Library/Fonts/ArialHB.ttc',font_size)
    
    for i,l in enumerate(labels):
        text = f'{l}'
        draw.text([coords[i,1],coords[i,0]],text,font=font,fill=fill)
    
    return image


def draw_adjmat_on_image(A,vert_coords,im_shape):
    # assert 2d coords!!
    assert(vert_coords.shape[1] == 2)
    im = np.zeros(im_shape)
    num_verts = A.shape[0]
    
    # To avoid double drawing, trim to upper triangle
    # NB: diagonal should be 0
    # A = np.triu(A)
    
    for idx in range(num_verts):
        this_coord = np.round(vert_coords[idx,...]).astype(int)
        neighbor_idx = np.where(A[idx,:])[0]
        
        for neighbor in neighbor_idx:
            neighbor_coord = np.round(vert_coords[neighbor,...]).astype(int)
            # print(neighbor_coord)
            rr,cc = draw.line(this_coord[0],this_coord[1],neighbor_coord[0],neighbor_coord[1])
            im[rr,cc] = idx+1
            
    return im
        

def draw_adjmat_on_image_3d(A,vert_coords,im_shape):
    # assert 2d coords!!
    assert(vert_coords.shape[1] == 3)
    im = np.zeros(im_shape)
    num_verts = A.shape[0]
    
    # To avoid double drawing, trim to upper triangle
    # NB: diagonal should be 0
    A = np.triu(A)
    
    for idx in range(num_verts):
        this_coord = np.round(vert_coords[idx,...]).astype(int)
        neighbor_idx = np.where(A[idx])[0]
        
        for neighbor in neighbor_idx:
            neighbor_coord = np.round(vert_coords[neighbor,...]).astype(int)
            # print(neighbor_coord)
            lin = draw.line_nd(this_coord,neighbor_coord)
            im[lin] = idx + 1
    return im


def most_likely_label(labeled,im):
    label = 0
    if len(im[im>0]) > 0:
        unique,counts = np.unique(im[im > 0],return_counts=True)
        label = unique[counts.argmax()]
        if label == 0:
            label = np.nan
    return label


def colorize_segmentation(seg,value_dict,dtype=int):
    '''
    Given a segmentation label image, colorize the segmented labels using a dictionary of label: value
    '''
    
    assert( len(np.unique(seg[1:]) == len(value_dict)) )
    colorized = np.zeros_like(seg,dtype=dtype)
    for k,v in value_dict.items():
        colorized[seg == k] = v
    return colorized
    
def gaussian_blur_3d(image,sigma_xy=1,sigma_z=1):
    
    im_blur = np.zeros_like(image)
    Z,Y,X = image.shape
    
    for z,im in enumerate(image):
        im_blur[z,...] = filters.gaussian(im, sigma=sigma_xy)
    
    for x in range(X):
        for y in range(Y):
            im_blur[:,y,x] = filters.gaussian(image[:,y,z], sigma=sigma_z)
    
    return im_blur

