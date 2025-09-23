#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:30:54 2022

@author: xies
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage import draw, filters, morphology, exposure, measure

from scipy import sparse, ndimage
import pandas as pd
from typing import Iterable
from functools import reduce
from aicsshparam import shtools

def get_pad(current_size:int, target_size:int):
    # https://stackoverflow.com/questions/79413708/how-can-i-make-the-image-centered-while-padding-in-python
    difference = target_size - current_size
    if difference<=0: return (0,0)

    left_pad = difference//2
    right_pad = difference-left_pad
    return left_pad,right_pad

def pad_image_to_size_centered(im:np.array,size,pad_value=0):
    assert(im.ndim == len(size))

    padding = (get_pad(im.shape[0],size[0]), # Z-pad
               get_pad(im.shape[1],size[1]), # Y-pad
               get_pad(im.shape[2],size[2]),) # X-pad
    image_padded = np.pad(im, padding, constant_values = pad_value)
    return image_padded

def create_average_object_from_multiple_masks(mask_list:Iterable,prealign:bool=True):

    if prealign:
        mask_list = [shtools.align_image_2d(m)[0].squeeze() for m in mask_list]
    
    max_size = np.array([m.shape for m in mask_list]).max(axis=0)
    padded_mask = [pad_image_to_size_centered(m,max_size).astype(float) for m in mask_list]
    average_object = np.array(padded_mask).mean(axis=0)

    return average_object

def trim_multimasks_to_shared_bounding_box(mask_list:Iterable, border:int=5):
    '''
    Trim multiple masks to just the overlapping active pixels
    with a border tolerance (default = 5px)

    Parameters
    ----------
    mask_list: iterable
        Collection of all the masks
    border : int
        Size of the border. Default is 5.

    Returns
    -------
    trimmed_mask : np.array
        The mask where the off pixels are trimmed.
    '''
    mask = reduce(lambda x,y: x+y,mask_list)
    assert(mask.dtype==bool)
    assert(mask.ndim == 3)

    bbox = measure.regionprops(mask.astype(int))
    slc = get_mask_slices(bbox[0], mask.shape, border= border)
    trimmed_masks = []
    for m in mask_list:
        trimmed_masks.append(m[slc[0],slc[1],slc[2]])

    return trimmed_masks

def trim_image_to_bounding_box(mask:np.array, border:int=5):
    '''
    Trim a binary image to just its active pixels, with a border tolerance (default = 5px)

    Parameters
    ----------
    mask : np.array
        3d images only
    border : int
        Size of the border. Default is 5.

    Returns
    -------
    trimmed_mask : np.array
        The mask where the off pixels are trimmed.
    '''
    assert(mask.dtype==bool)
    assert(mask.ndim == 3)

    bbox = measure.regionprops(mask.astype(int))
    slc = get_mask_slices(bbox[0], mask.shape, border= border)
    trimmed_mask = mask[slc[0],slc[1],slc[2]]
    return trimmed_mask

def filter_seg_by_largest_object(seg):
    all_labels = np.unique(seg)[1:]
    new_seg = np.zeros_like(seg)
    for label in all_labels:
        mask = seg == label
        new_mask = filter_mask_by_largest_object(mask)
        new_seg[new_mask] = label
    return new_seg
    
def filter_mask_by_largest_object(mask):
    labels = measure.label(mask)
    df = pd.DataFrame(measure.regionprops_table(labels, properties=['label','area']))
    df = df.sort_values(by='area',ascending=False)
    largest_object = df.iloc[0]['label']
    return labels == largest_object

def rotate_image_3d(im,rot_angles_by_axes,order=[0,1,2]):
    '''
    Rotate a 3d image by specified axes, in the specified order.

    Parameters
    ----------
    im : np.array
        3d image (only)
    rot_angles_by_axes : list or np.array
        Angles in DEGREES. The order corresponds to each axis of rotation. For example,
        if the axes of im are: [Z,Y,X], then the angles = [a,b,c] specified are:

            a - rotation about the z axis
            b - rotation about the y axis
            c - rotation about the x axis

    order : TYPE, optional
        order of operation. Default is [0,1,2]

    Returns
    -------
    im : TYPE
        rotated image.

    '''
    assert(im.ndim == 3)
    assert(len(rot_angles_by_axes) == 3)
    rot_angles_by_axes = np.array(rot_angles_by_axes)

    # @todo: reorder for non default order

    axes = [[0,1],[0,2],[1,2]]
    for i,angle in enumerate(rot_angles_by_axes):
        im = ndimage.rotate(im, angle = angle, axes = axes[i])

    # T = transform.EuclideanTransform(rotation=euler_angles,dimensionality=3)
    # im = transform.warp(im,T)
    return im


def get_mask_slices(prop, max_dims, border = 0):

    zmin,ymin,xmin,zmax,ymax,xmax = prop['bbox']
    zmin = max(0,zmin - border)
    ymin = max(0,ymin - border)
    xmin = max(0,xmin - border)

    (ZZ,YY,XX) = max_dims
    zmax = min(ZZ,zmax + border)
    ymax = min(YY,ymax + border)
    xmax = min(XX,xmax + border)

    slc = [slice(None)] * 3

    slc[0] = slice( int(zmin),int(zmax))
    slc[1] = slice(int(ymin),int(ymax))
    slc[2] = slice(int(xmin),int(xmax))

    return slc


def fill_in_cube(img,coordinates,label,size=5):
    '''
    Takes in a 3D image and 'fill' a cube with given size at the given
    coordinate and given label
    '''
    assert(img.ndim == 3)
    assert(img.ndim == len(coordinates))

    [z,y,x] = coordinates
    ZZ,YY,XX = img.shape
    lower_x = max(x - size,0)
    lower_y = max(y - size,0)
    lower_z = max(z - size//2,0)
    higher_x = min(x + size,XX)
    higher_y = min(y + size,YY)
    higher_z = min(z + size//2,ZZ)
    img[lower_z:higher_z,lower_y:higher_y,lower_x:higher_x] = label

    return img



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

def adjdict_to_mat(adjdict:dict):
    I = []; J = []
    for center,neighbors in adjdict.items():
        for n in neighbors:
            I.append(center)
            J.append(n)
    I = np.array(I).astype(int)
    J = np.array(J).astype(int)
    V = np.ones(len(I)).astype(int)

    return sparse.coo_array( (V, (I,J)))

def draw_thick_line(volume, start, end, radius=2, value=255):
    """
    Draw a thick 3D line into a 3D numpy array (volume) by placing
    spheres along a line from `start` to `end`.

    Parameters:
    - volume: np.ndarray, shape (Z, Y, X)
        The 3D volume to modify.
    - start, end: tuple of int
        Start and end points of the line, in (z, y, x) format.
    - radius: int
        Radius of the spherical thickness (in voxels).
    - value: int
        The value to assign to the voxels in the thick line.
    """
    # Compute line coordinates
    z_coords, y_coords, x_coords = draw.line_nd(start, end)

    # Get ball structuring element and its offset positions
    ball_mask = morphology.ball(radius)
    dz, dy, dx = np.where(ball_mask)  # coordinates of 1s in the ball
    dz -= radius
    dy -= radius
    dx -= radius

    # Iterate over each point in the line
    for zc, yc, xc in zip(z_coords, y_coords, x_coords):
        # Calculate voxel positions for the ball centered at (zc, yc, xc)
        zz = dz + zc
        yy = dy + yc
        xx = dx + xc

        # Clip to volume bounds
        inside = (zz >= 0) & (zz < volume.shape[0]) & \
                 (yy >= 0) & (yy < volume.shape[1]) & \
                 (xx >= 0) & (xx < volume.shape[2])

        volume[zz[inside], yy[inside], xx[inside]] = value
    return volume

def draw_adjmat_on_image_3d(A,
                            vert_coords:pd.DataFrame,
                            im_shape:Iterable[int],
                            line_width:int=1,
                            colorize:bool=False):
    '''
    Convert a adjacency matrix and corresponding 3D coordinates into a 3D image

    Parameters
    ----------
    A : array (numpy or sparse)
        (Sparse) sdjacency matrix where each vertex is an element in vert_coords
    vert_coords : pd.DataFrame
        vertex coordiates in 3D, labelled by the same index given in A.
        with columns 'Z' 'Y' 'X'
    im_shape : Iterable[int]
        the 3d shape of 3d Image to generate
    line_width: int
        width of connecting lines. Default is 1
    colorize: bool
        whether to put the correct 'label' on the half of each line segment that
        emenates from the centroid. Default is False, which sets all segments to 255.

    Returns
    -------
    im : np.array
        Image of the connectivity graph represented by A

    '''
    # assert 3d coords
    assert('Z' in vert_coords.columns)
    im = np.zeros(im_shape)
    # num_verts = A.shape[0]

    # To avoid double drawing, trim to upper triangle
    # NB: diagonal should be 0
    # Convert matrix to COO sprase for fast I/J interation
    A = sparse.coo_array(A)
    A = sparse.triu(A)

    for src,dest in zip(A.row, A.col):
        this_coord = np.round(vert_coords.loc[src][['Z','Y','X']]).astype(int)
        neighbor_coord = np.round(vert_coords.loc[dest][['Z','Y','X']]).astype(int)

        if not colorize:
            if line_width == 1:
                mask = draw.line_nd(this_coord,neighbor_coord)
                im[mask] = 255
            else:
                im = draw_thick_line(im,this_coord,neighbor_coord,radius =line_width, value=255)
        else:
            half_way = np.round( (this_coord + neighbor_coord) /2 )
            if line_width == 1:
                mask = draw.line_nd(this_coord,half_way)
                im[mask] = src
                mask = draw.line_nd(neighbor_coord,half_way)
                im[mask] = dest
            else:
                im = draw_thick_line(im,this_coord,half_way,radius =line_width, value=src)
                im = draw_thick_line(im,neighbor_coord,half_way,radius =line_width, value=dest)

    return im


def most_likely_label(labeled,im,pixel_threshold=25):
    '''
    For use as an property function to give to skimage.measure.regionprops

    Given a mask image, return the intensity value within that image that the highest occurence


    '''
    label = 0
    if len(im[im>0]) > 0:
        unique,counts = np.unique(im[im > 0],return_counts=True)
        label = unique[counts.argmax()]
        if counts.max() < pixel_threshold:
            label = np.nan
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

    assert(image.ndim == 3)
    from skimage.util import img_as_float
    image = img_as_float(image)

    im_blur = np.zeros_like(image)
    Z,Y,X = image.shape

    for z,im in enumerate(image):
        im_blur[z,...] = filters.gaussian(im, sigma=sigma_xy)

    for x in range(X):
        for y in range(Y):
            im_blur[:,y,x] = filters.gaussian(image[:,y,x], sigma=sigma_z)

    return im_blur


def normalize_exposure_by_axis(im_stack:np.array, axis:int=0):
    '''
    Uses histogram normalization to re-normalize the intensity of an image stack
    along the given axis. Image stack has to be >3D.

    Parameters
    ----------
    im_stack : np.array
        input >3D image stack. Will cast dtype to float.
    axis : int, optional
        axis of normalization. The default is 0.

    Returns
    -------
    normalized_stack : np.array

    '''

    assert(im_stack.ndim > 2)
    im_stack = im_stack.astype(float)
    # Move the axis to the 0th
    im_stack = np.moveaxis(im_stack,axis,0)

    normalized_stack = np.zeros_like(im_stack)
    for i,im in enumerate(im_stack):
        normalized_stack[i,...] = exposure.equalize_hist(im)

    normalized_stack = np.moveaxis(normalized_stack,0,axis)
    return normalized_stack
