#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:33:32 2025

@author: xies
"""

import pyvista as pv
import numpy as np
from trimesh import Trimesh, transformations
from skimage import measure
from scipy.ndimage import binary_fill_holes
import warnings

def mask2mesh(mask):
    '''
    Converts a single contiguous mask into a mesk using marching cubes. Uses trimesh API.

    Parameters
    ----------
    mask : np.array
        boolean array of 4-connected pixels

    Returns
    -------
    Trimesh mesh object

    '''
    verts,faces,normals,values = measure.marching_cubes(mask,0)
    mesh = Trimesh(vertices = verts,
                           faces = faces,
                           process=True)
    return mesh


def rotate_mesh(mesh:Trimesh, angle:float, rotation_axis=[0,1,0],origin=None):
    '''
    Rotate mesh about a given axis

    Parameters
    ----------
    mesh : Trimesh
        mesh to rotate
    angle : float
        angle (in radians)
    rotation axis: array-like
        3D vector of the axis of rotation. Default is [0,1,0], aka y-axis
    origin : array-like
        origin of rotation. Default is [0,0,0]

    Returns
    -------
    Rotated Trimesh mesh object
    '''
    
    assert(len(rotation_axis) == 3)
    
    if angle > 2*np.pi:
        warnings.warn('Make sure angle is in radians.')
        
    T = transformations.rotation_matrix(angle,direction=rotation_axis,point=origin)

    return mesh.apply_transform(T)

def mesh2mask(mesh:Trimesh, pixel_size, origin=[0,0,0]):
   
    T = transformations.translation_matrix(origin)
    mesh = mesh.apply_transform(T)
    
    mask = mesh.voxelized(pitch=pixel_size)
    mask = mask.matrix
    mask = binary_fill_holes(mask)
    return mask

def plot_trimesh(tmesh, pl = None):
    m = pv.wrap(tmesh)

    if pl is None:
        pl = pv.Plotter()
        
    pl.add_mesh(m)
    pl.show()

    return pl