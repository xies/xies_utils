#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:47:35 2024

From: https://github.com/fmi-basel/improc/blob/master/improc/mesh.py

"""

import numpy as np
from tqdm import tqdm
from vtk.util import numpy_support
import vtk
from scipy.ndimage import find_objects


def numpy_img_to_vtk(img, spacing, origin=(0., 0., 0.), deep_copy=True):
    '''Converts a numpy array to vtk image data.
    
    Args:
        img: numpy array
        spacing: tuple defining the px/voxel size
        origin: origin point in physical coordinates
        deep_copy: if False memory will be shared with the original numpy array. 
        It requires keeping a handle on the numpy array to prevent garbage collection.
    '''

    vtk_data = numpy_support.numpy_to_vtk(num_array=img.ravel(order='C'),
                                          deep=deep_copy)
    imageVTK = vtk.vtkImageData()
    imageVTK.SetSpacing(spacing[::-1])
    imageVTK.SetOrigin(origin[::-1])
    imageVTK.SetDimensions(img.shape[::-1])
    imageVTK.GetPointData().SetScalars(vtk_data)

    return imageVTK


def extract_smooth_mesh(imageVTK,
                        label_range,
                        smoothing_iterations=30,
                        pass_band_param=0.01,
                        target_reduction=0.95):
    '''Extract mesh/contour for labels in imageVTK, smooth and decimate.
    
    Multiple labels can be extracted at once, however touching labels 
    will share vertices and the label ids are lost during smoothing/decimation.
    Processing is slow for small objects in a large volume and should be cropped beforehand.
    
    Args:
        imageVTK: vtk image data
        label_range: range of labels to extract. A tuple (l,l) will extract 
            a mesh for a single label id l
        smoothing_iterations: number of iterations for vtkWindowedSincPolyDataFilter
        pass_band_param: pass band param in range [0.,2.] for vtkWindowedSincPolyDataFilter.
            Lower value remove higher frequencies.
        target_reduction: target reduction for vtkQuadricDecimation
    '''

    n_contours = label_range[1] - label_range[0] + 1

    # alternative vtkDiscreteMarchingCubes is slower and creates some weird missalignment lines when applied to tight crops
    dfe = vtk.vtkDiscreteFlyingEdges3D()
    dfe.SetInputData(imageVTK)
    dfe.ComputeScalarsOff(
    )  # numpy image labels --> cells (faces) scalar values
    dfe.ComputeNormalsOff()
    dfe.ComputeGradientsOff()
    dfe.InterpolateAttributesOff()
    dfe.GenerateValues(n_contours, label_range[0],
                       label_range[1])  # numContours, rangeStart, rangeEnd
    dfe.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(dfe.GetOutputPort())
    smoother.SetNumberOfIterations(
        smoothing_iterations)  #this has little effect on the error!
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    # smoother.SetFeatureAngle(120)
    smoother.SetPassBand(
        pass_band_param)  # from 0 to 2, 2 keeps high frequencies
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.GenerateErrorScalarsOff()
    smoother.GenerateErrorVectorsOff()
    smoother.Update()

    # vtkQuadricDecimation looks cleaner than vtkDecimatePro (no unexpected sharp edges)
    # but drop points scalar value --> can be added back if doing one instance a time
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputConnection(smoother.GetOutputPort())
    decimate.SetTargetReduction(target_reduction)
    decimate.VolumePreservationOn()
    decimate.Update()

    return decimate.GetOutput()


def labels_to_mesh(labels,
                   spacing,
                   smoothing_iterations=30,
                   pass_band_param=0.01,
                   target_reduction=0.95,
                   margin=5,
                   show_progress=True):
    '''Extract mesh/contour for a labels provided as a numpy array, smooth and decimate.
    
    Meshes are exctracted one label at a time one object crop.
    
    Args:
        imageVTK: vtk image data
        spacing: tuple defining the px/voxel size
        smoothing_iterations: number of iterations for vtkWindowedSincPolyDataFilter
        pass_band_param: pass band param in range [0.,2.] for vtkWindowedSincPolyDataFilter.
            Lower value remove higher frequencies.
        target_reduction: target reduction for vtkQuadricDecimation
        margin: margin bounding box used to crop each label. Needs at least margin=1 to extract 
        closed contours (i.e. label should not touch the bounding box), slightly more for the 
        smoothing operation.
        show_progress: display a progress bar. Useful for images containing many labels
    '''

    appendFilter = vtk.vtkAppendPolyData()

    iterable = find_objects(labels)
    if show_progress:
        iterable = tqdm(iterable)

    for idx, loc in enumerate(iterable, start=1):
        if loc:

            loc = tuple(
                slice(max(0, sl.start - margin), sl.stop + margin)
                for sl in loc)
            crop = (labels[loc] == idx).astype(np.uint8)
            origin = tuple(sl.start * s for sl, s in zip(loc, spacing))
            imageVTK = numpy_img_to_vtk(crop, spacing, origin, deep_copy=False)
            instance_mesh = extract_smooth_mesh(imageVTK, (1, 1),
                                                smoothing_iterations,
                                                pass_band_param,
                                                target_reduction)

            # add the label id as point data
            scalars = numpy_support.numpy_to_vtk(
                num_array=np.ones(instance_mesh.GetNumberOfPoints()) * idx,
                deep=True,
                array_type=vtk.VTK_INT)
            scalars.SetName('label_id')
            instance_mesh.GetPointData().SetScalars(scalars)
            
            appendFilter.AddInputData(instance_mesh)

    appendFilter.Update()

    return appendFilter.GetOutput()
