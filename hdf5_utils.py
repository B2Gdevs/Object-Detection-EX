################################################################################
# hdf5_utils
#   Utilities for creating and manipulating HDF5 files
# Copyright (c) 2018 Mayachira Inc.
# Written by: Elliot Staudt - staudt@mayachitra.com
# Sponsored by the Office of Naval Research
# Contract Number: N-6833518-C-0199
################################################################################

import h5py
import os
import numpy as np

# function to make an HDF5 file for use as a data store of
def create_datastore(input_name,_type,class_names):
    filename = input_name

    # This is the file we will be writing to
    f = h5py.File(filename,"w")

    # attributes of the input data
    # type: video, image directory, single image
    f.attrs['dataset_type'] = _type
    f.attrs['HDF5_Version'] = h5py.version.hdf5_version
    f.attrs['h5py_version'] = h5py.version.version

    # save the names of the classes that were used to detect
    data = np.array(class_names, dtype=object)
    string_dt = h5py.special_dtype(vlen=str)
    f.create_dataset("class_names", data=data, dtype=string_dt)

    #  bounding box data for each frame will in the form of three numpy arrays
    frames = f.create_group('frames')
    # This will be automatic in the future
    return f

def add_frame(file,frameName,rawDict):

    frame = file['frames'].create_group(frameName)


    ds = frame.create_dataset("bounding_boxes",rawDict['boxes'].shape,dtype='f')
    ds[...] = rawDict['boxes']
    ds = frame.create_dataset("scores",rawDict['scores'].shape,dtype='f')
    ds[...] = rawDict['scores']
    ds = frame.create_dataset("descriptors",rawDict['descriptors'].shape,dtype='f')
    ds[...] = rawDict['descriptors']
    ds = frame.create_dataset("segmentations",rawDict['segmentations'].shape,dtype='f')
    ds[...] = rawDict['segmentations']

    return file

def save_frame_data(filename,_type,class_names,rawDict,net,dataset):
    # This is the file we will be writing to
    f = h5py.File(filename,"w")

    # attributes of the input data
    # type: video, image directory, single image
    f.attrs['dataset_type'] = _type
    f.attrs['HDF5_Version'] = h5py.version.hdf5_version
    f.attrs['h5py_version'] = h5py.version.version
    f.attrs['NET']=net
    f.attrs['DATASET']=dataset

    # save the names of the classes that were used to detect
    data = np.array(class_names, dtype=object)
    string_dt = h5py.special_dtype(vlen=str)
    f.create_dataset("class_names", data=data, dtype=string_dt)

    # save the data
    ds = f.create_dataset("bounding_boxes",rawDict['boxes'].shape,dtype='f')
    ds[...] = rawDict['boxes']
    ds = f.create_dataset("scores",rawDict['scores'].shape,dtype='f')
    ds[...] = rawDict['scores']
    ds = f.create_dataset("features",rawDict['features'].shape,dtype='f')
    ds[...] = rawDict['features']

    # close the file and exit
    f.close()

    return
