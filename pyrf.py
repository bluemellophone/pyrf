#!/usr/bin/env python
from __future__ import print_function, division
# Standard
from os.path import realpath, dirname
#from os import makedirs
from collections import OrderedDict as odict
import shutil
import ctypes as C
# Scientific
import numpy as np
import cv2
#from PIL import Image
import os
import sys
import time
import threading
#import random
#import xml.etree.ElementTree as xml
# https://github.com/bluemellophone/detecttools
try:
    import detecttools  # NOQA
except ImportError:
    sys.path.append(os.path.expanduser('~/code'))
    try:
        import detecttools  # NOQA
    except ImportError:
        print('Cannot find detecttools!')
        raise
import detecttools.ctypes_interface as ctypes_interface
#from detecttools.directory import Directory
from detecttools.ibeisdata import IBEIS_Data


__LOCATION__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


#============================
# CTypes Interface Data Types
#============================


# Bindings for Numpy Arrays
FLAGS_RW = 'aligned, c_contiguous, writeable'
CNPFLOAT = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=FLAGS_RW)
CNPINT   = np.ctypeslib.ndpointer(dtype=np.uint8,   ndim=2, flags=FLAGS_RW)

# Bindings for C Variable Types
COBJ     = C.c_void_p
CCHAR    = C.c_char_p
CINT     = C.c_int
CBOOL    = C.c_bool
CFLOAT   = C.c_float


#=================================
# Default / Constructor Parameters
#=================================


'''
    This defines the default constructor parameters for the algorithm.
    These values may be overwritten by passing in a dictionary to the
    class constructor using kwargs

    constructor_parameters = [
        (parameter type, parameter name, parameter default value),
    ]

    IMPORTANT:
    The order of this list must match the C++ constructor parameter order
'''
constructor_parameters = [
    (CINT,  'patch_width',              32),
    (CINT,  'patch_height',             32),
    (CINT,  'out_scale',                128),
    (CINT,  'default_split',            -1),

    (CINT,  'pos_like',                 0),
    # 0 - Hough
    # 1 - Classification
    # 2 - Regression

    (CBOOL, 'legacy',                   False),
    (CBOOL, 'include_horizontal_flip',   True),
    (CINT,  'patch_sample_density_pos',     4),
    (CINT,  'patch_sample_density_neg',     4),
    (CCHAR, 'scales',                       '6 1.0 0.75 0.55 0.40 0.30 0.20'),
    (CCHAR, 'ratios',                       '1 1.0'),
]

# Do not touch
PARAM_ODICT = odict([(key, val) for (_type, key, val) in constructor_parameters])
PARAM_TYPES = [_type for (_type, key, val) in constructor_parameters]


#============================
# Python Interface
#============================


def _kwargs(kwargs, key, value):
    if key not in kwargs.keys():
        kwargs[key] = value


def _build_shared_c_library(rebuild=False):
    if rebuild:
        shutil.rmtree(os.path.join(__LOCATION__, 'build'))

    retVal = os.system('./build_rf_unix.sh')

    if retVal != 0:
        print('[rf] C Shared Library failed to compile')
        sys.exit(0)

    print('[rf] C Shared Library built')


def _prepare_inventory(directory_path, images, total, train=True, positive=True):
        output_fpath = directory_path + '.txt'
        output = open(output_fpath, 'w')

        if train:
            output.write(str(total) + ' 1\n')
        else:
            output.write(str(total) + '\n')

        for counter, image in enumerate(images):
            if counter % int(len(images) / 10) == 0:
                print('%0.2f' % (float(counter) / len(images)))

            filename = os.path.join(directory_path, image.filename)

            if train:
                i = 1
                cv2.imwrite(filename + '_boxes.jpg', image.show(display=False))
                for bndbox in image.bounding_boxes():
                    if positive and bndbox[0] != category:
                        continue

                    _filename = filename + '_' + str(i) + '.jpg'

                    xmax = bndbox[1]  # max
                    xmin = bndbox[2]  # xmin
                    ymax = bndbox[3]  # ymax
                    ymin = bndbox[4]  # ymin

                    width, height = (xmax - xmin), (ymax - ymin)

                    temp = cv2.imread(image.image_path())  # Load
                    temp = temp[ymin:ymax, xmin:xmax]      # Crop

                    target_width = 128
                    if width > target_width:
                        _width = int(target_width)
                        _height = int((_width / width) * height)
                        temp = cv2.resize(temp, (_width, _height), interpolation=cv2.INTER_LANCZOS4)  # Resize
                        width = _width
                        height = _height

                    xmax = width
                    xmin = 0
                    ymax = height
                    ymin = 0

                    if positive:
                        postfix = ' %d %d %d %d %d %d' % (xmin, ymin, xmax, ymax, xmin + width / 2, ymin + height / 2)
                    else:
                        postfix = ' %d %d %d %d' % (xmin, ymin, xmax, ymax)

                    cv2.imwrite(_filename, temp)  # Save
                    output.write(_filename + postfix + '\n')
                    i += 1
            else:
                postfix = ''
                cv2.imwrite(filename, cv2.imread(image.image_path()))  # Save
                output.write(filename + postfix + '\n')

        output.close()

        return output_fpath


def ibeis(dataset_path, category, pos_path, neg_path, val_path, test_path, **kwargs):

    dataset = IBEIS_Data(dataset_path, **kwargs)

    # How does the data look like?
    dataset.print_distribution()

    # Get all images using a specific positive set
    data = dataset.dataset(
        category,
        neg_exclude_categories=kwargs['neg_exclude_categories'],
        max_rois_pos=kwargs['max_rois_pos'],
        max_rois_neg=kwargs['max_rois_neg'],
    )

    (pos, pos_rois), (neg, neg_rois), val, test = data

    print('[rf] Caching Positives')
    pos_fpath = _prepare_inventory(pos_path, pos, pos_rois)

    print('[rf] Caching Negatives')
    neg_fpath = _prepare_inventory(neg_path, neg, neg_rois, positive=False)

    print('[rf] Caching Validation')
    test_fpath = _prepare_inventory(val_path, val, len(val), train=False)

    print('[rf] Caching Test')
    test_fpath = _prepare_inventory(test_path, test, len(test), train=False)  # FIXME UNUSED  # NOQA

    return pos_fpath, neg_fpath


class Random_Forest_Detector(object):

    #=============================
    # Algorithm Constructor
    #=============================
    def __init__(rf, libname='pyrf', rebuild=False, **kwargs):

        print('[rf] Testing Random_Forest')

        if rebuild:
            _build_shared_c_library(rebuild)

        '''
        Loads the compiled lib and defines its functions
        '''
        root_dir = realpath(dirname(__file__))
        rf.CLIB, LOAD_FUNCTION = ctypes_interface.load_clib(libname, root_dir)

        '''
        def_lib_func is used to expose the Python bindings that are declared
        inside the .cpp files to the Python clib object.

        def_lib_func(return type, function name, list of parameter types)

        IMPORTANT:
        For functions that return void, use Python None as the return value.
        For functions that take no parameters, use the Python empty list [].
        '''
        LOAD_FUNCTION(COBJ, 'constructor',      PARAM_TYPES)
        LOAD_FUNCTION(None, 'train',            [COBJ, CCHAR, CINT, CCHAR, CCHAR])
        LOAD_FUNCTION(CINT, 'detect',           [COBJ, COBJ, CCHAR, CCHAR, CBOOL, CBOOL, CBOOL, CINT, CINT, CFLOAT, CFLOAT, CFLOAT, CINT])
        LOAD_FUNCTION(None, 'detect_results',   [COBJ, CNPFLOAT])
        LOAD_FUNCTION(None, 'segment',          [COBJ])
        LOAD_FUNCTION(COBJ, 'load',             [COBJ, CCHAR, CCHAR, CINT])
        LOAD_FUNCTION(None, 'save',             [COBJ])
        # Add any algorithm-specific functions here

        '''
        Create the C object using the default parameter values and any updated
        parameter values from kwargs
        '''
        _PARAM_ODICT = PARAM_ODICT.copy()
        _PARAM_ODICT.update(kwargs)

        print('[rf] New Random_Forest Object Created')
        print('[rf] Algorithm Settings=%r' % (_PARAM_ODICT,))

        PARAM_VALUES = _PARAM_ODICT.values()  # pass all parameters to the C constructor
        rf.detector = rf.CLIB.constructor(*PARAM_VALUES)

    def _run(rf, target, args):
        t = threading.Thread(target=target, args=args)
        t.daemon = True
        t.start()
        while t.is_alive():  # wait for the thread to exit
            t.join(.1)

    #=============================
    # Train Algorithm with Data
    #=============================

    def train(rf, database_path, category, pos_path, neg_path, val_path, test_path, tree_path, **kwargs):
        _kwargs(kwargs, 'num_trees', 10)

        def _rmtreedir(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
        def _ensuredir(path):
            if os.path.isdir(path):
                os.makedirs(path)

        print('[rf] Clearing Test Cache Directories')
        _rmtreedir(pos_path)
        _rmtreedir(neg_path)
        _rmtreedir(val_path)
        _rmtreedir(test_path)
        #if os.path.isdir(pos_path):         shutil.rmtree(pos_path)
        #if os.path.isdir(neg_path):         shutil.rmtree(neg_path)
        #if os.path.isdir(val_path):         shutil.rmtree(val_path)
        #if os.path.isdir(test_path):        shutil.rmtree(test_path)

        print('[rf] Creating Test Cache Directories')
        _ensuredir(pos_path)
        _ensuredir(neg_path)
        _ensuredir(val_path)
        _ensuredir(test_path)
        _ensuredir(trees_path)
        #if not os.path.isdir(pos_path):       os.makedirs(pos_path)
        #if not os.path.isdir(neg_path):       os.makedirs(neg_path)
        #if not os.path.isdir(val_path):       os.makedirs(val_path)
        #if not os.path.isdir(test_path):      os.makedirs(test_path)
        #if not os.path.isdir(trees_path):     os.makedirs(trees_path)

        # Gather training data from IBEIS database
        fpath_pos, fpath_neg = ibeis(database_path, category, pos_path, neg_path, val_path, test_path, **kwargs)

        # Run training algorithm
        params = [rf.detector, tree_path, kwargs['num_trees'], fpath_pos, fpath_neg]
        rf._run(rf.CLIB.train, params)

    def retrain(rf):
        rf._run(rf.CLIB.retrain, [rf.detector])

    #=============================
    # Run Algorithm
    #=============================

    def detect(rf, forest, image_fpath, result_fpath, **kwargs):
        start = time.time()

        _kwargs(kwargs, 'save_detection_images',   False)
        _kwargs(kwargs, 'save_scales',             False)
        _kwargs(kwargs, 'draw_supressed',          False)
        _kwargs(kwargs, 'detection_width',         128)
        _kwargs(kwargs, 'detection_height',        80)
        _kwargs(kwargs, 'percentage_left',         0.50)
        _kwargs(kwargs, 'percentage_top',          0.50)
        _kwargs(kwargs, 'nms_margin_percentage',   0.75)
        _kwargs(kwargs, 'min_contour_area',        300)

        length = rf.CLIB.detect(
            rf.detector,
            forest,
            image_fpath,
            result_fpath,
            kwargs['save_detection_images'],
            kwargs['save_scales'],
            kwargs['draw_supressed'],
            kwargs['detection_width'],
            kwargs['detection_height'],
            kwargs['percentage_left'],
            kwargs['percentage_top'],
            kwargs['nms_margin_percentage'],
            kwargs['min_contour_area'],
        )

        results = np.empty((length, 8), np.float32)
        rf.CLIB.detect_results(rf.detector, results)

        done = time.time()
        return results, done - start

    def segment(rf):
        rf._run(rf.CLIB.segment, [rf.detector])

    #=============================
    # Load / Save Trained Data
    #=============================

    def load(rf, tree_path, prefix, num_trees=10):
        return rf.CLIB.load(rf.detector, tree_path, prefix, num_trees)

    def save(rf):
        rf._run(rf.CLIB.save, [rf.detector])


if __name__ == '__main__':

    # Create detector
    detector = Random_Forest_Detector()
    category = 'zebra_plains'

    dataset_path = '../IBEIS2014/'
    pos_path     = os.path.join('results', category, 'train-positives')
    neg_path     = os.path.join('results', category, 'train-negatives')
    val_path     = os.path.join('results', category, 'val')
    test_path     = os.path.join('results', category, 'test')
    detect_path = os.path.join('results', category, 'detect')
    trees_path     = os.path.join('results', category, 'trees')
    tree_prefix = category + '-'

    #=================================
    # Train / Detect Configurations
    #=================================

    train_config = {
        'object_min_width':             32,
        'object_min_height':            32,
        'neg_exclude_categories':       [category],

        'mine_negatives':               True,
        'mine_max_keep':                10,
        'mine_exclude_categories':      [category],
        'mine_width_min':               128,
        'mine_width_max':               512,
        'mine_height_min':              128,
        'mine_height_max':              512,

        'max_rois_pos':                 None,
        'max_rois_neg':                 'auto',
    }

    detect_config = {
        'save_detection_images':        True,
        'percentage_top':               0.40,
    }

    #=================================
    # Train Random Forest
    #=================================

    # _trees_path = os.path.join(trees_path, tree_prefix)
    # detector.train(dataset_path, category, pos_path, neg_path, val_path, test_path, _trees_path, **train_config)

    #=================================
    # Detect using Random Forest
    #=================================

    print('[rf] Clearing Detect Cache Directories')
    if os.path.isdir(detect_path):
        shutil.rmtree(detect_path)

    print('[rf] Creating Detect Cache Directories')
    if not os.path.isdir(detect_path):
        os.makedirs(detect_path)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, tree_prefix)

    # Get input images
    test_file = open(test_path + '.txt', 'r')
    test_file.readline()
    files = [ line.strip() for line in test_file ]

    for i in range(len(files)):
        src_fpath = files[i]
        dst_fpath = os.path.join(detect_path, files[i].split('/')[-1])

        results, timing = detector.detect(forest, src_fpath, dst_fpath, **detect_config)

        print('[rf] %s | Time: %.3f' % (src_fpath, timing))
        print(results)
