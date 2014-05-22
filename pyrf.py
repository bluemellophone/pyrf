#!/usr/bin/env python
from __future__ import print_function, division
# Standard
import os
import sys
import time
import threading
from os.path import dirname, join, isdir, expanduser, realpath
from collections import OrderedDict as odict
import shutil
import ctypes as C
# Scientific
import numpy as np
import cv2
try:
    import detecttools  # NOQA
except ImportError:
    sys.path.append(expanduser('~/code'))
    try:
        import detecttools  # NOQA
    except ImportError:
        print('Cannot find detecttools!')
        raise
import detecttools.ctypes_interface as ctypes_interface
from detecttools.ibeisdata import IBEIS_Data
#from detecttools.directory import Directory
#from os import makedirs
#from PIL import Image
#import random
#import xml.etree.ElementTree as xml
# https://github.com/bluemellophone/detecttools


# JONCOMMENT: join will ignore the preceding arguments if any of the next
# arguments are absolute paths. the os.getcwd() does noting here
#__LOCATION__ = realpath(join(os.getcwd(), dirname(__file__)))
__LOCATION__ = realpath(dirname(__file__))


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


"""
    This defines the default constructor parameters for the algorithm.
    These values may be overwritten by passing in a dictionary to the
    class constructor using kwargs

    constructor_parameters = [
        (parameter type, parameter name, parameter default value),
    ]

    IMPORTANT:
    The order of this list must match the C++ constructor parameter order
"""
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

def rmtreedir(path):
    if isdir(path):
        shutil.rmtree(path)


def ensuredir(path):
    if isdir(path):
        os.makedirs(path)


def _kwargs(kwargs, key, value):
    if key not in kwargs:
        kwargs[key] = value


def _build_shared_c_library(rebuild=False):
    if rebuild:
        rmtreedir(join(__LOCATION__, 'build'))

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

        filename = join(directory_path, image.filename)

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
                    # Resize
                    temp = cv2.resize(temp, (_width, _height),
                                        interpolation=cv2.INTER_LANCZOS4)
                    width = _width
                    height = _height

                xmax = width
                xmin = 0
                ymax = height
                ymin = 0

                if positive:
                    postfix = ' %d %d %d %d %d %d' % (xmin, ymin, xmax, ymax,
                                                        xmin + width / 2,
                                                        ymin + height / 2)
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


def get_training_data_from_ibeis(dataset_path, category, pos_path, neg_path,
                                 val_path, test_path, **kwargs):

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

        #Load the compiled lib and defines its functions
        root_dir = realpath(dirname(__file__))
        rf.CLIB, LOAD_FUNCTION = ctypes_interface.load_clib(libname, root_dir)

        """
        def_lib_func is used to expose the Python bindings that are declared
        inside the .cpp files to the Python clib object.

        def_lib_func(return type, function name, list of parameter types)

        IMPORTANT:
        For functions that return void, use Python None as the return value.
        For functions that take no parameters, use the Python empty list [].
        """
        LOAD_FUNCTION(COBJ, 'constructor',      PARAM_TYPES)
        LOAD_FUNCTION(None, 'train',            [COBJ, CCHAR, CINT, CCHAR, CCHAR])
        LOAD_FUNCTION(CINT, 'detect',           [COBJ, COBJ, CCHAR, CCHAR, CBOOL, CBOOL, CBOOL, CINT, CINT, CFLOAT, CFLOAT, CFLOAT, CINT])
        LOAD_FUNCTION(None, 'detect_results',   [COBJ, CNPFLOAT])
        LOAD_FUNCTION(None, 'segment',          [COBJ])
        LOAD_FUNCTION(COBJ, 'load',             [COBJ, CCHAR, CCHAR, CINT])
        LOAD_FUNCTION(None, 'save',             [COBJ])
        # Add any algorithm-specific functions here

        """
        Create the C object using the default parameter values and any updated
        parameter values from kwargs
        """
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

    def train(rf, database_path, category, pos_path, neg_path,
              val_path, test_path, tree_path, **kwargs):
        _kwargs(kwargs, 'num_trees', 10)

        print('[rf] Clearing Test Cache Directories')
        rmtreedir(pos_path)
        rmtreedir(neg_path)
        rmtreedir(val_path)
        rmtreedir(test_path)

        print('[rf] Creating Test Cache Directories')
        ensuredir(pos_path)
        ensuredir(neg_path)
        ensuredir(val_path)
        ensuredir(test_path)
        ensuredir(trees_path)

        # Gather training data from IBEIS database
        fpath_pos, fpath_neg = get_training_data_from_ibeis(
            database_path, category, pos_path, neg_path, val_path,
            test_path, **kwargs)

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
    pos_path    = join('results', category, 'train-positives')
    neg_path    = join('results', category, 'train-negatives')
    val_path    = join('results', category, 'val')
    test_path   = join('results', category, 'test')
    detect_path = join('results', category, 'detect')
    trees_path  = join('results', category, 'trees')
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

    # _trees_path = join(trees_path, tree_prefix)
    # detector.train(dataset_path, category, pos_path, neg_path, val_path, test_path, _trees_path, **train_config)

    #=================================
    # Detect using Random Forest
    #=================================

    print('[rf] Clearing Detect Cache Directories')
    rmtreedir(detect_path)

    print('[rf] Creating Detect Cache Directories')
    ensuredir(detect_path)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, tree_prefix)

    # Get input images
    test_file = open(test_path + '.txt', 'r')
    test_file.readline()
    files = [line.strip() for line in test_file ]

    for i in range(len(files)):
        src_fpath = files[i]
        dst_fpath = join(detect_path, files[i].split('/')[-1])

        results, timing = detector.detect(forest, src_fpath, dst_fpath,
                                          **detect_config)

        print('[rf] %s | Time: %.3f' % (src_fpath, timing))
        print(results)
