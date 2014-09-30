from __future__ import absolute_import, division, print_function
# Standard
#import time
import sys
from six.moves import zip
import threading
from os.path import dirname, realpath
from collections import OrderedDict as odict
import ctypes as C
# Scientific
import numpy as np
from os.path import join
import detecttools.ctypes_interface as ctypes_interface
from .pyrf_helpers import (ensuredir, rmtreedir, get_training_data_from_ibeis,
                           _build_shared_c_library)


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

# hesaff style
str_t     = C.c_char_p
int_t     = C.c_int
byte_t    = C.c_char
float_t   = C.c_float
str_list_t   = C.POINTER(str_t)
int_array_t  = np.ctypeslib.ndpointer(dtype=int_t, ndim=1, flags=FLAGS_RW)
float_array_t  = np.ctypeslib.ndpointer(dtype=float_t, ndim=1, flags=FLAGS_RW)

results_dtype   = np.float32
results_t       = np.ctypeslib.ndpointer(dtype=results_dtype, ndim=2, flags=FLAGS_RW)
results_array_t = np.ctypeslib.ndpointer(dtype=results_t, ndim=1, flags=FLAGS_RW)


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


DETECT_PARAM_TYPES = [CBOOL, CBOOL, CBOOL, CINT, CINT, CFLOAT, CFLOAT, CFLOAT,
                      CINT]


def load_pyrf_clib(rebuild=False):
    """ Loads the pyrf dynamic library and defines its functions """
    #if VERBOSE:
    #    print('[rf] Testing Random_Forest')
    if rebuild:
        _build_shared_c_library(rebuild)
    # FIXME: This will break on packaging
    root_dir = realpath(dirname(__file__))
    libname = 'pyrf'
    rf_clib, def_cfunc = ctypes_interface.load_clib(libname, root_dir)
    """
    def_lib_func is used to expose the Python bindings that are declared
    inside the .cpp files to the Python clib object.

    def_lib_func(return type, function name, list of parameter types)

    IMPORTANT:
    For functions that return void, use Python None as the return value.
    For functions that take no parameters, use the Python empty list [].
    """
    def_cfunc(COBJ, 'constructor',      PARAM_TYPES)
    def_cfunc(None, 'train',            [COBJ, CCHAR, CINT, CCHAR, CCHAR])
    def_cfunc(CINT, 'detect',           [COBJ, COBJ, CCHAR, CCHAR] + DETECT_PARAM_TYPES )
    def_cfunc(None, 'detect_many',      [COBJ, COBJ, int_t, str_list_t, str_list_t,
                                         int_array_t, results_array_t] + DETECT_PARAM_TYPES)
    def_cfunc(None, 'detect_results',   [COBJ, CNPFLOAT])
    def_cfunc(None, 'segment',          [COBJ])
    def_cfunc(COBJ, 'load',             [COBJ, CCHAR, CCHAR, CINT])
    def_cfunc(None, 'save',             [COBJ])
    return rf_clib


# Load the dnamic library at module load time
RF_CLIB = load_pyrf_clib()


def _new_pyrf(**kwargs):
    """ Create the C object using the default parameter values and any updated
    parameter values from kwargs """
    param_odict = PARAM_ODICT.copy()
    param_odict.update(kwargs)

    print('[rf] New Random_Forest Object Created')
    print('[rf] Algorithm Settings=%r' % (param_odict,))

    param_values = param_odict.values()  # pass all parameters to the C constructor
    pyrf_ptr = RF_CLIB.constructor(*param_values)
    return pyrf_ptr


def _kwargs(kwargs, key, value):
    if key not in kwargs:
        kwargs[key] = value


def _cast_strlist_to_C(py_strlist):
    """
    Converts a python list of strings into a c array of strings
    adapted from "http://stackoverflow.com/questions/3494598/passing-a-list-of
    -strings-to-from-python-ctypes-to-c-function-expecting-char"
    Avi's code
    """
    c_strarr = (str_t * len(py_strlist))()
    c_strarr[:] = py_strlist
    return c_strarr


def arrptr_to_np(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy
    Input:
        c_arrpt - an array pointer returned from C
        shape   - shape of that array pointer
        arr_t   - the ctypes datatype of c_arrptr
    Avi's code
    """
    arr_t_size = C.POINTER(byte_t * dtype().itemsize)  # size of each item
    c_arr = C.cast(c_arrptr.astype(int), arr_t_size)   # cast to ctypes
    np_arr = np.ctypeslib.as_array(c_arr, shape)       # cast to numpy
    np_arr.dtype = dtype                               # fix numpy dtype
    return np_arr


def extract_2darr_list(size_list, ptr_list, arr_t, arr_dtype,
                        arr_dim):
    """
    size_list - contains the size of each output 2d array
    ptr_list  - an array of pointers to the head of each output 2d
                array (which was allocated in C)
    arr_t     - the C pointer type
    arr_dtype - the numpy array type
    arr_dim   - the number of columns in each output 2d array
    """
    arr_list = [arrptr_to_np(arr_ptr, (size, arr_dim), arr_t, arr_dtype)
                    for (arr_ptr, size) in zip(ptr_list, size_list)]
    return arr_list


class Random_Forest_Detector(object):
    #TODO: Picklize me!
    #dump(object, file)
    #dumps(object) -> string
    #load(file) -> object
    #loads(string) -> object

    #=============================
    # Algorithm Constructor
    #=============================
    def __init__(rf, **kwargs):
        rf.pyrf_ptr = _new_pyrf(**kwargs)
        rf.detect_params = None  # must be set before running detection

    def set_detect_params(rf, **kwargs):
        # Default params
        default_param_dict = odict([
            ('save_detection_images',   False),
            ('save_scales',             False),
            ('draw_supressed',          False),
            ('detection_width',         128),
            ('detection_height',        80),
            ('percentage_left',         0.50),
            ('percentage_top',          0.50),
            ('nms_margin_percentage',   0.75),
            ('min_contour_area',        300),
        ])
        default_param_dict.update(kwargs)
        rf.detect_params = default_param_dict.values()

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
              val_path, test_path, trees_path, **kwargs):
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
        trees_path = join(trees_path, category + '-')

        # Gather training data from IBEIS database
        fpath_pos, fpath_neg, fpath_val, fpath_test = get_training_data_from_ibeis(
            database_path, category, pos_path, neg_path, val_path,
            test_path, **kwargs)

        # Run training algorithm
        params = [rf.pyrf_ptr, trees_path, kwargs['num_trees'], fpath_pos, fpath_neg]
        rf._run(RF_CLIB.train, params)

    def retrain(rf):
        rf._run(RF_CLIB.retrain, [rf.pyrf_ptr])

    #=============================
    # Run Algorithm
    #=============================

    def detect_many(rf, forest, image_fpath_list, result_fpath_list):
        """ WIP """
        OPENMP_SOLUTION = '--pyrf-openmp' in sys.argv
        if OPENMP_SOLUTION:
            # OPENMP SOLUTION
            nImgs = len(image_fpath_list)
            c_src_strs = _cast_strlist_to_C(map(realpath, image_fpath_list))
            c_dst_strs = _cast_strlist_to_C(map(realpath, result_fpath_list))
            results_ptr_arr = np.empty(nImgs, results_t)  # outvar of results ptrs
            length_arr = np.empty(nImgs, int_t)  # outvar of lengths
            # Execute batch detection
            RF_CLIB.detect_many(
                rf.pyrf_ptr,
                forest,
                nImgs,
                c_src_strs,
                c_dst_strs,
                length_arr,
                results_ptr_arr,
                *rf.detect_params)

            results_list = extract_2darr_list(length_arr, results_ptr_arr, results_t, results_dtype, 8)

            # Finish getting results using lengths and heads of arrays
            #results_list = [arrptr_to_np(results_ptr, (len_, 8), results_t,
            #                             np.float32)
            #                for (results_ptr, len_) in zip(results_ptr_arr, length_arr)]
        else:
            # FOR LOOP SOLUTION
            results_list = []
            for image_fpath, result_fpath in zip(image_fpath_list, result_fpath_list):
                # Execute detection
                length = RF_CLIB.detect(
                    rf.pyrf_ptr,
                    forest,
                    image_fpath,
                    result_fpath,
                    *rf.detect_params)
                # Read results
                results = np.empty((length, 8), np.float32)
                RF_CLIB.detect_results(rf.pyrf_ptr, results)
                results_list.append(results)
        return results_list

    def detect(rf, forest, image_fpath, result_fpath):
        # Removed to simplify inner loop. User a Timer object around this instead.
        #start = time.time()

        # Removed because this will be interpreted as individual
        # function calls, which is not very python efficient
        #_kwargs(kwargs, 'save_detection_images',   False)
        #_kwargs(kwargs, 'save_scales',             False)
        #_kwargs(kwargs, 'draw_supressed',          False)
        #_kwargs(kwargs, 'detection_width',         128)
        #_kwargs(kwargs, 'detection_height',        80)
        #_kwargs(kwargs, 'percentage_left',         0.50)
        #_kwargs(kwargs, 'percentage_top',          0.50)
        #_kwargs(kwargs, 'nms_margin_percentage',   0.75)
        #_kwargs(kwargs, 'min_contour_area',        300)

        # Execute detection
        length = RF_CLIB.detect(
            rf.pyrf_ptr,
            forest,
            image_fpath,
            result_fpath,
            *rf.detect_params)
        #    kwargs['save_detection_images'],
        #    kwargs['save_scales'],
        #    kwargs['draw_supressed'],
        #    kwargs['detection_width'],
        #    kwargs['detection_height'],
        #    kwargs['percentage_left'],
        #    kwargs['percentage_top'],
        #    kwargs['nms_margin_percentage'],
        #    kwargs['min_contour_area'],
        #)

        # Read results
        results = np.empty((length, 8), np.float32)
        RF_CLIB.detect_results(rf.pyrf_ptr, results)

        #done = time.time()
        return results  # , done - start

    def segment(rf):
        rf._run(RF_CLIB.segment, [rf.pyrf_ptr])

    #=============================
    # Load / Save Trained Data
    #=============================

    def load(rf, tree_path, prefix, num_trees=10):
        return RF_CLIB.load(rf.pyrf_ptr, tree_path, prefix, num_trees)

    def save(rf):
        rf._run(RF_CLIB.save, [rf.pyrf_ptr])
