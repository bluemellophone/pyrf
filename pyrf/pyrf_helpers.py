#============================
# Python Interface
#============================
from __future__ import absolute_import, division, print_function
from os.path import join, realpath, dirname
import cv2
import random
import numpy as np
import ctypes as C
import detecttools.ctypes_interface as ctypes_interface


def _cast_list_to_c(py_list, dtype):
    """
    Converts a python list of strings into a c array of strings
    adapted from "http://stackoverflow.com/questions/3494598/passing-a-list-of
    -strings-to-from-python-ctypes-to-c-function-expecting-char"
    Avi's code
    """
    c_arr = (dtype * len(py_list))()
    c_arr[:] = py_list
    return c_arr


def _arrptr_to_np(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy
    Input:
        c_arrpt - an array pointer returned from C
        shape   - shape of that array pointer
        arr_t   - the ctypes datatype of c_arrptr
    Avi's code
    """
    arr_t_size = C.POINTER(C.c_char * dtype().itemsize)           # size of each item
    c_arr = C.cast(c_arrptr.astype(int), arr_t_size)              # cast to ctypes
    np_arr = np.ctypeslib.as_array(c_arr, shape)                  # cast to numpy
    np_arr.dtype = dtype                                          # fix numpy dtype
    np_arr = np.require(np_arr, dtype=dtype, requirements=['O'])  # prevent memory leaks
    return np_arr


def _extract_np_array(size_list, ptr_list, arr_t, arr_dtype,
                        arr_dim):
    """
    size_list - contains the size of each output 2d array
    ptr_list  - an array of pointers to the head of each output 2d
                array (which was allocated in C)
    arr_t     - the C pointer type
    arr_dtype - the numpy array type
    arr_dim   - the number of columns in each output 2d array
    """
    arr_list = [_arrptr_to_np(arr_ptr, (size, arr_dim), arr_t, arr_dtype)
                    for (arr_ptr, size) in zip(ptr_list, size_list)]
    return arr_list


def _load_c_shared_library(METHODS):
    ''' Loads the pyrf dynamic library and defines its functions '''
    root_dir = realpath(join('..', dirname(__file__)))
    libname = 'pyrf'
    rf_clib, def_cfunc = ctypes_interface.load_clib(libname, root_dir)
    # Load and expose methods from lib
    for method in METHODS.keys():
        def_cfunc(METHODS[method][1], method, METHODS[method][0])
    return rf_clib


def _cache_data(src_path_list, dst_path, format_str='data_%07d.JPEG', **kwargs):
    '''
        src_path_list                    (required)
        dst_path                         (required)
        chips_norm_width                 (required)
        chips_norm_height                (required)
        chips_prob_flip_horizontally     (required)
        chips_prob_flip_vertically       (required)
    '''
    if kwargs['chips_norm_width'] is not None:
        kwargs['chips_norm_width'] = int(kwargs['chips_norm_width'])
    if kwargs['chips_norm_height'] is not None:
        kwargs['chips_norm_height'] = int(kwargs['chips_norm_height'])
    chip_filename_list = []
    counter = 0
    for src_path in src_path_list:
        if kwargs['verbose']:
            print("Processing %r" % (src_path, ))
        # Load the iamge
        image = cv2.imread(src_path)
        # Get the shape of the iamge
        height_, width_, channels_ = image.shape
        # Determine new image size
        if kwargs['chips_norm_width'] is not None and kwargs['chips_norm_height'] is None:
            # Normalizing width (with respect to aspect ratio)
            width  = kwargs['chips_norm_width']
            height = int( ( width / width_ ) * height_ )
        elif kwargs['chips_norm_height'] is not None and kwargs['chips_norm_width'] is None:
            # Normalizing height (with respect to aspect ratio)
            height = kwargs['chips_norm_height']
            width  = int( ( height / height_ ) * width_ )
        elif kwargs['chips_norm_width'] is not None and kwargs['chips_norm_height'] is not None:
            # Normalizing width and height (ignoring aspect ratio)
            width  = kwargs['chips_norm_width']
            height = kwargs['chips_norm_height']
        else:
            width  = width_
            height = height_
        # Check for patch size limitation
        if width < kwargs['patch_width'] or height < kwargs['patch_height']:
            print('\t[WARNING] Image size is too small for the patch size, skipping image ')
            continue
        # Resize the image
        image_ = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        # Flip the image (if nessicary)
        if kwargs['chips_prob_flip_horizontally'] is not None and random.uniform(0.0, 1.0) <= kwargs['chips_prob_flip_horizontally']:
            image_ = cv2.flip(image_, 1)
        if kwargs['chips_prob_flip_vertically']   is not None and random.uniform(0.0, 1.0) <= kwargs['chips_prob_flip_vertically']:
            image_ = cv2.flip(image_, 0)
        # Get the images destination filename
        chip_filename = format_str % (counter, )
        # Write the iamge
        cv2.imwrite(join(dst_path, chip_filename), image_)
        # Append the image's destaintion filename to the return list
        chip_filename_list.append(chip_filename)
        # Increment the counter
        counter += 1
    return chip_filename_list
