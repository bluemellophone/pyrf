from __future__ import absolute_import, division, print_function
# Standard
from collections import OrderedDict as odict
import multiprocessing
import ctypes as C
# Scientific
import utool as ut
import numpy as np
import time
from os.path import join, exists, abspath, isdir
import shutil
from detecttools.directory import Directory
from pyrf.pyrf_helpers import (_load_c_shared_library, _cast_list_to_c, _cache_data, _extract_np_array)


VERBOSE_RF = ut.get_argflag('--verbrf') or ut.VERBOSE
QUIET_RF   = ut.get_argflag('--quietrf') or ut.QUIET


#============================
# CTypes Interface Data Types
#============================
'''
    Bindings for C Variable Types
'''
NP_FLAGS       = 'aligned, c_contiguous, writeable'
# Primatives
C_OBJ          = C.c_void_p
C_BYTE         = C.c_char
C_CHAR         = C.c_char_p
C_INT          = C.c_int
C_BOOL         = C.c_bool
C_FLOAT        = C.c_float
NP_INT8        = np.uint8
NP_FLOAT32     = np.float32
# Arrays
C_ARRAY_CHAR   = C.POINTER(C_CHAR)
C_ARRAY_FLOAT  = C.POINTER(C_FLOAT)
NP_ARRAY_INT   = np.ctypeslib.ndpointer(dtype=C_INT,          ndim=1, flags=NP_FLAGS)
NP_ARRAY_FLOAT = np.ctypeslib.ndpointer(dtype=NP_FLOAT32,     ndim=2, flags=NP_FLAGS)
RESULTS_ARRAY  = np.ctypeslib.ndpointer(dtype=NP_ARRAY_FLOAT, ndim=1, flags=NP_FLAGS)


#=================================
# Method Parameter Types
#=================================
'''
IMPORTANT:
    For functions that return void, use Python None as the return value.
    For functions that take no parameters, use the Python empty list [].
'''

METHODS = {}
METHODS['init'] = ([
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], C_OBJ)

METHODS['forest'] = ([
    C_OBJ,           # detector
    C_ARRAY_CHAR,    # tree_path_array
    C_INT,           # _tree_path_num
    C_BOOL,          # serial
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], C_OBJ)

METHODS['train'] = ([
    C_OBJ,           # detector
    C_CHAR,          # train_pos_chip_path
    C_ARRAY_CHAR,    # train_pos_chip_filename_array
    C_INT,           # _train_pos_chip_num
    C_CHAR,          # train_neg_chip_path
    C_ARRAY_CHAR,    # train_neg_chip_filename_array
    C_INT,           # _train_neg_chip_num
    C_CHAR,          # trees_path
    C_INT,           # patch_width
    C_INT,           # patch_height
    C_FLOAT,         # patch_density
    C_INT,           # trees_num
    C_INT,           # trees_offset
    C_INT,           # trees_max_depth
    C_INT,           # trees_max_patches
    C_INT,           # trees_leaf_size
    C_INT,           # trees_pixel_tests
    C_FLOAT,         # trees_prob_optimize_mode
    C_BOOL,          # serial
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], None)

METHODS['detect'] = ([
    C_OBJ,           # detector
    C_OBJ,           # forest
    C_ARRAY_CHAR,    # input_gpath_array
    C_INT,           # _input_gpath_num
    C_ARRAY_CHAR,    # output_gpath_array
    C_ARRAY_CHAR,    # output_scale_gpath_array
    C_INT,           # mode
    C_FLOAT,         # sensitivity
    C_ARRAY_FLOAT,   # scale_array
    C_INT,           # _scale_num
    C_INT,           # nms_min_area_contour
    C_FLOAT,         # nms_min_area_overlap
    RESULTS_ARRAY,   # results_val_array
    NP_ARRAY_INT,    # results_len_array
    C_INT,           # RESULT_LENGTH
    C_BOOL,          # serial
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], None)
RESULT_LENGTH = 8

#=================================
# Load Dynamic Library
#=================================
RF_CLIB = _load_c_shared_library(METHODS)


#=================================
# Random Forest Detector
#=================================
class Random_Forest_Detector(object):

    def __init__(rf, verbose=VERBOSE_RF, quiet=QUIET_RF):
        '''
            Create the C object for the PyRF detector.

            Args:
                verbose (bool, optional): verbose flag; defaults to --verbrf flag

            Returns:
                detector (object): the Random Forest Detector object
        '''
        rf.verbose = verbose
        rf.quiet = quiet
        if rf.verbose and not rf.quiet:
            print('[pyrf py] New Random_Forest Object Created')
        rf.detector_c_obj = RF_CLIB.init(rf.verbose, rf.quiet)

    def forest(rf, tree_path_list, **kwargs):
        '''
            Create the forest object by loading a list of tree paths.

            Args:
                tree_path_list (list of str): list of tree paths as strings
                serial (bool, optional): flag to signify if to load the forest in serial;
                    defaults to False
                verbose (bool, optional): verbose flag; defaults to object's verbose or
                    selectively enabled for this function

            Returns:
                forest (object): the forest object of the loaded trees
        '''
        # Default values
        params = odict([
            ('serial',                       False),
            ('verbose',                      rf.verbose),
            ('quiet',                        rf.quiet),
        ])
        params.update(kwargs)

        # Data integrity
        assert len(tree_path_list) > 0, \
            'Must specify at least one tree path to load'
        assert all( [ exists(tree_path) for tree_path in tree_path_list ] ), \
            'At least one specified tree path does not exist'

        params_list = [
            _cast_list_to_c(tree_path_list, C_CHAR),
            len(tree_path_list),
        ] + params.values()
        return RF_CLIB.forest(rf.detector_c_obj, *params_list)

    def train_folder(rf, train_pos_path, train_neg_path, trees_path, **kwargs):
        direct = Directory(train_pos_path, include_file_extensions='images')
        train_pos_cpath_list = direct.files()
        direct = Directory(train_neg_path, include_file_extensions='images')
        train_neg_cpath_list = direct.files()
        return rf.train(train_pos_cpath_list, train_neg_cpath_list, trees_path, **kwargs)

    def train(rf, train_pos_cpath_list, train_neg_cpath_list, trees_path, **kwargs):
        '''
            Train a new forest with the given positive chips and negative chips.

            Args:
                train_pos_chip_path_list (list of str): list of positive training chips
                train_neg_chip_path_list (list of str): list of negative training chips
                trees_path (str): string path of where the newly trained trees are to be saved

            Kwargs:
                chips_norm_width (int, optional): Chip normalization width for resizing;
                    the chip is resized to have a width of chips_norm_width and
                    whatever resulting height in order to best match the original
                    aspect ratio; defaults to 128

                    If both chips_norm_width and chips_norm_height are specified,
                    the original aspect ratio of the chip is not respected
                chips_norm_height (int, optional): Chip normalization height for resizing;
                    the chip is resized to have a height of chips_norm_height and
                    whatever resulting width in order to best match the original
                    aspect ratio; defaults to None

                    If both chips_norm_width and chips_norm_height are specified,
                    the original aspect ratio of the chip is not respected
                chips_prob_flip_horizontally (float, optional): The probability
                    that a chips is flipped horizontally before training to make
                    the training set invariant to horizontal flips in the image;
                    defaults to 0.5; 0.0 <= chips_prob_flip_horizontally <= 1.0
                chips_prob_flip_vertically (float, optional): The probability
                    that a chips is flipped vertivcally before training to make
                    the training set invariant to vertical flips in the image;
                    defaults to 0.5; 0.0 <= chips_prob_flip_vertically <= 1.0
                patch_width (int, optional): the width of the patches for extraction
                    in the tree; defaults to 32; patch_width > 0
                patch_height (int, optional): the height of the patches for extraction
                    in the tree; defaults to 32; patch_height > 0
                patch_density (float, optional): the number of patches to extract from
                    each chip as a function of density; the density is calculated as:
                        samples = patch_density * [(chip_width * chip_height) / (patch_width * patch_height)]
                    and specifies how many times a particular pixel is sampled
                    from the chip; defaults to 4.0; patch_density > 0
                trees_num (int, optional): the number of trees to train in parallel;
                    defaults to 10
                trees_offset (int, optional): the tree number that begins the sequence
                    of when a tree is trained; defaults to None

                    If None is specified, the trees_offset value is automatically guessed
                    by using the number of files in trees_path

                    Tree model files are overwritten if the offset has overlap with
                    previouly generated trees
                trees_max_depth (int, optional): the maximum depth of the tree during
                    training, this can used for regularization; defaults to 16
                trees_max_patches (int, optional): the maximum number of patches that
                    should be extracted for training between positives AND negatives
                    (the detector attempts to balance between the number of positive
                    and negative patches to be roughly the same in quantity);
                    defaults to 64000
                trees_leaf_size (int, optional): the number of patches in a node that
                    specifies the threshold for becoming a leaf; defaults to 20

                    A node becomes a leaf under two conditions:
                        1.) The maximum tree depth has been reached (trees_max_depth)
                        2.) The patches in the node is less than trees_leaf_size and
                            is stopped prematurely
                trees_pixel_tests (int, optional): the number of pixel tests to perform
                    at each node; defaults to 2000
                trees_prob_optimize_mode (float, optional): The probability of the
                    tree optimizing between classification and regression; defaults to
                    0.5
                serial (bool, optional): flag to signify if to run training in serial;
                    defaults to False
                verbose (bool, optional): verbose flag; defaults to object's verbose or
                    selectively enabled for this function

            Returns:
                None
        '''
        # Default values
        params = odict([
            ('chips_norm_width',             128),
            ('chips_norm_height',            None),
            ('chips_prob_flip_horizontally', 0.5),
            ('chips_prob_flip_vertically',   0.0),
            ('patch_width',                  32),
            ('patch_height',                 32),
            ('patch_density',                4.0),
            ('trees_num',                    10),
            ('trees_offset',                 None),
            ('trees_max_depth',              16),
            ('trees_max_patches',            64000),
            ('trees_leaf_size',              20),
            ('trees_pixel_tests',            10000),
            ('trees_prob_optimize_mode',     0.5),
            ('serial',                       False),
            ('verbose',                      rf.verbose),
            ('quiet',                        rf.quiet),
        ])
        params.update(kwargs)
        # Make the tree path absolute
        trees_path = abspath(trees_path)

        # cout << "AIM FOR A SPLIT OF 24k - 32k POSITIVE & NEGATIVE PATCHES EACH FOR GOOD REGULARIZATION AT DEPTH 16" << endl;

        # Ensure the trees_path exists
        ut.ensuredir(trees_path)
        data_path = join(trees_path, 'data')
        if isdir(data_path):
            shutil.rmtree(data_path)
        ut.ensuredir(data_path)
        data_path_pos = join(data_path, 'pos')
        ut.ensuredir(data_path_pos)
        data_path_neg = join(data_path, 'neg')
        ut.ensuredir(data_path_neg)

        # Try to figure out the correct tree offset
        if params['trees_offset'] is None:
            direct = Directory(trees_path, include_file_extensions=['txt'])
            params['trees_offset'] = len(direct.files()) + 1
            if not params['quiet']:
                print('[pyrf py] Auto Tree Offset: %d' % params['trees_offset'])

        # Data integrity
        assert params['chips_norm_width'] is None or params['chips_norm_width'] >= params['patch_width'], \
            'Normalization width too small for patch width'
        assert params['chips_norm_height'] is None or params['chips_norm_height'] >= params['patch_height'], \
            'Normalization height too small for patch height'
        assert params['patch_width'] > 0, \
            'Patch width must be positive'
        assert params['patch_height'] > 0, \
            'Patch height must be positive'
        assert params['patch_density'] > 0.0, \
            'Patch density must be positive'
        assert 0.0 <= params['chips_prob_flip_horizontally'] and params['chips_prob_flip_horizontally'] <= 1.0, \
            'Horizontal flip probability must be between 0 and 1'
        assert 0.0 <= params['chips_prob_flip_vertically'] and params['chips_prob_flip_vertically'] <= 1.0, \
            'Vertical flip probability must be between 0 and 1'
        assert params['trees_num'] > 0, \
            'Number of trees must be positive'
        assert params['trees_offset'] >= 0, \
            'Number of trees must be non-negative'
        assert params['trees_max_depth'] > 1, \
            'Tree depth must be greater than 1'
        assert params['trees_max_patches'] % 2 == 0 and params['trees_max_patches'] > 0, \
            'A tree must have an even (positive) number of patches'
        assert 0.0 <= params['trees_prob_optimize_mode'] and params['trees_prob_optimize_mode'] <= 1.0, \
            'Tree optimization mode probability must be between 0 and 1 (inclusive)'
        assert all( [ exists(train_pos_cpath) for train_pos_cpath in train_pos_cpath_list ] ), \
            'At least one specified positive chip path does not exist'
        assert all( [ exists(train_neg_cpath) for train_neg_cpath in train_neg_cpath_list ] ), \
            'At least one specified positive chip path does not exist'
        # We will let the C++ code perform the patch size checks

        if not params['quiet']:
            print('[pyrf py] Caching positives into %r' % (data_path_pos, ))
        train_pos_chip_filename_list = _cache_data(train_pos_cpath_list, data_path_pos, **params)

        if not params['quiet']:
            print('[pyrf py] Caching negatives into %r' % (data_path_neg, ))
        train_neg_chip_filename_list = _cache_data(train_neg_cpath_list, data_path_neg, **params)

        # We no longer need these parameters (and they should not be transferred to the C++ library)
        del params['chips_norm_width']
        del params['chips_norm_height']
        del params['chips_prob_flip_horizontally']
        del params['chips_prob_flip_vertically']

        # Run training algorithm
        params_list = [
            data_path_pos,
            _cast_list_to_c(train_pos_chip_filename_list, C_CHAR),
            len(train_pos_chip_filename_list),
            data_path_neg,
            _cast_list_to_c(train_neg_chip_filename_list, C_CHAR),
            len(train_neg_chip_filename_list),
            trees_path,
        ] + params.values()
        RF_CLIB.train(rf.detector_c_obj, *params_list)
        if not params['quiet']:
            print('\n\n[pyrf py] *************************************')
            print('[pyrf py] Training Completed')

    def detect(rf, forest, input_gpath_list, **kwargs):
        '''
            Run detection with a given loaded forest on a list of images

            Args:
                forest (object): the forest obejct that you want to use during
                    detection
                input_gpath_list (list of str): the list of image paths that you want
                    to test

            Kwargs:
                output_gpath_list (list of str, optional): the paralell list of output
                    image paths for detection debugging or results; defaults to None

                    When this list is None no images are outputted for any test
                    images, whereas the list can be a parallel list where some values
                    are strings and others are None
                output_scale_gpath_list (list of str, optional): the paralell list of output
                    scale image paths for detection debugging or results; defaults
                    to None

                    When this list is None no images are outputted for any test
                    images, whereas the list can be a parallel list where some values
                    are strings and others are None
                mode (int, optional): the mode that the detector outputs; detaults to 0
                    0 - Hough Voting - the output is a Hough image that predicts the
                        locations of the obejct centeroids
                    0 - Classification Map - the output is a classification probability
                        map across the entire image where no regression information
                        is utilized
                sensitivity (float, optional): the sensitivity of the detector;

                        mode = 0 - defaults to 128.0
                        mode = 1 - defaults to 255.0

                scale_list (list of float, optional): the list of floats that specifies the scales
                    to try during testing;
                    defaults to [1.0, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10]

                        scale > 1.0 - Upscale the image
                        scale = 1.0 - Original image size
                        scale < 1.0 - Downscale the image

                    The list of scales highly impacts the performance of the detector and
                    should be carefully chosen

                    The scales are applied to BOTH the width and the height of the image
                    in order to scale the image and an interpolation of OpenCV's
                    CV_INTER_LANCZOS4 is used
                batch_size (int, optional): the number of images to test at a single
                    time in paralell (if None, the number of CPUs is used); defaults to None
                nms_min_area_contour (int, optional): the minimum size of a centroid
                    candidate region; defaults to 300
                nms_min_area_overlap (float, optional, DEPRICATED): the allowable overlap in
                    bounding box predictions; defaults to 0.75
                serial (bool, optional): flag to signify if to run detection in serial;

                        len(input_gpath_list) >= batch_size - defaults to False
                        len(input_gpath_list) <  batch_size - defaults to False

                verbose (bool, optional): verbose flag; defaults to object's verbose or
                    selectively enabled for this function

            Yields:
                (str, (list of dict)): tuple of the input image path and a list
                    of dictionaries specifying the detected bounding boxes

                    The dictionaries returned by this function are of the form:
                        centerx (int): the x position of the object's centroid

                            Note that the center of the bounding box and the location of
                            the object's centroid can be different
                        centery (int): the y position of the obejct's centroid

                            Note that the center of the bounding box and the location of
                            the object's centroid can be different
                        xtl (int): the top left x position of the bounding box
                        ytl (int): the top left y position of the bounding box
                        width (int): the width of the bounding box
                        height (int): the hiehgt of the bounding box
                        confidence (float): the confidence that this bounding box is of
                            the class specified by the trees used during testing
                        suppressed (bool, DEPRICATED): the flag of if this bounding
                            box has been marked to be suppressed by the detection
                            algorithm

        '''
        # Default values
        params = odict([
            ('output_gpath_list',            None),
            ('output_scale_gpath_list',      None),
            ('mode',                         0),
            ('sensitivity',                  None),
            ('scale_list',                   [1.0, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10]),
            ('_scale_num',                   None),  # This value always gets overwritten
            ('batch_size',                   None),
            ('nms_min_area_contour',         100),
            ('nms_min_area_overlap',         0.75),
            ('results_val_array',            None),  # This value always gets overwritten
            ('results_len_array',            None),  # This value always gets overwritten
            ('RESULT_LENGTH',                None),  # This value always gets overwritten
            ('serial',                       False),
            ('verbose',                      rf.verbose),
            ('quiet',                        rf.quiet),
        ])
        params.update(kwargs)
        params['RESULT_LENGTH'] = RESULT_LENGTH
        output_gpath_list = params['output_gpath_list']
        output_scale_gpath_list = params['output_scale_gpath_list']
        # We no longer want these parameters in params
        del params['output_gpath_list']
        del params['output_scale_gpath_list']

        if params['sensitivity'] is None:
            assert params['mode'] in [0, 1], 'Invalid mode provided'
            if params['mode'] == 0:
                params['sensitivity'] = 128.0
            elif params['mode'] == 1:
                params['sensitivity'] = 255.0

        # Try to determine the parallel processing batch size
        if params['batch_size'] is None:
            try:
                cpu_count = multiprocessing.cpu_count()
                if not params['quiet']:
                    print('[pyrf py] Detecting with %d CPUs' % (cpu_count, ))
                params['batch_size'] = cpu_count
            except:
                params['batch_size'] = 8

        # To eleminate downtime, add 1 to batch_size
        # params['batch_size'] +=

        # Data integrity
        assert params['mode'] >= 0, \
            'Detection mode must be non-negative'
        assert 0.0 <= params['sensitivity'], \
            'Sensitivity must be non-negative'
        assert len(params['scale_list']) > 0 , \
            'The scale list cannot be empty'
        assert all( [ scale > 0.0 for scale in params['scale_list'] ]), \
            'All scales must be positive'
        assert params['batch_size'] > 0, \
            'Batch size must be positive'
        assert params['nms_min_area_contour'] > 0, \
            'Non-maximum suppression minimum contour area cannot be negative'
        assert 0.0 <= params['nms_min_area_overlap'] and params['nms_min_area_overlap'] <= 1.0, \
            'Non-maximum supression minimum area overlap percentage must be between 0 and 1 (inclusive)'

        # Convert optional parameters to C-valid default options
        if output_gpath_list is None:
            output_gpath_list = [''] * len(input_gpath_list)
        elif output_gpath_list is not None:
            assert len(output_gpath_list) == len(input_gpath_list), \
                'Output image path list is invalid or is not the same length as the input list'
            for index in range(len(output_gpath_list)):
                if output_gpath_list[index] is None:
                    output_gpath_list[index] = ''
        output_gpath_list = _cast_list_to_c(output_gpath_list, C_CHAR)

        if output_scale_gpath_list is None:
            output_scale_gpath_list = [''] * len(input_gpath_list)
        elif output_scale_gpath_list is not None:
            assert len(output_scale_gpath_list) == len(input_gpath_list), \
                'Output scale image path list is invalid or is not the same length as the input list'
            for index in range(len(output_scale_gpath_list)):
                if output_scale_gpath_list[index] is None:
                    output_scale_gpath_list[index] = ''
        output_scale_gpath_list = _cast_list_to_c(output_scale_gpath_list, C_CHAR)

        # Prepare for C
        params['_scale_num'] = len(params['scale_list'])
        params['scale_list'] = _cast_list_to_c(params['scale_list'], C_FLOAT)
        if not params['quiet']:
            print('[pyrf py] Detecting over %d scales' % (params['_scale_num'], ))

        # Run training algorithm
        batch_size = params['batch_size']
        del params['batch_size']  # Remove this value from params
        batch_num = int(len(input_gpath_list) / batch_size) + 1
        # Detect for each batch
        for batch in ut.ProgressIter(range(batch_num), lbl="[pyrf py]", freq=1, invert_rate=True):
            begin = time.time()
            start = batch * batch_size
            end   = start + batch_size
            if end > len(input_gpath_list):
                end = len(input_gpath_list)
            input_gpath_list_        = input_gpath_list[start:end]
            output_gpath_list_       = output_gpath_list[start:end]
            output_scale_gpath_list_ = output_scale_gpath_list[start:end]
            num_images = len(input_gpath_list_)
            # Set image detection to be run in serial if less than half a batch to run
            if num_images < min(batch_size / 2, 8):
                params['serial'] = True
            # Final sanity check
            assert len(input_gpath_list_) == len(output_gpath_list_) and len(input_gpath_list_) == len(output_scale_gpath_list_)
            params['results_val_array'] = np.empty(num_images, dtype=NP_ARRAY_FLOAT)
            params['results_len_array'] = np.empty(num_images, dtype=C_INT)
            # Make the params_list
            params_list = [
                forest,
                _cast_list_to_c(input_gpath_list_, C_CHAR),
                num_images,
                _cast_list_to_c(output_gpath_list_, C_CHAR),
                _cast_list_to_c(output_scale_gpath_list_, C_CHAR)
            ] + params.values()
            RF_CLIB.detect(rf.detector_c_obj, *params_list)
            results_list = _extract_np_array(params['results_len_array'], params['results_val_array'], NP_ARRAY_FLOAT, NP_FLOAT32, RESULT_LENGTH)
            conclude = time.time()
            if not params['quiet']:
                print('[pyrf py] Took %r seconds to compute %d images' % (conclude - begin, num_images, ))
            for input_gpath, result_list in zip(input_gpath_list_, results_list):
                if params['mode'] == 0:
                    result_list_ = []
                    for result in result_list:
                        # Unpack result into a nice Python dictionary and return
                        temp = {}
                        temp['centerx']    = int(result[0])
                        temp['centery']    = int(result[1])
                        temp['xtl']        = int(result[2])
                        temp['ytl']        = int(result[3])
                        temp['width']      = int(result[4])
                        temp['height']     = int(result[5])
                        temp['confidence'] = float(np.round(result[6], decimals=4))
                        temp['suppressed'] = int(result[7]) == 1
                        result_list_.append(temp)
                    yield (input_gpath, result_list_)
                else:
                    yield (input_gpath, None)
            params['results_val_array'] = None
            params['results_len_array'] = None

    # Pickle functions
    def dump(rf, file):
        '''
            UNIMPLEMENTED

            Args:
                file (object)

            Returns:
                None
        '''
        pass

    def dumps(rf):
        '''
            UNIMPLEMENTED

            Returns:
                string
        '''
        pass

    def load(file):
        '''
            UNIMPLEMENTED

            Args:
                file (object)

            Returns:
                detector (object)
        '''
        pass

    def loads(string):
        '''
            UNIMPLEMENTED

            Args:
                string (str)

            Returns:
                detector (object)
        '''
        pass
