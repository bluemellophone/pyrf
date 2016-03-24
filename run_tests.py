#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, split
from pyrf import Random_Forest_Detector
import utool as ut


TEST_DATA_DETECT_URL = 'https://lev.cs.rpi.edu/public/data/testdata_detect.zip'
TEST_DATA_MODEL_URL = 'https://lev.cs.rpi.edu/public/models/rf.zip'


def test_pyrf():
    r"""
    CommandLine:
        python run_tests.py --test-test_pyrf

    Example:
        >>> # ENABLE_DOCTEST
        >>> from run_tests import *  # NOQA
        >>> result = test_pyrf()
        >>> print(result)
    """

    #=================================
    # Initialization
    #=================================

    category = 'zebra_plains'

    #detect_config = {
    #    'save_detection_images':        True,
    #    'percentage_top':               0.40,
    #}

    testdata_dir = ut.unixpath('~/code/pyrf/results')
    # assert ut.checkpath(testdata_dir)
    if ut.get_argflag('--vd'):
        print(ut.ls(testdata_dir))

    # Create detector
    detector = Random_Forest_Detector()

    test_path = ut.grab_zipped_url(TEST_DATA_DETECT_URL, appname='utool')
    models_path = ut.grab_zipped_url(TEST_DATA_MODEL_URL, appname='utool')
    trees_path = join(models_path, category)
    detect_path = join(test_path, category, 'detect')
    ut.ensuredir(detect_path)
    ut.ensuredir(test_path)
    ut.ensuredir(trees_path)

    #=================================
    # Load Input Images
    #=================================

    # Get input images
    import vtool as vt
    big_gpath_list = ut.list_images(test_path, fullpath=True, recursive=False)
    print(big_gpath_list)
    # Resize images to standard size
    if ut.get_argflag('--small'):
        big_gpath_list = big_gpath_list[0:8]
    #big_gpath_list = big_gpath_list[0:8]
    output_dir = join(test_path, 'resized')
    std_gpath_list = vt.resize_imagelist_to_sqrtarea(big_gpath_list,
                                                     sqrt_area=800,
                                                     output_dir=output_dir,
                                                     checkexists=True)
    dst_gpath_list = [join(detect_path, split(gpath)[1]) for gpath in std_gpath_list]
    #ut.view_directory(test_path)
    #ut.view_directory('.')
    print(std_gpath_list)
    num_images = len(std_gpath_list)
    #assert num_images == 16, 'the test has diverged!'
    print('Testing on %r images' % num_images)

    #=================================
    # Load Pretrained Forests
    #=================================

    # Load forest, so we don't have to reload every time
    trees_fpath_list = ut.ls(trees_path, '*.txt')
    #forest = detector.load(trees_path, category + '-')
    forest = detector.forest(trees_fpath_list)
    #detector.set_detect_params(**detect_config)
    results_list1 = []

    #=================================
    # Detect using Random Forest
    #=================================

    with ut.Timer('[test_pyrf] for loop detector.detect') as t1:
        if not ut.get_argflag('--skip1'):
            results_list1 = detector.detect(forest, std_gpath_list, output_gpath_list=dst_gpath_list)
            #for ix, (img_fpath, dst_fpath) in enumerate(zip(std_gpath_list, dst_gpath_list)):
            #    #img_fname = split(img_fpath)[1]
            #    #dst_fpath = join(detect_path, img_fname)
            #    #print('  * img_fpath = %r' % img_fpath)
            #    #print('  * dst_fpath = %r' % dst_fpath)
            #    with ut.Timer('[test_pyrf] detector.detect ix=%r' % (ix,)):
            #        results = detector.detect(forest, img_fpath, dst_fpath)
            #    results_list1.append(results)
            #    print('num results = %r' % len(results))
            #else:
            #    print('...skipped')

    #with ut.Timer('[test_pyrf] detector.detect_many') as t2:
    #    results_list2 = detector.detect_many(forest, std_gpath_list,
    #                                         dst_gpath_list, use_openmp=True)
    detector.free_forest(forest)

    print('')
    print('+ --------------')
    print('| total time1: %r' % t1.ellapsed)
    #print('| total time2: %r' % t2.ellapsed)
    print('|')
    print('| num results1 = %r' % (list(map(len, results_list1))))
    #print('| num results2 = %r' % (list(map(len, results_list2))))
    #assert results_list2 == results_list1
    return locals()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/pyrf
        python ~/code/pyrf/run_tests.py
        python ~/code/pyrf/run_tests.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
