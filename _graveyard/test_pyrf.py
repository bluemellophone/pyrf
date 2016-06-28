#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, split
from pyrf import Random_Forest_Detector
import utool


TEST_DATA_DETECT_URL = 'https://lev.cs.rpi.edu/public/data/testdata_detect.zip'
TEST_DATA_MODEL_URL = 'https://lev.cs.rpi.edu/public/models/rf.zip'


def test_pyrf():
    category = 'zebra_plains'

    detect_config = {
        'save_detection_images':        True,
        'save_scales':                  True,
        'percentage_top':               0.40,
    }

    #=================================
    # Train / Detect Initialization
    #=================================

    testdata_dir = utool.unixpath('~/code/pyrf/results')
    # assert utool.checkpath(testdata_dir)
    if utool.get_argflag('--vd'):
        print(utool.ls(testdata_dir))

    # Create detector
    detector = Random_Forest_Detector()

    test_path = utool.grab_zipped_url(TEST_DATA_DETECT_URL, appname='utool')
    models_path = utool.grab_zipped_url(TEST_DATA_MODEL_URL, appname='utool')
    trees_path = join(models_path, category)

    results_path  = join(utool.unixpath('~/code/pyrf/results'), category)
    # detect_path   = join(results_path, 'detect')
    trees_path    = join(results_path, 'trees')

    detect_path = join(test_path, category, 'detect')
    utool.ensuredir(detect_path)
    utool.ensuredir(test_path)
    utool.ensuredir(trees_path)

    #=================================
    # Detect using Random Forest
    #=================================

    # Get input images
    from vtool import image
    big_gpath_list = utool.list_images(test_path, fullpath=True, recursive=False)
    print(big_gpath_list)
    # Resize images to standard size
    if utool.get_argflag('--small'):
        big_gpath_list = big_gpath_list[0:8]
    #big_gpath_list = big_gpath_list[0:8]
    output_dir = join(test_path, 'resized')
    std_gpath_list = image.resize_imagelist_to_sqrtarea(big_gpath_list,
                                                        sqrt_area=800,
                                                        output_dir=output_dir,
                                                        checkexists=True)
    dst_gpath_list = [join(detect_path, split(gpath)[1]) for gpath in std_gpath_list]
    #utool.view_directory(test_path)
    #utool.view_directory('.')
    print(std_gpath_list)
    num_images = len(std_gpath_list)
    #assert num_images == 16, 'the test has diverged!'
    print('Testing on %r images' % num_images)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, category + '-', num_trees=25)
    detector.set_detect_params(**detect_config)
    results_list1 = []
    with utool.Timer('[test_pyrf] for loop detector.detect') as t1:
        if not utool.get_argflag('--skip1'):
            for ix, (img_fpath, dst_fpath) in enumerate(zip(std_gpath_list, dst_gpath_list)):
                #img_fname = split(img_fpath)[1]
                #dst_fpath = join(detect_path, img_fname)
                #print('  * img_fpath = %r' % img_fpath)
                #print('  * dst_fpath = %r' % dst_fpath)
                with utool.Timer('[test_pyrf] detector.detect ix=%r' % (ix,)):
                    results = detector.detect(forest, img_fpath, dst_fpath)
                results_list1.append(results)
                print('num results = %r' % len(results))
            else:
                print('...skipped')

    # with utool.Timer('[test_pyrf] detector.detect_many') as t2:
    #     results_list2 = detector.detect_many(forest, std_gpath_list,
    #                                          dst_gpath_list, use_openmp=True)

    print('')
    print('+ --------------')
    print('| total time1: %r' % t1.ellapsed)
    # print('| total time2: %r' % t2.ellapsed)
    print('|')
    print('| num results1 = %r' % (list(map(len, results_list1))))
    # print('| num results2 = %r' % (list(map(len, results_list2))))
    #assert results_list2 == results_list1
    return locals()


if __name__ == '__main__':
    test_locals = utool.run_test(test_pyrf)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
