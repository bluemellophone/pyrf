#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from os.path import join, split
from pyrf import Random_Forest_Detector
#from detecttools.directory import Directory
#from pyrf.pyrf_helpers import rmtreedir, ensuredir
# import cv2
import utool

TEST_DATA_DETECT_URL = 'https://www.dropbox.com/s/s4gkjyxjgghr18c/testdata_detect.zip'
TEST_DATA_MODEL_URL = 'https://dl.dropboxusercontent.com/s/9814r3d2rkiq5t3/rf.zip'


def test_pyrf():
    # testdata_dir = utool.unixpath('~/code/pyrf/testdata_detect')
    testdata_dir = utool.unixpath('~/code/pyrf/results')

    #assert utool.checkpath(testdata_dir)

    # if utool.get_flag('--vd'):
    #     print(utool.ls(testdata_dir))

    # STOP: This is not a training file. This is a test script and should
    # remain stable. Fork it into train_pyrf if you want to use it for that
    # purpose

    # Create detector
    detector = Random_Forest_Detector()
    #category = 'giraffe'
    category = 'zebra_grevys'

    #dataset_path = '../IBEIS2014/'
    #pos_path    = join(testdata_dir, category, 'train-positives')
    #neg_path    = join(testdata_dir, category, 'train-negatives')
    #val_path    = join(testdata_dir, category, 'val')
    #test_path   = join(testdata_dir, category, 'test')
    #detect_path = join(testdata_dir, category, 'detect')
    #trees_path  = join(testdata_dir, category, 'trees')

    test_path = utool.grab_zipped_url(TEST_DATA_DETECT_URL, appname='utool')
    models_path = utool.grab_zipped_url(TEST_DATA_MODEL_URL, appname='utool')
    trees_path = join(models_path, category)
    detect_path = join(test_path, category, 'detect')
    utool.ensuredir(detect_path)

    utool.assertpath(test_path, verbose=True)
    utool.assertpath(trees_path, verbose=True)

    #=================================
    # Train / Detect Configurations
    #=================================

    train_config = {
        'object_min_width':             32,
        'object_min_height':            32,
        'neg_exclude_categories':       [category],

        'mine_negatives':               True,
        'mine_max_keep':                3,
        'mine_exclude_categories':      [category],
        'mine_width_min':               128,
        'mine_width_max':               512,
        'mine_height_min':              128,
        'mine_height_max':              512,

        'max_rois_pos':                 None,
        'max_rois_neg':                 1200,
    }

    detect_config = {
        'save_detection_images':        True,
        'percentage_top':               0.40,
    }

    #=================================
    # Train Random Forest
    #=================================

    # detector.train(dataset_path, category, pos_path, neg_path, val_path, test_path, trees_path, **train_config)

    #=================================
    # Detect using Random Forest
    #=================================

    # Get input images
    from vtool import image
    big_gpath_list = utool.list_images(test_path, fullpath=True, recursive=False)
    print(big_gpath_list)
    # Resize images to standard size
    output_dir = join(test_path, 'resized')
    std_gpath_list = image.resize_imagelist_to_sqrtarea(big_gpath_list,
                                                        sqrt_area=800,
                                                        output_dir=output_dir)
    dst_gpath_list = [join(detect_path, split(gpath)[1]) for gpath in std_gpath_list]
    #utool.view_directory(test_path)
    #utool.view_directory('.')
    print(std_gpath_list)
    num_images = len(std_gpath_list)
    assert num_images == 16, 'the test has diverged!'
    print('Testing on %r images' % num_images)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, category + '-')
    detector.set_detect_params(**detect_config)
    #for ix, img_fname in enumerate(std_gpath_list[::-1]):
    for ix, (img_fpath, dst_fpath) in enumerate(zip(std_gpath_list, dst_gpath_list)):
        #img_fname = split(img_fpath)[1]
        #dst_fpath = join(detect_path, img_fname)
        print(img_fpath)
        print(dst_fpath)
        #with utool.Timer('[rf] img_fpath=%r' % (img_fpath,)):
        #    results = detector.detect(forest, img_fpath, dst_fpath)
        #print('[rf] %s | Time: %.3f' % (img_fpath, timing))
        #print(results)

    with utool.Timer('[rf] parallel'):
        parallel_results = detector.detect_many(forest, std_gpath_list, dst_gpath_list)
    print(parallel_results)

    return locals()


if __name__ == '__main__':
    test_locals = utool.run_test(test_pyrf)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
