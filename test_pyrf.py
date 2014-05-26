#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from os.path import join
from pyrf import Random_Forest_Detector
from pyrf.pyrf_helpers import rmtreedir, ensuredir


if __name__ == '__main__':
    import utool
    #detectmodels_url =
    testdata_url = '~/code/pyrf/testdata_detect'
    #testdata_dir = utool.grab_zipped_url(testdata_url)
    detectmodels_dir = utool.grab_zipped_url('https://dl.dropboxusercontent.com/s/9814r3d2rkiq5t3/rf.zip')
    detectmodels_dir = r'C:\Users\joncrall\AppData\Roaming\utool\rf'
    detectmodels_dir = utool.get_app_resource_dir('utool', 'rf')
    testdata_dir = utool.unixpath('~/code/pyrf/testdata_detect')

    assert utool.checkpath(testdata_dir)
    assert utool.checkpath(detectmodels_dir)


    if utool.get_flag('--vd'):
        print(utool.ls(detectmodels_dir))
        print(utool.ls(testdata_dir))

    # Create detector
    detector = Random_Forest_Detector()
    category = 'zebra_grevys'

    dataset_path = '../IBEIS2014/'
    pos_path    = join(detectmodels_dir, category, 'train-positives')
    neg_path    = join(detectmodels_dir, category, 'train-negatives')
    val_path    = join(detectmodels_dir, category, 'val')
    detect_path = join(detectmodels_dir, category, 'detect')
    trees_path  = join(detectmodels_dir, category)
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
    for ix, img_fname in enumerate(utool.list_images(testdata_dir)):
        img_fpath = join(testdata_dir, img_fname)
        dst_fpath = join(detect_path, img_fpath.split('/')[-1])

        results, timing = detector.detect(forest, img_fpath, dst_fpath,
                                          **detect_config)

        print('[rf] %s | Time: %.3f' % (img_fpath, timing))
        print(results)
