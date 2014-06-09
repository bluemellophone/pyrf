#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from os.path import join
from pyrf import Random_Forest_Detector
from detecttools.directory import Directory
from pyrf.pyrf_helpers import rmtreedir, ensuredir
import utool

if __name__ == '__main__':
    # testdata_dir = utool.unixpath('~/code/pyrf/testdata_detect')
    testdata_dir = utool.unixpath('~/code/pyrf/results')
    assert utool.checkpath(testdata_dir)
    
    if utool.get_flag('--vd'):
        print(utool.ls(testdata_dir))

    # Create detector
    detector = Random_Forest_Detector()
    category = 'zebra_grevys'

    dataset_path = '../IBEIS2014/'
    pos_path    = join(testdata_dir, category, 'train-positives')
    neg_path    = join(testdata_dir, category, 'train-negatives')
    val_path    = join(testdata_dir, category, 'val')
    test_path   = join(testdata_dir, category, 'test')
    detect_path = join(testdata_dir, category, 'detect')
    trees_path  = join(testdata_dir, category, 'trees')
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
    # detector.train(dataset_path, category, pos_path, neg_path, val_path, test_path, trees_path, **train_config)

    #=================================
    # Detect using Random Forest
    #=================================

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, tree_prefix)

    # Get input images
    for ix, img_fname in enumerate(utool.list_images(test_path)[150:]):
        img_fpath = join(test_path, img_fname)
        dst_fpath = join(detect_path, img_fpath.split('/')[-1])

        results, timing = detector.detect(forest, img_fpath, dst_fpath,
                                          **detect_config)

        print('[rf] %s | Time: %.3f' % (img_fpath, timing))
        print(results)
