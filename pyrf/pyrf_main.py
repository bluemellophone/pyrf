#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from os.path import join, isdir
import os
import shutil
from ._pyrf import detector


def rmtreedir(path):
    if isdir(path):
        shutil.rmtree(path)


def ensuredir(path):
    if isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    # Create detector
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
