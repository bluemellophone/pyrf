#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, split
from pyrf import Random_Forest_Detector
from detecttools.directory import Directory
from detecttools.ibeisdata import IBEIS_Data
import utool


def train_pyrf():
    boosting = 3
    num_trees = 5
    category = 'zebra_plains'

    #=================================
    # Train / Detect Configurations
    #=================================

    train_config = {
        'object_min_width':             32,
        'object_min_height':            32,
        'mine_negatives':               True,
        'mine_max_keep':                1,
        'mine_exclude_categories':      [category],
        'mine_width_min':               128,
        'mine_width_max':               512,
        'mine_height_min':              128,
        'mine_height_max':              512,

        'neg_exclude_categories':       [category],
        # 'max_rois_pos':                 1200,
        # 'max_rois_neg':                 1200,
        'max_rois_pos':                 50,
        'max_rois_neg':                 50,
        'num_trees':                    num_trees,
    }

    detect_config = {
        'save_detection_images':        True,
        'percentage_top':               0.40,
    }

    #=================================
    # Train / Detect Initialization
    #=================================

    # Create detector
    detector = Random_Forest_Detector()

    dataset_path = utool.unixpath('~/code/IBEIS2014/')
    dataset = IBEIS_Data(dataset_path, **train_config)

    results_path  = join(utool.unixpath('~/code/pyrf/results'), category)
    pos_path      = join(results_path, 'train-positives')
    neg_path      = join(results_path, 'train-negatives')
    val_path      = join(results_path, 'val')
    test_path     = join(results_path, 'test')
    test_pos_path = join(results_path, 'test-positives')
    test_neg_path = join(results_path, 'test-negatives')
    detect_path   = join(results_path, 'detect')
    trees_path    = join(results_path, 'trees')

    for phase in range(1, boosting + 1):
        print("*********************")
        print("Phase: %s" % phase)
        print("*********************")
        raw_input()
        #=================================
        # Train Random Forest
        #=================================
        detector.train(dataset, category, pos_path, neg_path, val_path,
                        test_path, test_pos_path, test_neg_path,
                        trees_path, reshuffle=(phase == 0), **train_config)

        #=================================
        # Detect using Random Forest
        #=================================
        # TEST_DATA_MODEL_URL = 'https://dl.dropboxusercontent.com/s/9814r3d2rkiq5t3/rf.zip'
        # models_path = utool.grab_zipped_url(TEST_DATA_MODEL_URL, appname='utool')
        # trees_path = join(models_path, category)

        # Load forest, so we don't have to reload every time
        forest = detector.load(trees_path, category + '-', num_trees=(phase * num_trees))
        detector.set_detect_params(**detect_config)

        # Calculate error on test set
        direct = Directory(test_path , include_file_extensions=["jpg"])
        accuracy_list = []
        image_filepath_list = direct.files()
        for index, image_filepath in enumerate(image_filepath_list):
            image_path, image_filename = split(image_filepath)
            predictions = detector.detect(forest, image_filepath, join(detect_path, image_filename))
            image = dataset[image_filename]
            accuracy, true_pos, false_pos, false_neg = image.accuracy(predictions, category)
            accuracy_list.append(accuracy)
            progress = "%0.2f" % (float(index) / len(image_filepath_list))
            print(image, accuracy, progress)
            # image.show(prediction_list=predictions, category=category)
        print(sum(accuracy_list) / len(accuracy_list))

        #=================================
        # Eval and prep boosting train set
        #=================================
        if phase < boosting:
            detector.boosting()

if __name__ == '__main__':
    train_pyrf()
