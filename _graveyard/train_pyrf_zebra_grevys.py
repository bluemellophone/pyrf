#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, split, isdir
from pyrf import Random_Forest_Detector
from detecttools.directory import Directory
from detecttools.ibeisdata import IBEIS_Data
import utool
import sys
import shutil


def rmtreedir(path):
    if isdir(path):
        shutil.rmtree(path)


def ensuredir(path):
    utool.ensuredir(path)


def train_pyrf():
    # boosting = 3
    num_trees = 5
    category = 'zebra_grevys'

    #=================================
    # Train / Detect Configurations
    #=================================

    train_config = {
        'object_min_width':        32,
        'object_min_height':       32,
        'mine_negatives':          True,
        'mine_max_keep':           1,
        'mine_exclude_categories': [category],
        'mine_width_min':          128,
        'mine_width_max':          512,
        'mine_height_min':         128,
        'mine_height_max':         512,

        'neg_exclude_categories':  [category],
        'max_rois_pos':            900,
        'max_rois_neg':            550,
        'num_trees':               num_trees,
    }

    detect_config = {
        'save_detection_images':   True,
        'percentage_top':          0.40,
    }

    #=================================
    # Train / Detect Initialization
    #=================================

    # Create detector
    detector = Random_Forest_Detector()

    # Gather Dataset
    dataset_path = utool.unixpath('~/code/IBEIS2014/')
    dataset = IBEIS_Data(dataset_path, **train_config)

    results_path  = join(utool.unixpath('~/code/pyrf/results'), category)
    # pos_path      = join(results_path, 'train-positives')
    # neg_path      = join(results_path, 'train-negatives')
    # val_path      = join(results_path, 'val')
    test_path     = join(results_path, 'test')
    # test_pos_path = join(results_path, 'test-positives')
    # test_neg_path = join(results_path, 'test-negatives')
    detect_path   = join(results_path, 'detect')
    trees_path    = join(results_path, 'trees')

    # # Ensure result path for the category
    # # rmtreedir(results_path)
    # ensuredir(results_path)

    # for phase in range(1, boosting + 1):
    #     print("*********************")
    #     print("Phase: %s" % phase)
    #     print("*********************")
    #     # raw_input()
    #     # =================================
    #     # Train Random Forest
    #     #=================================
    #     detector.train(dataset, category, pos_path, neg_path, val_path,
    #                     test_path, test_pos_path, test_neg_path,
    #                     trees_path, reshuffle=(phase == 1), **train_config)

    #     if phase < boosting:
    #         #=================================
    #         # Detect using Random Forest
    #         #=================================

    #         # Load forest, so we don't have to reload every time
    #         forest = detector.load(trees_path, category + '-', num_trees=(phase * num_trees))
    #         detector.set_detect_params(**detect_config)

    #         # Ensure output detection paths
    #         rmtreedir(detect_path)
    #         ensuredir(detect_path)

    #         # Calculate error on test set
    #         direct = Directory(test_path, include_file_extensions=["jpg"])
    #         accuracy_list = []
    #         image_filepath_list = direct.files()
    #         dst_filepath_list   = [ join(detect_path, split(image_filepath)[1]) for image_filepath in image_filepath_list ]
    #         predictions_list = detector.detect_many(forest, image_filepath_list, dst_filepath_list, use_openmp=True)
    #         for index, (predictions, image_filepath) in enumerate(zip(predictions_list, image_filepath_list)):
    #             image_path, image_filename = split(image_filepath)
    #             image = dataset[image_filename]
    #             accuracy, true_pos, false_pos, false_neg = image.accuracy(predictions, category)
    #             accuracy_list.append(accuracy)
    #             progress = "%0.2f" % (float(index) / len(image_filepath_list))
    #             print("TEST %s %0.4f %s" % (image, accuracy, progress), end='\r')
    #             sys.stdout.flush()
    #             # image.show(prediction_list=predictions, category=category)
    #         print(' ' * 100, end='\r')
    #         print("TEST ERROR: %0.4f" % (1.0 - (float(sum(accuracy_list)) / len(accuracy_list))))

    #         #=================================
    #         # Eval and prep boosting train set
    #         #=================================
    #         detector.boosting(phase, forest, dataset, category, pos_path, neg_path,
    #                           test_pos_path, test_neg_path, detect_path)

    ####################################
    # New FAST
    ####################################

    detector = Random_Forest_Detector(
        scales='6 1.0 0.75 0.55 0.40 0.30 0.20'
    )

    # Ensure output detection paths
    detect_path_temp = detect_path + "_1"
    rmtreedir(detect_path_temp)
    ensuredir(detect_path_temp)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, category + '-', num_trees=25)
    detector.set_detect_params(**detect_config)

    # Calculate error on test set
    direct = Directory(test_path, include_file_extensions=["jpg"])
    accuracy_list  = []
    true_pos_list  = []
    false_pos_list = []
    false_neg_list = []
    image_filepath_list = direct.files()
    dst_filepath_list   = [ join(detect_path_temp, split(image_filepath)[1]) for image_filepath in image_filepath_list ]
    predictions_list = detector.detect_many(forest, image_filepath_list, dst_filepath_list, use_openmp=True)
    for index, (predictions, image_filepath) in enumerate(zip(predictions_list, image_filepath_list)):
        image_path, image_filename = split(image_filepath)
        image = dataset[image_filename]
        accuracy, true_pos, false_pos, false_neg = image.accuracy(predictions, category)
        accuracy_list.append(accuracy)
        true_pos_list.append(true_pos)
        false_pos_list.append(false_pos)
        false_neg_list.append(false_neg)
        progress = "%0.2f" % (float(index) / len(image_filepath_list))
        print("TEST %s %0.4f %s" % (image, accuracy, progress), end='\r')
        sys.stdout.flush()
        # image.show(prediction_list=predictions, category=category)
    print(' ' * 100, end='\r')
    print("1 TEST ERROR     : %0.4f" % (1.0 - (float(sum(accuracy_list)) / len(accuracy_list))))
    print("1 TEST TRUE POS  : %d" % (sum(true_pos_list)))
    print("1 TEST FALSE POS : %d" % (sum(false_pos_list)))
    print("1 TEST FALSE NEG : %d" % (sum(false_neg_list)))

    ####################################
    # New SLOW
    ####################################

    detector = Random_Forest_Detector(
        scales='11 1.5 1.25 1.0 0.8 0.64 0.51 0.41 0.33 0.26 0.21 0.17'
    )

    # Ensure output detection paths
    detect_path_temp = detect_path + "_2"
    rmtreedir(detect_path_temp)
    ensuredir(detect_path_temp)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, category + '-', num_trees=25)
    detector.set_detect_params(**detect_config)

    # Calculate error on test set
    direct = Directory(test_path, include_file_extensions=["jpg"])
    accuracy_list  = []
    true_pos_list  = []
    false_pos_list = []
    false_neg_list = []
    image_filepath_list = direct.files()
    dst_filepath_list   = [ join(detect_path_temp, split(image_filepath)[1]) for image_filepath in image_filepath_list ]
    predictions_list = detector.detect_many(forest, image_filepath_list, dst_filepath_list, use_openmp=True)
    for index, (predictions, image_filepath) in enumerate(zip(predictions_list, image_filepath_list)):
        image_path, image_filename = split(image_filepath)
        image = dataset[image_filename]
        accuracy, true_pos, false_pos, false_neg = image.accuracy(predictions, category)
        accuracy_list.append(accuracy)
        true_pos_list.append(true_pos)
        false_pos_list.append(false_pos)
        false_neg_list.append(false_neg)
        progress = "%0.2f" % (float(index) / len(image_filepath_list))
        print("TEST %s %0.4f %s" % (image, accuracy, progress), end='\r')
        sys.stdout.flush()
        # image.show(prediction_list=predictions, category=category)
    print(' ' * 100, end='\r')
    print("2 TEST ERROR     : %0.4f" % (1.0 - (float(sum(accuracy_list)) / len(accuracy_list))))
    print("2 TEST TRUE POS  : %d" % (sum(true_pos_list)))
    print("2 TEST FALSE POS : %d" % (sum(false_pos_list)))
    print("2 TEST FALSE NEG : %d" % (sum(false_neg_list)))

    # ####################################
    # # Current FAST
    # ####################################

    # detector = Random_Forest_Detector(
    #     scales='6 1.0 0.75 0.55 0.40 0.30 0.20'
    # )

    # # Use pre-trained trees?
    # TEST_DATA_MODEL_URL = 'https://lev.cs.rpi.edu/public/models/rf.zip'
    # models_path = utool.grab_zipped_url(TEST_DATA_MODEL_URL, appname='utool')
    # trees_path = join(models_path, category)

    # # Ensure output detection paths
    # detect_path_temp = detect_path + "_3"
    # rmtreedir(detect_path_temp)
    # ensuredir(detect_path_temp)

    # # Load forest, so we don't have to reload every time
    # forest = detector.load(trees_path, category + '-', num_trees=10)
    # detector.set_detect_params(**detect_config)

    # # Calculate error on test set
    # direct = Directory(test_path, include_file_extensions=["jpg"])
    # accuracy_list  = []
    # true_pos_list  = []
    # false_pos_list = []
    # false_neg_list = []
    # image_filepath_list = direct.files()
    # dst_filepath_list   = [ join(detect_path_temp, split(image_filepath)[1]) for image_filepath in image_filepath_list ]
    # predictions_list = detector.detect_many(forest, image_filepath_list, dst_filepath_list, use_openmp=True)
    # for index, (predictions, image_filepath) in enumerate(zip(predictions_list, image_filepath_list)):
    #     image_path, image_filename = split(image_filepath)
    #     image = dataset[image_filename]
    #     accuracy, true_pos, false_pos, false_neg = image.accuracy(predictions, category)
    #     accuracy_list.append(accuracy)
    #     true_pos_list.append(true_pos)
    #     false_pos_list.append(false_pos)
    #     false_neg_list.append(false_neg)
    #     progress = "%0.2f" % (float(index) / len(image_filepath_list))
    #     print("TEST %s %0.4f %s" % (image, accuracy, progress), end='\r')
    #     sys.stdout.flush()
    #     # image.show(prediction_list=predictions, category=category)
    # print(' ' * 100, end='\r')
    # print("3 TEST ERROR     : %0.4f" % (1.0 - (float(sum(accuracy_list)) / len(accuracy_list))))
    # print("3 TEST TRUE POS  : %d" % (sum(true_pos_list)))
    # print("3 TEST FALSE POS : %d" % (sum(false_pos_list)))
    # print("3 TEST FALSE NEG : %d" % (sum(false_neg_list)))

    # ####################################
    # # Current SLOW
    # ####################################

    # detector = Random_Forest_Detector(
    #     scales='11 1.5 1.25 1.0 0.8 0.64 0.51 0.41 0.33 0.26 0.21 0.17'
    # )

    # # Ensure output detection paths
    # detect_path_temp = detect_path + "_4"
    # rmtreedir(detect_path_temp)
    # ensuredir(detect_path_temp)

    # # Load forest, so we don't have to reload every time
    # forest = detector.load(trees_path, category + '-', num_trees=10)
    # detector.set_detect_params(**detect_config)

    # # Calculate error on test set
    # direct = Directory(test_path, include_file_extensions=["jpg"])
    # accuracy_list  = []
    # true_pos_list  = []
    # false_pos_list = []
    # false_neg_list = []
    # image_filepath_list = direct.files()
    # dst_filepath_list   = [ join(detect_path_temp, split(image_filepath)[1]) for image_filepath in image_filepath_list ]
    # predictions_list = detector.detect_many(forest, image_filepath_list, dst_filepath_list, use_openmp=True)
    # for index, (predictions, image_filepath) in enumerate(zip(predictions_list, image_filepath_list)):
    #     image_path, image_filename = split(image_filepath)
    #     image = dataset[image_filename]
    #     accuracy, true_pos, false_pos, false_neg = image.accuracy(predictions, category)
    #     accuracy_list.append(accuracy)
    #     true_pos_list.append(true_pos)
    #     false_pos_list.append(false_pos)
    #     false_neg_list.append(false_neg)
    #     progress = "%0.2f" % (float(index) / len(image_filepath_list))
    #     print("TEST %s %0.4f %s" % (image, accuracy, progress), end='\r')
    #     sys.stdout.flush()
    #     # image.show(prediction_list=predictions, category=category)
    # print(' ' * 100, end='\r')
    # print("4 TEST ERROR     : %0.4f" % (1.0 - (float(sum(accuracy_list)) / len(accuracy_list))))
    # print("4 TEST TRUE POS  : %d" % (sum(true_pos_list)))
    # print("4 TEST FALSE POS : %d" % (sum(false_pos_list)))
    # print("4 TEST FALSE NEG : %d" % (sum(false_neg_list)))


if __name__ == '__main__':
    train_pyrf()
