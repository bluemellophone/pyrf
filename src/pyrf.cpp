#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "pyrf.h"
#include "CRForest-Detector-Class.hpp"
#include "CRForest.h"

using namespace std;

typedef unsigned char uint8;

#ifdef __cplusplus
extern "C"
{
#endif
#define PYTHON_RANDOM_FOREST extern RANDOM_FOREST_DETECTOR_EXPORT

// TODO: REMOVE STRING WHERE CHAR* SHOULD BE USED
PYTHON_RANDOM_FOREST CRForestDetectorClass *init(bool verbose, bool quiet)
{
    CRForestDetectorClass *detector = new CRForestDetectorClass(verbose, quiet);
    return detector;
}

PYTHON_RANDOM_FOREST CRForest *forest(CRForestDetectorClass *detector, char **tree_path_array,
                                      int _tree_path_num, bool serial, bool verbose,
                                      bool quiet)
{
    // Convert char* pointers to vector of strings for convenience
    vector<string> tree_path_vector(_tree_path_num);
    for (int index = 0; index < _tree_path_num; ++index)
    {
        tree_path_vector[index] = tree_path_array[index];
    }
    CRForest *result = detector->forest(tree_path_vector, serial, verbose, quiet);
    tree_path_vector.clear();
    return result;
}

PYTHON_RANDOM_FOREST void train(CRForestDetectorClass *detector, char *train_pos_chip_path,
                                char **train_pos_chip_filename_array, int _train_pos_chip_num,
                                char *train_neg_chip_path, char **train_neg_chip_filename_array,
                                int _train_neg_chip_num, char *trees_path,
                                int patch_width, int patch_height,
                                float patch_density, int trees_num, int trees_offset,
                                int trees_max_depth, int trees_max_patches,
                                int trees_leaf_size, int trees_pixel_tests,
                                float trees_prob_optimize_mode, bool serial, bool verbose,
                                bool quiet)
{
    // Convert char* to nice strings, we are not Neanderthals
    string train_pos_chip_path_string = train_pos_chip_path;
    string train_neg_chip_path_string = train_neg_chip_path;
    string trees_path_string          = trees_path;

    // Convert char* pointers to vector of strings for convenience
    vector<string> train_pos_chip_filename_vector(_train_pos_chip_num);
    for (int index = 0; index < _train_pos_chip_num; ++index)
    {
        train_pos_chip_filename_vector[index] = train_pos_chip_filename_array[index];
    }

    // Convert char* pointers to vector of strings for convenience
    vector<string> train_neg_chip_filename_vector(_train_neg_chip_num);
    for (int index = 0; index < _train_neg_chip_num; ++index)
    {
        train_neg_chip_filename_vector[index] = train_neg_chip_filename_array[index];
    }

    detector->train(train_pos_chip_path_string, train_pos_chip_filename_vector,
                    train_neg_chip_path_string, train_neg_chip_filename_vector,
                    trees_path_string,
                    patch_width, patch_height, patch_density, trees_num,
                    trees_offset, trees_max_depth, trees_max_patches,
                    trees_leaf_size, trees_pixel_tests, trees_prob_optimize_mode,
                    serial, verbose, quiet);
    train_pos_chip_filename_vector.clear();
    train_neg_chip_filename_vector.clear();
}

PYTHON_RANDOM_FOREST void detect(CRForestDetectorClass *detector, CRForest *forest,
                                 char **input_gpath_array, int _input_gpath_num,
                                 char **output_gpath_array,
                                 char **output_scale_gpath_array, int mode,
                                 float sensitivity, float *scale_array, int _scale_num,
                                 int nms_min_area_contour, float nms_min_area_overlap,
                                 float** results_array, int* length_array,
                                 int RESULT_LENGTH, bool serial, bool verbose, bool quiet)
{
    vector<float> scale_vector(_scale_num);
    for (int index = 0; index < _scale_num; ++index)
    {
        scale_vector[index] = scale_array[index];
    }

    if( ! quiet )
    {
        // Parallel processing of the images, ideally, one image per core
        if(serial)
        {
            cout << "[pyrf c] Detecting images parallelized across scales" << endl;
        }
        else
        {
            cout << "[pyrf c] Detecting images parallelized across batch" << endl;
        }
    }

    #pragma omp parallel for if(!serial)
    for (int index = 0; index < _input_gpath_num; ++index)
    {
        string input_gpath = input_gpath_array[index];
        string output_gpath = output_gpath_array[index];
        string output_scale_gpath = output_scale_gpath_array[index];
        // Run detection
        int length = detector->detect(forest, input_gpath, output_gpath,
                                       output_scale_gpath, mode, sensitivity,
                                       scale_vector, nms_min_area_contour,
                                       nms_min_area_overlap, &results_array[index],
                                       RESULT_LENGTH, serial, verbose, quiet);
        length_array[index] = length;
    }
    scale_vector.clear();
}

PYTHON_RANDOM_FOREST void free_detector(CRForestDetectorClass *detector, bool verbose, bool quiet)
{
    delete detector;
}

PYTHON_RANDOM_FOREST void free_forest(CRForest *forest, bool verbose, bool quiet)
{
    delete forest;
}

#ifdef __cplusplus
}
#endif
