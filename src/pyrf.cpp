/*
Source: http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/Ctypes.ipynb

TO BEGIN:
1.) Search, replace RANDOM_FOREST with an all-caps moniker for the alrogithm
    Examples: 'HESAFF', 'RANDOM_FOREST', 'DPM_MKL'

2.) Search, replace CRForestDetectorClass with the C++ class for the algorithm.

3.) Include all C++ header files for the algorithm you intend to wrap

4.) Change the wrapper functions for all of the pre-set functions to
    match the appropriate functions in the algorithm's C++ code
*/
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "pyrf.h"

#include "CRForest-Detector-Class.hpp"
#include "CRForest.h"

typedef unsigned char uint8;

#ifdef __cplusplus
    extern "C" {
#endif
    #define PYTHON_RANDOM_FOREST extern RANDOM_FOREST_DETECTOR_EXPORT

        //=============================
        // Algorithm Constructor
        //=============================
        PYTHON_RANDOM_FOREST CRForestDetectorClass* constructor(
                int     patch_width,
                int     patch_height,
                int     out_scale,
                int     default_split,
                int     positive_like,
                bool    legacy,
                bool    include_horizontal_flip,
                int     patch_sample_density_pos,
                int     patch_sample_density_neg,
                char*   scales,
                char*   ratios
            )
        {
            CRForestDetectorClass* detector = new CRForestDetectorClass(
                patch_width,
                patch_height,
                out_scale,
                default_split,
                positive_like,
                legacy,
                include_horizontal_flip,
                patch_sample_density_pos,
                patch_sample_density_neg,
                scales,
                ratios
            );
            return detector;
        }

        //=============================
        // Train Algorithm with Data
        //=============================
        PYTHON_RANDOM_FOREST void train(
                CRForestDetectorClass* detector,
                char* tree_path,
                int num_trees,
                char* training_inventory_pos,
                char* training_inventory_neg
            )
        {
            detector->run_train(
                tree_path,
                num_trees,
                training_inventory_pos,
                training_inventory_neg
            );
        }

        //=============================
        // Run Algorithm
        //=============================

        PYTHON_RANDOM_FOREST int detect(
                CRForestDetectorClass* detector,
                CRForest* forest,
                char* detection_image_filepath,
                char* detection_result_filepath,
                bool save_detection_images,
                bool save_scales,
                bool draw_supressed,
                int detection_width,
                int detection_height,
                float percentage_left,
                float percentage_top,
                float nms_margin_percentage,
                int min_contour_area
            )
        {
            return detector->run_detect(
                forest,
                detection_image_filepath,
                detection_result_filepath,
                save_detection_images,
                save_scales,
                draw_supressed,
                detection_width,
                detection_height,
                percentage_left,
                percentage_top,
                nms_margin_percentage,
                min_contour_area
            );
        }


        PYTHON_RANDOM_FOREST void detect_many(
                CRForestDetectorClass* detector,
                CRForest* forest,
                int nImgs,
                char** detection_image_filepath_list,
                char** detection_result_filepath_list,
                int* length_array,
                float** results_array,
                bool save_detection_images,
                bool save_scales,
                bool draw_supressed,
                int detection_width,
                int detection_height,
                float percentage_left,
                float percentage_top,
                float nms_margin_percentage,
                int min_contour_area
            )
        {
            int index;
            #pragma omp parallel for
            for(index=0;index < nImgs;++index)
            {
                 int length = detector->run_detect(
                        forest,
                        detection_image_filepath_list[index],
                        detection_result_filepath_list[index],
                        save_detection_images,
                        save_scales,
                        draw_supressed,
                        detection_width,
                        detection_height,
                        percentage_left,
                        percentage_top,
                        nms_margin_percentage,
                        min_contour_area
                        );
                 length_array[index] = length;
                 results_array[index] = new float[length];
                 detector->detect_results(results_array[index]);
            }
        }

        PYTHON_RANDOM_FOREST void detect_results(
                CRForestDetectorClass* detector,
                float *results
            )
        {
            detector->detect_results(
                results
            );
        }


        PYTHON_RANDOM_FOREST void segment(CRForestDetectorClass* detector)
        {
            // detector->segment();
        }

        //=============================
        // Load / Save Trained Data
        //=============================

        PYTHON_RANDOM_FOREST CRForest* load(
                CRForestDetectorClass* detector,
                char* tree_path,
                char* prefix,
                int num_trees
            )
        {
            return detector->load_forest(tree_path, prefix, num_trees);
        }

        PYTHON_RANDOM_FOREST void save(CRForestDetectorClass* detector)
        {
            // Not used
        }

#ifdef __cplusplus
    }
#endif
