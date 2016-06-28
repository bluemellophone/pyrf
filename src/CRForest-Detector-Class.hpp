/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch

// You may use, copy, reproduce, and distribute this Software for any
// non-commercial purpose, subject to the restrictions of the
// Microsoft Research Shared Source license agreement ("MSR-SSLA").
// Some purposes which can be non-commercial are teaching, academic
// research, public demonstrations and personal experimentation. You
// may also distribute this Software with books or other teaching
// materials, or publish the Software on websites, that are intended
// to teach the use of the Software for academic or other
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works
// in any form for commercial purposes. Examples of commercial
// purposes would be running business operations, licensing, leasing,
// or selling the Software, distributing the Software for use with
// commercial products, using the Software in the creation or use of
// commercial products or any other activity which purpose is to
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create
// derivative works of such portions of the Software and distribute
// the modified Software for non-commercial purposes, as provided
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA,
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE
// WORKS.

// When using this software, please acknowledge the effort that
// went into development by referencing the paper:
//
// Gall J. and Lempitsky V., Class-Specific Hough Forests for
// Object Detection, IEEE Conference on Computer Vision and Pattern
// Recognition (CVPR'09), 2009.

// Note that this is not the original software that was used for
// the paper mentioned above. It is a re-implementation for Linux.

*/


#define PATH_SEP "/"

#include <stdexcept>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv/highgui.h>

#include "CRForestDetector.h"

using namespace std;

struct CRForestDetectorClass
{
public:
    CRForestDetectorClass(bool verbose, bool quiet)
    {
        if( ! quiet )
        {
            #ifdef _OPENMP
                    cout << "[pyrf c++] --- RUNNING PYRF DETECTOR IN PARALLEL ---" << endl;
            #else
                    cout << "[pyrf c++] --- RUNNING PYRF DETECTOR IN SERIAL ---" << endl;
            #endif
        }
    }
    ~CRForestDetectorClass() {
        // Nothing to do
    }

    CRForest *forest(vector<string> &tree_path_vector, bool serial, bool verbose, bool quiet)
    {
        // Init forest with number of trees
        CRForest *crForest = new CRForest( tree_path_vector.size() );
        // Load forest
        crForest->loadForest(tree_path_vector, serial, verbose && ! quiet);
        return crForest;
    }

    // Init and start training
    void train(string train_pos_chip_path_string,
               vector<string> &train_pos_chip_filename_vector,
               string train_neg_chip_path_string,
               vector<string> &train_neg_chip_filename_vector,
               string trees_path_string, int patch_width, int patch_height,
               float patch_density, int trees_num, int trees_offset,
               int trees_max_depth, int trees_max_patches,
               int trees_leaf_size, int trees_pixel_tests,
               float trees_prob_optimize_mode, bool serial, bool verbose, bool quiet)
    {
        // Init new forest with number of trees
        CRForest crForest( trees_num );

        // Init random generator
        time_t t = time(NULL);
        int seed = (int) t;
        CvRNG cvRNG(seed);

        // Init training data
        CRPatch Train(&cvRNG, patch_width, patch_height, 2);

        // Extract pos training patches
        cout << "[pyrf c++] Loading positive patches..." << endl;
        int pos_patches = extract_Patches(Train, &cvRNG, train_pos_chip_path_string,
                                          train_pos_chip_filename_vector, 1, patch_density,
                                          trees_max_patches / 2, verbose, quiet);
        if( ! quiet )
        {
            cout << endl << "[pyrf c++] ...Loaded " << pos_patches << " patches" << endl;
        }

        // Extract neg training patches
        cout << "[pyrf c++] Loading negative patches..." << endl;
        int neg_patches = extract_Patches(Train, &cvRNG, train_neg_chip_path_string,
                                          train_neg_chip_filename_vector, 0, patch_density,
                                          pos_patches, verbose, quiet);  // We pass pos_patches as max_patches for balance
        if( ! quiet )
        {
            cout << endl << "[pyrf c++] ...Loaded " << neg_patches << " patches" << endl;
        }

        // Train forest and save file
        crForest.trainForest(trees_leaf_size, trees_max_depth, &cvRNG, Train,
                             trees_pixel_tests, trees_prob_optimize_mode,
                             trees_path_string.c_str(), trees_offset, patch_width,
                             patch_height, serial, verbose && ! quiet);
    }

    // Run detector
    int detect(CRForest *forest, string input_gpath, string output_gpath,
               string output_scale_gpath, int mode, float sensitivity,
               vector<float> &scale_vector, int nms_min_area_contour,
               int nms_min_area_overlap, float **results, int RESULT_LENGTH,
               bool serial, bool verbose, bool quiet)
    {
        // This threshold value is important, but not really because it can be controlled
        // with the sensitivity value
        // int threshold = int(255.0 * 0.90);
        bool debug_flag = true;
        int threshold = int(255 * 0.90);
        int accumulate_mode = 1; // 1 - max, 0 - add | 0 - hough, 1 - classification
        float density = 0.990;
        verbose = true;

        // Load forest into detector object
        CRForestDetector crDetect(forest);
        char buffer[512];

        if(serial)
        {
            cout << "[pyrf c++] SERIAL" << endl;
        }
        else
        {
            cout << "[pyrf c++] PARALELL" << endl;
        }
        // Storage for output
        vector<IplImage *> vImgDetect(scale_vector.size());
        vector<vector<vector<vector<CvPoint > > > > manifests(scale_vector.size());
        // <VECTOR> | scale | y | x | cvPoint

        // Load image and create temporary objects
        IplImage *img = cvLoadImage(input_gpath.c_str(), CV_LOAD_IMAGE_COLOR);
        #pragma omp critical(imageLoad)
        {
            if (!img)
            {
                cout << "[pyrf c++] Could not load image file: " << input_gpath << endl;
                exit(-1);
            }
            else
            {
                if(! quiet)
                {
                    cout << "[pyrf c++] Loaded image file: " << input_gpath << endl;
                }
            }
        }

        IplImage *combined = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
        IplImage *combinedAdd = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
        IplImage *combinedMax = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
        IplImage *upscaled = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
        IplImage *output   = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U,  1);
        IplImage *debug    = cvLoadImage(input_gpath.c_str(), CV_LOAD_IMAGE_COLOR);

        // Prepare scale_vector
        int w, h, k;
        for (k = 0; k < vImgDetect.size(); ++k)
        {
            w = int(img->width * scale_vector[k] + 0.5);
            h = int(img->height * scale_vector[k] + 0.5);
            vImgDetect[k] = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1 );
            // manifests
            manifests[k].resize(h);
            for (int y = 0; y < h; ++y)
            {
                manifests[k][y].resize(w);
            }
        }

        // Detection for all scale_vector
        crDetect.detectPyramid(img, vImgDetect, manifests, scale_vector, mode, serial);

        // Create combined images for add and max
        vector<vector<float> > temp;
        cvSet(combined, cvScalar(0));    // Set combined to 0
        cvSet(combinedAdd, cvScalar(0)); // Set combinedAdd to 0
        cvSet(combinedMax, cvScalar(0)); // Set combinedMax to 0
        for (k = 0; k < vImgDetect.size(); k++)
        {
            // Add scale to combined
            cvResize(vImgDetect[k], upscaled);
            // Smooth the result
            cvSmooth( upscaled, upscaled, CV_GAUSSIAN, 5);
            cvAdd(upscaled, combinedAdd, combinedAdd);
            cvMax(upscaled, combinedMax, combinedMax);
            // Before we release, output the scale
            if (output_scale_gpath.length() > 0)
            {
                // Save scale output mode image
                cvConvertScale( vImgDetect[k], vImgDetect[k], sensitivity);
                cvSmooth( vImgDetect[k], vImgDetect[k], CV_GAUSSIAN, 5);
                sprintf(buffer, "%s_scaled_%0.02f.JPEG", output_scale_gpath.c_str(), scale_vector[k]);
                cvSaveImage(buffer, vImgDetect[k]);
            }
            // Release images
            cvReleaseImage(&vImgDetect[k]);
        }
        // Release memory
        vImgDetect.clear();
        cvReleaseImage(&upscaled);

        // Take minimum of add and max, this will give good negatives and good centers.
        if(mode == 0)
        {
            cvSmooth( combinedAdd, combinedAdd, CV_GAUSSIAN, 5);
            cvSmooth( combinedMax, combinedMax, CV_GAUSSIAN, 5);

            CvPoint minloc, maxloc;
            double minvalAdd, maxvalAdd;
            double minvalMax, maxvalMax;
            cvMinMaxLoc(combinedAdd, &minvalAdd, &maxvalAdd, &minloc, &maxloc, 0);
            cvMinMaxLoc(combinedMax, &minvalMax, &maxvalMax, &minloc, &maxloc, 0);

            // Create combined
            cvConvertScale( combinedMax, combined, maxvalAdd / maxvalMax );
            cvMax( combined, combinedAdd, combined );

            // Smooth the image
            cvSmooth( combined, combined, CV_GAUSSIAN, 5);

            // Find strength
            double minvalComb, maxvalComb;
            cvMinMaxLoc(combined, &minvalComb, &maxvalComb, &minloc, &maxloc, 0);

            if(debug_flag && ! quiet)
            {
                #pragma omp critical(detectionStrengths)
                {
                    cout << "[pyrf c++]" << endl;
                    cout << "[pyrf c++] ADD  Detected - min: " << minvalAdd  << ", max: " << maxvalAdd  << " / " << scale_vector.size() << endl;
                    cout << "[pyrf c++] MAX  Detected - min: " << minvalMax  << ", max: " << maxvalMax  << " / " << scale_vector.size() << endl;
                    cout << "[pyrf c++] COMB Detected - min: " << minvalComb << ", max: " << maxvalComb << " / " << scale_vector.size() << endl;
                }
            }

            if(debug_flag)
            {
                if (output_gpath.length() > 0)
                {
                    sprintf(buffer, "%s_debug_add.JPEG", output_gpath.c_str());
                    cvConvertScale( combinedAdd, combinedAdd, sensitivity / scale_vector.size() );
                    cvSaveImage(buffer, combinedAdd);

                    sprintf(buffer, "%s_debug_max.JPEG", output_gpath.c_str());
                    cvConvertScale( combinedMax, combinedMax, sensitivity );
                    cvSaveImage(buffer, combinedMax);
                }
            }

            // Release memory
            cvReleaseImage(&combinedAdd);
            cvReleaseImage(&combinedMax);

            // Scale to output
            cvConvertScale( combined, output, sensitivity / scale_vector.size() );
            cvSmooth( output, output, CV_GAUSSIAN, 5);

            // Release memory
            cvReleaseImage(&combined);

            if (output_gpath.length() > 0)
            {
                // Save output mode image
                cvSaveImage(output_gpath.c_str(), output);
            }

            // if (output_gpath.length() > 0)
            // {
            //     cvErode(output, output, NULL, 3);
            //     cvSmooth( output, output, CV_GAUSSIAN, 5);
            //     // Save output mode image
            //     sprintf(buffer, "%s_dialate.JPEG", output_gpath.c_str());
            //     cvSaveImage(buffer, output);
            // }

            cvThreshold(output, output, threshold, 0, CV_THRESH_TOZERO);

            if(debug_flag)
            {
                if (output_gpath.length() > 0)
                {
                    // Save output mode image
                    sprintf(buffer, "%s_debug_thresh.JPEG", output_gpath.c_str());
                    cvSaveImage(buffer, output);
                }
            }

            // Calculate contours for scaled image
            CvSeq *contours;
            CvMemStorage *storage = cvCreateMemStorage(0);
            cvFindContours(output, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
            cvClearMemStorage(storage);

            CvRect rect;
            float centerx, centery, xtl, ytl, width, height, confidence, supressed;
            vector<int> left, right, bottom, top;
            int minx, maxx, miny, maxy;
            int x_, y_;
            int i, x, y, j;
            int ls, bs, rs, ts;

            /////// DEBUG ///////
            int red, green, blue;
            int xm, ym;
            time_t t = time(NULL);
            int seed = (int) t;
            CvRNG cvRNG(seed);
            uchar* ptr;
            /////// DEBUG ///////

            for (i = 0; contours != 0; contours = contours->h_next, ++i)
            {
                rect = cvBoundingRect(contours);
                if(rect.width * rect.height >= nms_min_area_contour)
                {

                    centerx   = rect.x + (rect.width  / 2);
                    centery   = rect.y + (rect.height / 2);
                    xm = int(rect.width  * 0.10);
                    ym = int(rect.height * 0.10);

                    if(debug_flag)
                    {
                        red   = cvRandInt( &cvRNG ) % 256;
                        green = cvRandInt( &cvRNG ) % 256;
                        blue  = cvRandInt( &cvRNG ) % 256;
                    }

                    left.clear();
                    right.clear();
                    bottom.clear();
                    top.clear();

                    // #pragma omp critical(memoryIntensive)
                    // {
                    for(k = 0; k < manifests.size(); ++k)
                    {
                        minx = std::max(int((centerx - rect.width)  * scale_vector[k]), 0);
                        maxx = std::min(int((centerx + rect.width)  * scale_vector[k]), int(manifests[k][0].size()));
                        miny = std::max(int((centery - rect.height) * scale_vector[k]), 0);
                        maxy = std::min(int((centery + rect.height) * scale_vector[k]), int(manifests[k].size()));

                        for(y = miny; y < maxy; ++y)
                        {
                            for(x = minx; x < maxx; ++x)
                            {
                                for (j = 0; j < manifests[k][y][x].size(); ++j)
                                {
                                    x_ = int(manifests[k][y][x][j].x / scale_vector[k]);
                                    y_ = int(manifests[k][y][x][j].y / scale_vector[k]);

                                    if(debug_flag)
                                    {
                                        ptr = (uchar*) ( debug->imageData + y_ * debug->widthStep );
                                        ptr[3 * x_ + 0] = blue;
                                        ptr[3 * x_ + 1] = green;
                                        ptr[3 * x_ + 2] = red;
                                    }

                                    if(x_ < centerx)
                                    {
                                        left.push_back(x_);
                                    }
                                    else
                                    {
                                        right.push_back(x_);
                                    }
                                    if(y_ < centery)
                                    {
                                        bottom.push_back(y_);
                                    }
                                    else
                                    {
                                        top.push_back(y_);
                                    }
                                }
                                // Release memory (! effects the bounding box regressions !)
                                // manifests[k][y][x].clear();
                            }
                        }
                    }

                    ls = left.size();
                    bs = bottom.size();
                    rs = right.size();
                    ts = top.size();
                    // cout << left.size() << " " << bottom.size() << " " << right.size() << " " << top.size() << endl;

                    if(debug_flag)
                    {
                        cvCircle(debug, cvPoint(centerx, centery), 3, cvScalar(0, 0, 255), -1);

                        if(ls > 0)
                        {
                            xtl = accumulate(left.begin(), left.end(), 0.0) / ls;
                        }
                        else
                        {
                            xtl = centerx;
                        }
                        if(bs > 0)
                        {
                            ytl = accumulate(bottom.begin(), bottom.end(), 0.0) / bs;
                        }
                        else
                        {
                            ytl = centery;
                        }
                        if(rs > 0)
                        {
                            width = accumulate(right.begin(), right.end(), 0.0) / rs;
                        }
                        else
                        {
                            width = centerx;
                        }
                        if(ts > 0)
                        {
                            height = accumulate(top.begin(), top.end(), 0.0) / ts;
                        }
                        else
                        {
                            height = centery;
                        }
                        cvRectangle(debug, cvPoint(xtl, ytl), cvPoint(width, height), cvScalar(0, 0, 255), 3);

                        if(ls > 0)
                        {
                            xtl = *min_element( left.begin(), left.end() );
                        }
                        else
                        {
                            xtl = centerx;
                        }
                        if(bs > 0)
                        {
                            ytl = *min_element( bottom.begin(), bottom.end() );
                        }
                        else
                        {
                            ytl = centery;
                        }
                        if(rs > 0)
                        {
                            width = *max_element( right.begin(), right.end() );
                        }
                        else
                        {
                            width = centerx;
                        }
                        if(ts > 0)
                        {
                            height = *max_element( top.begin(), top.end() );
                        }
                        else
                        {
                            height = centery;
                        }
                        // cout << xtl << " " << ytl << " " << width << " " << height << endl;
                        cvRectangle(debug, cvPoint(xtl, ytl), cvPoint(width, height), cvScalar(0, 255, 255), 3);

                    }

                    if(ls > 0)
                    {
                        std::sort(left.begin(), left.end(), std::greater<int>());
                        xtl    = left[int(density * left.size())];
                    }
                    else
                    {
                        xtl = centerx;
                    }
                    if(bs > 0)
                    {
                        std::sort(bottom.begin(), bottom.end(), std::greater<int>());
                        ytl    = bottom[int(density * bottom.size())];
                    }
                    else
                    {
                        ytl = centery;
                    }
                    if(rs > 0)
                    {
                        std::sort(right.begin(), right.end());
                        width  = right[int(density * right.size())];
                    }
                    else
                    {
                        width = centerx;
                    }
                    if(ts > 0)
                    {
                        std::sort(top.begin(), top.end());
                        height = top[int(density * top.size())];
                    }
                    else
                    {
                        height = centery;
                    }

                    if(debug_flag)
                    {
                        cvRectangle(debug, cvPoint(xtl, ytl), cvPoint(width, height), cvScalar(255, 255, 0), 3);
                    }

                    // Free up space
                    // left.clear();
                    // right.clear();
                    // bottom.clear();
                    // top.clear();
                    // }

                    // Fix width and height
                    width  -= xtl;
                    height -= ytl;
                    confidence = 0.0;
                    supressed = 0.0;

                    vector<float> temp_(RESULT_LENGTH);
                    temp_[0] = centerx;
                    temp_[1] = centery;
                    temp_[2] = xtl;
                    temp_[3] = ytl;
                    temp_[4] = width;
                    temp_[5] = height;
                    temp_[6] = confidence;
                    temp_[7] = supressed;
                    temp.push_back(temp_);
                }
            }

            // Free up space
            left.clear();
            right.clear();
            bottom.clear();
            top.clear();
            cvReleaseMemStorage(&storage);
        }
        else if(mode == 1)
        {
            //#define USE_MAX 0
            #ifdef USE_MAX
            cvConvertScale( combinedMax, output, sensitivity );
            #else
            cvConvertScale( combinedAdd, output, sensitivity / scale_vector.size() );
            #endif

            // Release memory
            cvReleaseImage(&combinedAdd);
            cvReleaseImage(&combinedMax);
            cvReleaseImage(&combined);

            if (output_gpath.length() > 0)
            {
                // Save output mode image
                cvSaveImage(output_gpath.c_str(), output);
            }
        }

        // free memory
        manifests.clear();

        if(debug_flag)
        {
            if(output_gpath.length() > 0)
            {
                sprintf(buffer, "%s_debug_original.JPEG", output_gpath.c_str());
                cvSaveImage(buffer, img);

                sprintf(buffer, "%s_debug.JPEG", output_gpath.c_str());
                cvSaveImage(buffer, debug);
            }
        }

        // Release image
        cvReleaseImage(&img);
        cvReleaseImage(&output);
        cvReleaseImage(&debug);

        // Save results
        int size = temp.size();
        *results = new float[ size * RESULT_LENGTH ];
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < RESULT_LENGTH; ++j)
            {
                (*results)[i * RESULT_LENGTH + j] = temp[i][j];
            }
        }
        temp.clear();
        return size;
    }

private:
    // Extract patches from training data
    int extract_Patches(CRPatch &Train, CvRNG *pRNG, string train_pos_chip_path_string,
                        vector<string> &train_pos_chip_filename_vector, int label,
                        float patch_density, int max_patches, bool verbose, bool quiet)
    {
        string img_filepath;
        IplImage *img;
        int patch_total = 0;
        // load postive images and extract patches
        for (int i = 0; i < train_pos_chip_filename_vector.size(); ++i)
        {
            if (patch_total > max_patches)
            {
                if (verbose && ! quiet)
                {
                    cout << endl << "[pyrf c++] Skipping image file: " << img_filepath;
                }
                continue;
            }
            // Print status
            if( ! quiet )
            {
                if (i % 10 == 0) cout << i << " " << flush;
            }
            // Get the image's filepah
            img_filepath = train_pos_chip_path_string + "/" + train_pos_chip_filename_vector[i];
            // Load image
            img = cvLoadImage(img_filepath.c_str(), CV_LOAD_IMAGE_COLOR);
            if (!img)
            {
                cout << endl << "[pyrf c++] Could not load image file: " << img_filepath;
                continue;
            }
            // Extract patches
            patch_total += Train.extractPatches(img, patch_density, label);
            // Release image
            cvReleaseImage(&img);
        }
        return patch_total;
    }

    int pdf(vector<int> &data, float percentile)
    {
        // INVOCATION:
        //     xtl    = pdf(left,   density);
        //     ytl    = pdf(bottom, density);
        //     width  = pdf(right,  density);
        //     height = pdf(top,    density);
        int val = 0, counter = 0, total = 0;
        int cutoff = int(percentile * data.size());
        for(int c = 0; c < data.size(); ++c)
        {
            if(data[c] != val || c == data.size() - 1)
            {
                total += counter;
                if(total >= cutoff)
                {
                    break;
                }
                val = data[c];
                counter = 0;
            }
            else
            {
               counter++;
            }
        }
        return val;
    }
};
