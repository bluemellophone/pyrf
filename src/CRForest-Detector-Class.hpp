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
#include <climits>

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
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        // INIT / PRE VOTING
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        // This threshold value is important, but not really because it can be controlled
        // with the sensitivity value
        bool debug_flag = true;
        int threshold = int(255 * 0.99);
        float density = 0.990;

        // Load forest into detector object
        CRForestDetector crDetect(forest);
        char buffer[512];

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

        // Output images and storage for output information from voting
        IplImage *combined = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
        IplImage *upscaled = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
        IplImage *output   = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U,  1);
        IplImage *debug    = cvLoadImage(input_gpath.c_str(), CV_LOAD_IMAGE_COLOR);

        vector<IplImage *> vImgDetect(scale_vector.size());
        vector<vector<vector<vector<CvPoint > > > > manifests(scale_vector.size());
        vector<CvRect > peaks;
        // <VECTOR> | scale | y | x | cvPoint
        
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        // VOTING / RE-VOTING
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        bool revote = true;
        int voting_round = 0;
        CvRect rect;
        int i, x, y, j, k, w, h;

        while(revote)
        {
            voting_round++;

            // Prepare vImgDetect and manifest
            for (k = 0; k < vImgDetect.size(); ++k)
            {
                w = int(img->width * scale_vector[k] + 0.5);
                h = int(img->height * scale_vector[k] + 0.5);
                vImgDetect[k] = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1 );
                // manifests
                manifests[k].clear();
                manifests[k].resize(h);
                for (y = 0; y < h; ++y)
                {
                    manifests[k][y].resize(w);
                }
            }

            // Detection for all scale_vector
            crDetect.detectPyramid(img, vImgDetect, manifests, scale_vector, peaks, mode, serial);

            // Create combined images for add and max
            cvSet(combined, cvScalar(0));    // Set combined to 0
            for (k = 0; k < vImgDetect.size(); k++)
            {
                // Add scale to combined
                cvResize(vImgDetect[k], upscaled);
                // Smooth the result
                cvSmooth( upscaled, upscaled, CV_GAUSSIAN, 5);
                cvAdd(upscaled, combined, combined);
                // Before we release, output the scale
                if (output_scale_gpath.length() > 0)
                {
                    // Save scale output mode image
                    cvConvertScale( vImgDetect[k], vImgDetect[k], sensitivity);
                    cvSmooth( vImgDetect[k], vImgDetect[k], CV_GAUSSIAN, 5);
                    sprintf(buffer, "%s_scaled__%d_%0.02f.JPEG", output_scale_gpath.c_str(), voting_round, scale_vector[k]);
                    cvSaveImage(buffer, vImgDetect[k]);
                }
                // Release images
                cvReleaseImage(&vImgDetect[k]);
            }
        
            // Erode the image
            // IplConvKernel* element = cvCreateStructuringElementEx(5, 5, 1, 1, CV_SHAPE_RECT);
            // cvErode( combined, combined, element, 3);

            // Smooth the image
            cvSmooth( combined, combined, CV_GAUSSIAN, 5);

            // Take minimum of add and max, this will give good negatives and good centers.
            if(mode == 0)
            {
                // Scale combined to output and threshold
                cvConvertScale( combined, output, sensitivity / scale_vector.size() );

                if (output_gpath.length() > 0)
                {
                    // Save output mode image

                    sprintf(buffer, "%s_%d.JPEG", output_gpath.c_str(), voting_round);
                    cvSaveImage(buffer, output);
                }

                cvThreshold(output, output, threshold, 0, CV_THRESH_TOZERO);
        
                if(debug_flag)
                {
                    if (output_gpath.length() > 0)
                    {
                        // Save output mode image
                        sprintf(buffer, "%s_%d_debug_thresh.JPEG", output_gpath.c_str(), voting_round);
                        cvSaveImage(buffer, output);
                    }   
                }

                // Calculate contours for output image
                CvSeq *contours;
                CvMemStorage *storage = cvCreateMemStorage(0);
                cvFindContours(output, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
                cvClearMemStorage(storage);

                // Find peaks
                peaks.clear();
                for (i = 0; contours != 0; contours = contours->h_next, ++i)
                {    
                    rect = cvBoundingRect(contours);
                    if(rect.width * rect.height >= nms_min_area_contour)
                    {
                        peaks.push_back(rect);
                    }
                }
                cvReleaseMemStorage(&storage);

                if(voting_round == 2)
                {
                    revote = false;
                }
            }
            else
            {
                revote = false;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        // POST REVOTING 
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        vector<vector<float> > temp;
        if(mode == 0)
        {
            if (output_gpath.length() > 0)
            {
                // Scale combined to output
                cvConvertScale( combined, output, sensitivity / scale_vector.size() );
                // Save output mode image
                cvSaveImage(output_gpath.c_str(), output);
            }

            if(debug_flag)
            {
                // Scale combined to output and threshold
                cvThreshold(output, output, threshold, 0, CV_THRESH_TOZERO);
                if (output_gpath.length() > 0)
                {
                    // Save output mode image
                    sprintf(buffer, "%s_debug_thresh.JPEG", output_gpath.c_str());
                    cvSaveImage(buffer, output);
                }   
            }

            int x_, y_;
            int cx, cy;
            int minx, miny, maxx, maxy;
            int pminx, pminy, pmaxx, pmaxy;
            int tminx, tminy, tmaxx, tmaxy;
            int cminx, cminy, cmaxx, cmaxy;
            int aminx, aminy, amaxx, amaxy;

            float confidence, supressed;

            vector<int> left, bottom, right, top;
            int red, green, blue;
            time_t t = time(NULL);
            int seed = (int) t;
            CvRNG cvRNG(seed);
            CvPoint point;
            uchar* ptr;

            for (i = 0; i < peaks.size(); ++i)
            {    
                rect = peaks[i];
                cx = int(rect.x + (rect.width  / 2));
                cy = int(rect.y + (rect.height / 2));

                if(debug_flag)
                {
                    red   = cvRandInt( &cvRNG ) % 256;
                    green = cvRandInt( &cvRNG ) % 256;
                    blue  = cvRandInt( &cvRNG ) % 256;
                }

                left.clear(); bottom.clear(); right.clear(); top.clear();

                left.push_back(cx);
                bottom.push_back(cy);
                right.push_back(cx);
                top.push_back(cy);

                pminx = INT_MAX; pminy = INT_MAX; pmaxx = 0; pmaxy = 0;
                tminx = cx; tminy = cy; tmaxx = cx; tmaxy = cy;
                cminx = 1; cminy = 1; cmaxx = 1; cmaxy = 1;
                
                for(k = 0; k < manifests.size(); ++k)
                {
                    minx = (rect.x)               * scale_vector[k];
                    miny = (rect.y)               * scale_vector[k];
                    maxx = (rect.x + rect.width)  * scale_vector[k];
                    maxy = (rect.y + rect.height) * scale_vector[k];

                    minx = max(minx, 0);
                    miny = max(miny, 0);
                    maxx = min(maxx, int(manifests[k][0].size()));
                    maxy = min(maxy, int(manifests[k].size()));

                    for(y = miny; y < maxy; ++y)
                    {
                        for(x = minx; x < maxx; ++x)
                        {
                            for (j = 0; j < manifests[k][y][x].size(); ++j)
                            {
                                point = manifests[k][y][x][j];
                                x_ = int(point.x / scale_vector[k]);
                                y_ = int(point.y / scale_vector[k]);

                                if(debug_flag)
                                {
                                    ptr = (uchar*) ( debug->imageData + y_ * debug->widthStep );
                                    ptr[3 * x_ + 0] = blue;
                                    ptr[3 * x_ + 1] = green;
                                    ptr[3 * x_ + 2] = red;
                                }

                                pminx = min(pminx, x_);
                                pminy = min(pminy, y_);
                                pmaxx = max(pmaxx, x_);
                                pmaxy = max(pmaxy, y_);

                                if(x_ < cx)
                                {
                                    tminx += x_;
                                    cminx++;
                                    left.push_back(x_);
                                }
                                else
                                {
                                    tmaxx += x_;
                                    cmaxx++;
                                    right.push_back(x_);
                                }
                                if(y_ < cy)
                                {
                                    tminy += y_;
                                    cminy++;
                                    bottom.push_back(y_);
                                }
                                else
                                {
                                    tmaxy += y_;
                                    cmaxy++;
                                    top.push_back(y_);
                                }
                            }
                        }
                    }
                }

                aminx = int(tminx / cminx);
                aminy = int(tminy / cminy);
                amaxx = int(tmaxx / cmaxx);
                amaxy = int(tmaxy / cmaxy);

                if(debug_flag)
                {
                    cvCircle(debug, cvPoint(cx, cy), 3, cvScalar(0, 0, 255), -1);
                    cvRectangle(debug, cvPoint(aminx, aminy), cvPoint(amaxx, amaxy), cvScalar(0, 0, 255), 3);
                    cvRectangle(debug, cvPoint(pminx, pminy), cvPoint(pmaxx, pmaxy), cvScalar(0, 255, 255), 3);
                }
                
                // TODO: Replace this with a min, max heap
                sort(left.begin(), left.end(), greater<int>());
                sort(bottom.begin(), bottom.end(), greater<int>());
                sort(right.begin(), right.end());
                sort(top.begin(), top.end());

                // Reuse these variables
                minx = left[int(density * left.size())];
                miny = bottom[int(density * bottom.size())];
                maxx = right[int(density * right.size())];
                maxy = top[int(density * top.size())];

                if(debug_flag)
                {   
                    cvRectangle(debug, cvPoint(minx, miny), cvPoint(maxx, maxy), cvScalar(255, 255, 0), 3);   
                }

                // Fix width and height
                confidence = 0.0;
                supressed = 0.0;

                vector<float> temp_(RESULT_LENGTH);
                temp_[0] = cx;
                temp_[1] = cy;
                temp_[2] = minx;
                temp_[3] = miny;
                temp_[4] = maxx - minx;
                temp_[5] = maxy - miny;
                temp_[6] = confidence;
                temp_[7] = supressed;
                temp.push_back(temp_);
            }
        }
        else if(mode == 1)
        {
            cvConvertScale( combined, output, sensitivity / scale_vector.size() );
            if (output_gpath.length() > 0)
            {
                // Save output mode image
                cvSaveImage(output_gpath.c_str(), output);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        // CLEAN UP
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
            
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
        cvReleaseImage(&combined);
        cvReleaseImage(&upscaled);
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
};
