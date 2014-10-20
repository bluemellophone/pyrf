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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv/highgui.h>

#include "CRForestDetector.h"

using namespace std;

struct CRForestDetectorClass
{
	public:
		int 	patch_width;
		int 	patch_height;
		int 	out_scale;
		int 	default_split;
		int 	positive_like;
	  	bool 	legacy;
	  	bool 	include_horizontal_flip;
		int 	patch_sample_density_pos;
		int 	patch_sample_density_neg;
		vector<float> scales;
		vector<float> ratios;
		vector<vector<vector<float> > > points;

	public:
		// Constructor
		CRForestDetectorClass(
			int 	param_patch_width,
			int 	param_patch_height,
			int 	param_out_scale,
			int 	param_default_split,
			int 	param_positive_like,
		  	bool 	param_legacy,
		  	bool 	param_include_horizontal_flip,
			int 	param_patch_sample_density_pos,
			int 	param_patch_sample_density_neg,
			char*	param_scales,
			char*	param_ratios
			)
		{
			#ifdef _OPENMP
			cout << "\n\n-----------------------------\n\nRUNNING PROGRAM IN PARALLEL\n\n-----------------------------\n\n" << endl;
			#else
			cout << "\n\n-----------------------------\n\nRUNNING PROGRAM IN SERIAL\n\n-----------------------------\n\n" << endl;
			#endif

			patch_width = 				param_patch_width;
			patch_height = 				param_patch_height;
			out_scale = 				param_out_scale;
			default_split = 			param_default_split;
			positive_like = 			param_positive_like;
		  	legacy = 					param_legacy;
		  	include_horizontal_flip = 	param_include_horizontal_flip;
			patch_sample_density_pos = 	param_patch_sample_density_pos;
			patch_sample_density_neg = 	param_patch_sample_density_neg;

			int size;
			string str_param_scales = param_scales;
			string str_param_ratios = param_ratios;

			istringstream iss1 (str_param_scales, istringstream::in);

			iss1 >> size;
			scales.resize(size);
			for(int i = 0; i < size; ++i)
			{
				iss1 >> scales[i];
			}

			istringstream iss2 (str_param_ratios, istringstream::in);
			iss2 >> size;
			ratios.resize(size);
			for(int i = 0; i < size; ++i)
			{
				iss2 >> ratios[i];
			}

		}

        CRForestDetectorClass(const CRForestDetectorClass& other)
        {
            this->patch_width              = other.patch_width;
            this->patch_height             = other.patch_height;
            this->out_scale                = other.out_scale;
            this->default_split            = other.default_split;
            this->positive_like            = other.positive_like;
            this->legacy                   = other.legacy;
            this->include_horizontal_flip  = other.include_horizontal_flip;
            this->patch_sample_density_pos = other.patch_sample_density_pos;
            this->patch_sample_density_neg = other.patch_sample_density_neg;
            this->scales = other.scales;
            this->ratios = other.ratios;
            //this->points = other.points;
        }

		// load test image filenames
		void loadImFile(std::vector<string>& vFilenames, string detection_inventory) {

			char buffer[400];

			ifstream in(detection_inventory.c_str());
			if(in.is_open()) {

				unsigned int size;
				in >> size; //size = 10;
				in.getline(buffer,400);
				vFilenames.resize(size);

				for(unsigned int i=0; i<size; ++i) {
					in.getline(buffer,400);
					vFilenames[i] = buffer;
				}

			} else {
				cerr << "File not found " << detection_inventory.c_str() << endl;
				exit(-1);
			}

			in.close();
		}

		// load positive training image filenames
		void loadTrainPosFile(	std::vector<string>& vFilenames,
								std::vector<CvRect>& vBBox,
								std::vector<std::vector<CvPoint> >& vCenter,
								string training_inventory_pos) {

			unsigned int size, numop;
			ifstream in(training_inventory_pos.c_str());

			if(in.is_open()) {
				in >> size;
				in >> numop;
				cout << "Load Train Pos Examples: " << size << " - " << numop << endl;

				vFilenames.resize(size);
				vCenter.resize(size);
				vBBox.resize(size);

				for(unsigned int i=0; i<size; ++i) {
					// Read filename
					in >> vFilenames[i];

					// Read bounding box
					in >> vBBox[i].x; in >> vBBox[i].y;
					in >> vBBox[i].width;
					vBBox[i].width -= vBBox[i].x;
					in >> vBBox[i].height;
					vBBox[i].height -= vBBox[i].y;

					if(vBBox[i].width<patch_width || vBBox[i].height<patch_height) {
					  cout << "Width or height are too small" << endl;
					  cout << vFilenames[i] << endl;
					  exit(-1);
					}

					// Read center points
					vCenter[i].resize(numop);
					for(unsigned int c=0; c<numop; ++c) {
						in >> vCenter[i][c].x;
						in >> vCenter[i][c].y;
					}
				}

				in.close();
			} else {
				cerr << "File not found " << training_inventory_pos.c_str() << endl;
				exit(-1);
			}
		}

		// load negative training image filenames
		void loadTrainNegFile(	std::vector<string>& vFilenames,
								std::vector<CvRect>& vBBox,
								string training_inventory_neg) {

			unsigned int size, numop;
			ifstream in(training_inventory_neg.c_str());

			if(in.is_open()) {
				in >> size;
				in >> numop;
				cout << "Load Train Neg Examples: " << size << " - " << numop << endl;

				vFilenames.resize(size);
				if(numop>0)
					vBBox.resize(size);
				else
					vBBox.clear();

				for(unsigned int i=0; i<size; ++i) {
					// Read filename
					in >> vFilenames[i];

					// Read bounding box (if available)
					if(numop>0) {
						in >> vBBox[i].x; in >> vBBox[i].y;
						in >> vBBox[i].width;
						vBBox[i].width -= vBBox[i].x;
						in >> vBBox[i].height;
						vBBox[i].height -= vBBox[i].y;

						if(vBBox[i].width<patch_width || vBBox[i].height<patch_height) {
						  cout << "Width or height are too small" << endl;
						  cout << vFilenames[i] << endl;
						  exit(-1);
						}
					}
				}

				in.close();
			} else {
				cerr << "File not found " << training_inventory_neg.c_str() << endl;
				exit(-1);
			}
		}

		bool new_center(std::vector<vector<vector<float> > >& points, float nms_margin_percentage, int centerx, int centery, int minx, int miny, int maxx, int maxy)
		{
			float distance;
			int left1, right1, bottom1, top1, left2, right2, bottom2, top2;
			for(unsigned int k=0;k<points.size(); ++k) {
				for(unsigned int i=0;i<points[k].size(); ++i) {
					left1 = points[k][i][0] - ((points[k][i][0] - points[k][i][2]) * nms_margin_percentage);
					right1 = points[k][i][0] + ((points[k][i][4] - points[k][i][0]) * nms_margin_percentage);
					bottom1 = points[k][i][1] - ((points[k][i][1] - points[k][i][3]) * nms_margin_percentage);
					top1 = points[k][i][1] + ((points[k][i][5] - points[k][i][1]) * nms_margin_percentage);

					// left2 = centerx - ((centerx - minx) * nms_margin_percentage);
					// right2 = centerx + ((maxx - centerx) * nms_margin_percentage);
					// bottom2 = centery - ((centery - miny) * nms_margin_percentage);
					// top2 = centery + ((maxy - centery) * nms_margin_percentage);

					if(left1 <= centerx && centerx <= right1 && bottom1 <= centery && centery <= top1)
					   //left2 <= points[k][i][0] && points[k][i][0] <= right2 && bottom2 <= points[k][i][1] && points[k][i][1] <= top2)
					{
						return false;
					}
				}
			}

			return true;
		}

		// Run detector
		int detect(CRForestDetector& crDetect,
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
			char buffer[512];  // was 200

			// Storage for output
			vector<vector<IplImage*> > vImgDetect(scales.size());

			// Load image
			IplImage *img = 0;
			img = cvLoadImage(detection_image_filepath,CV_LOAD_IMAGE_COLOR);
			if(!img) {
				cout << "Could not load image file: " << detection_image_filepath << endl;
				exit(-1);
			}

			// Prepare scales
			for(unsigned int k=0;k<vImgDetect.size(); ++k) {
				vImgDetect[k].resize(ratios.size());
				for(unsigned int c=0;c<vImgDetect[k].size(); ++c) {
					vImgDetect[k][c] = cvCreateImage( cvSize(int(img->width*scales[k]+0.5),int(img->height*scales[k]+0.5)), IPL_DEPTH_32F, 1 );
				}
			}

			// Detection for all scales
			crDetect.detectPyramid(img, vImgDetect, ratios, positive_like, legacy);

			// Store result of all scales
			IplImage* combined = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U , 1);
			IplImage* temp = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U , 1);
            cvSet(combined, cvScalar(0));  // Combined needs to be initialized to 0

			// Prepare results vector
			points.clear();
			points.resize(scales.size());
			double minVal; double maxVal;
			for(int k=vImgDetect.size() - 1;k >= 0; --k)
			{
				// cout << k;
				IplImage* scaled = cvCreateImage( cvSize(vImgDetect[k][0]->width,vImgDetect[k][0]->height) , IPL_DEPTH_8U , 1);

				for(unsigned int c=0;c<vImgDetect[k].size(); ++c)
				{
					// Find confidence
					cvMinMaxLoc(vImgDetect[k][c], &minVal, &maxVal);

					// Resize image
					cvConvertScale(vImgDetect[k][c], scaled, out_scale);
					cvResize(scaled, temp);

					// Save detection
					if(save_scales)
					{
						sprintf_s(buffer,"%s_scaled_%.2f_%d.png",detection_result_filepath,scales[k],c);
						cvSaveImage( buffer, scaled );
					}

					// Accumulate max image
					cvMax(combined, temp, combined);

					// Threshold scaled image
					cvThreshold(scaled, scaled, 250, 0, CV_THRESH_TOZERO);
					// sprintf_s(buffer,"%s_threshold_%.2f_%d.png", detection_result_filepath, scales[k], c);
					// cvSaveImage( buffer, scaled );

					// Calculate contours for scaled image
					CvSeq* contours = 0;
					CvMemStorage* storage = cvCreateMemStorage(0);
					cvFindContours(scaled, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

					// Draw contours on image
					// cvDrawContours(scaled, contours, cvScalar(255,255,255), cvScalarAll(255), 100);
					// sprintf_s(buffer,"%s_contours_%.2f_%d.png", detection_result_filepath, scales[k], c);
					// cvSaveImage( buffer, scaled );

					CvRect rect;
					int centerx, centery, minx, miny, maxx, maxy, area, supressed;

					int i = 0;
					for (int i = 0; contours != 0; contours = contours->h_next, ++i)
					{
						rect = cvBoundingRect(contours);

						centerx = (int) ((rect.x)+rect.width/2) / scales[k];
						centery = (int) ((rect.y)+rect.height/2) / scales[k];
						minx = (int) centerx - ((detection_width * (percentage_left)) / scales[k]);
						miny = (int) centery - ((detection_height * (percentage_top)) / scales[k]);
						maxx = (int) centerx + ((detection_width * (1 - percentage_left)) / scales[k]);
						maxy = (int) centery + ((detection_height * (1 - percentage_top)) / scales[k]);
						area = (maxx - minx) * (maxy - miny);

						if(area > min_contour_area && new_center(points, nms_margin_percentage, centerx, centery, minx, miny, maxx, maxy))
						{
							supressed = 0;
						}
						else
						{
							supressed = 1;
						}

						vector<float> temp;
						temp.push_back((float) centerx);
						temp.push_back((float) centery);
						temp.push_back((float) minx);
						temp.push_back((float) miny);
						temp.push_back((float) maxx);
						temp.push_back((float) maxy);
						temp.push_back((float) maxVal);
						temp.push_back((float) supressed);

						points[k].push_back(temp);

					}
					cvReleaseImage(&vImgDetect[k][c]);
				}
				cvReleaseImage(&scaled);

				// cout << k << endl;
			}

			int red, green, blue;
			int num_detections = 0;
			for(unsigned int k=0;k<points.size(); ++k) {
				// cout << "Scale: " << scales[k] << endl;
				for(unsigned int i=0;i<points[k].size(); ++i) {
					num_detections++;

					// Print on screen
					// if(points[k][i][7] == 0)
					// {
					// 	cout << "	[ ] ";
					// }
					// else
					// {
					// 	cout << "	[S] ";
					// }

					// cout << "(" << points[k][i][0] << "," << points[k][i][1] << ") ";
					// cout << "[" << points[k][i][2];
					// cout << " " << points[k][i][3];
					// cout << " " << points[k][i][4];
					// cout << " " << points[k][i][5] << "]";
					// cout << endl;

					// Output if not squelched
					if(points[k][i][7] == 0)
					{
						red = 255;
						green = 0;
						blue = 0;
				   	}
				   	else
				   	{
						red = 0;
						green = 0;
						blue = 255;
				   	}

				   	if(points[k][i][7] == 0 || draw_supressed)
				   	{
						// Draw on image
						CvPoint centroid[1];
						centroid[0].x = points[k][i][0];
						centroid[0].y = points[k][i][1];

						cvCircle(img, centroid[0], 3, CV_RGB(red, green, blue), -1, 0, 0);
						cvRectangle(img,	cvPoint(points[k][i][2], points[k][i][3]),
											cvPoint(points[k][i][4], points[k][i][5]),
											cvScalar(blue, green, red), 2, 8, 0);
				   }
				}
			}

			if(save_detection_images)
			{
				//cvSmooth(combined, combined, CV_GAUSSIAN, 3);
				sprintf_s(buffer,"%s.png", detection_result_filepath);
				cvSaveImage( buffer, combined );

				sprintf_s(buffer,"%s_points.png", detection_result_filepath);
				cvSaveImage( buffer, img );
			}

			// Release image
			cvReleaseImage(&img);
			cvReleaseImage(&combined);
			cvReleaseImage(&temp);

			return num_detections;
		}

		// Extract patches from training data
		void extract_Patches(	CRPatch& Train,
								CvRNG* pRNG,
								string training_inventory_pos,
								string training_inventory_neg
							)
		{

			vector<string> vFilenames;
			vector<CvRect> vBBox;
			vector<vector<CvPoint> > vCenter;

			//////////////////////

			// load positive file list
			loadTrainPosFile(vFilenames,  vBBox, vCenter, training_inventory_pos);

			// load postive images and extract patches
			for(int i=0; i<(int)vFilenames.size(); ++i) {

				cout << "0.";
			  	if(i%10==0) cout << i << " " << flush;
				// cout << i << " " << (int)vFilenames.size() << " " << vFilenames[i] << endl;

				cout << "1.";
				// Load image
				IplImage *img = 0;
				img = cvLoadImage(vFilenames[i].c_str(),CV_LOAD_IMAGE_COLOR);
				if(!img) {
					cout << "Could not load image file: " << vFilenames[i].c_str() << endl;
					exit(-1);
				}
				cout << ".1";

				// Extract positive training patches
				cout << "2.";
				Train.extractPatches(img, patch_sample_density_pos, 1, &vBBox[i], &vCenter[i], legacy);

				cout << "3.";
				if(include_horizontal_flip)
				{
					IplImage *img2 = 0;
					img2 = cvLoadImage(vFilenames[i].c_str(),CV_LOAD_IMAGE_COLOR);
					cvFlip(img2, img2, 1);

					Train.extractPatches(img2, patch_sample_density_pos, 1, &vBBox[i], &vCenter[i], legacy);

					cvReleaseImage(&img2);
				}

				// Release image
				cvReleaseImage(&img);


			}
			cout << endl;

			///////////////////

			// load negative file list
			loadTrainNegFile(vFilenames, vBBox, training_inventory_neg);

			// load negative images and extract patches

			for(int i=0; i<(int)vFilenames.size(); ++i) {

				if(i%10==0) cout << i << " " << flush;
				// cout << i << " " << (int)vFilenames.size() << " " << vFilenames[i] << endl;

				// Load image
				IplImage *img = 0;
				img = cvLoadImage(vFilenames[i].c_str(),CV_LOAD_IMAGE_COLOR);

				if(!img) {
					cout << "Could not load image file: " << vFilenames[i].c_str() << endl;
					exit(-1);
				}

				// Extract negative training patches
				if(vBBox.size()==vFilenames.size())
					Train.extractPatches(img, patch_sample_density_neg, 0, &vBBox[i], 0, legacy);
				else
					Train.extractPatches(img, patch_sample_density_neg, 0, 0, 0, legacy);

				if(include_horizontal_flip && false)
				{
					IplImage *img2 = 0;
					img2 = cvLoadImage(vFilenames[i].c_str(),CV_LOAD_IMAGE_COLOR);
					cvFlip(img2, img2, 1);

					if(vBBox.size()==vFilenames.size())
						Train.extractPatches(img2, patch_sample_density_neg, 0, &vBBox[i], 0, legacy);
					else
						Train.extractPatches(img2, patch_sample_density_neg, 0, 0, 0, legacy);

					cvReleaseImage(&img2);
				}


				// Release image
				cvReleaseImage(&img);


			}
			cout << endl;

		}

		CRForest* load_forest(char* tree_path, char* prefix, int num_trees)
		{
			// Init forest with number of trees
			CRForest* crForest = new CRForest( num_trees );
			// Load forest
			string str_tree_path = tree_path;
			string str_prefix = prefix;
			crForest->loadForest( (str_tree_path + "/" + str_prefix).c_str() );
			return crForest;
		}

		// Init and start detector
		int run_detect(CRForest* crForest,
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
			// Init detector
			CRForestDetector crDetect(crForest, patch_width, patch_height);

			// run detector
			return detect(	crDetect,
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

		void detect_results(float *results)
		{
			int size = 8;
			int index = 0;
			for(unsigned int k=0;k<points.size(); ++k) {
				for(unsigned int i=0;i<points[k].size(); ++i) {
					results[index * size + 0] = points[k][i][0];
					results[index * size + 1] = points[k][i][1];
					results[index * size + 2] = points[k][i][2];
					results[index * size + 3] = points[k][i][3];
					results[index * size + 4] = points[k][i][4];
					results[index * size + 5] = points[k][i][5];
					results[index * size + 6] = points[k][i][6];
					results[index * size + 7] = points[k][i][7];

					index++;
				}
			}
		}

		// Init and start training
		void run_train(	char* tree_path,
						int num_trees,
						char* training_inventory_pos,
						char* training_inventory_neg
						)
		{


			// Init forest with number of trees
			CRForest crForest( num_trees );

			// Init random generator
			time_t t = time(NULL);
			int seed = (int)t;

			CvRNG cvRNG(seed);

			// Create directory
			string tpath(tree_path);
			tpath.erase(tpath.find_last_of(PATH_SEP));
			string execstr = "mkdir ";
			execstr += tpath;
			int result = system( execstr.c_str() );

			// Init training data
			CRPatch Train(&cvRNG, patch_width, patch_height, 2);

			// Extract training patches
			extract_Patches(Train, &cvRNG,
							training_inventory_pos,
							training_inventory_neg);

			// Train forest
			crForest.trainForest(20, 15, &cvRNG, Train, 2000, default_split, tree_path, 0);

			// Save forest
			// crForest.saveForest(tree_path, 0);

		}
};
