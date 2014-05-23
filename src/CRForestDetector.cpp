/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void CRForestDetector::detectColor(IplImage *img, vector<IplImage* >& imgDetect, std::vector<float>& ratios, int positive_like, bool legacy) {

	// extract features
	vector<IplImage*> vImg;
	CRPatch::extractFeatureChannels(img, vImg, legacy);

	// reset output image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSetZero( imgDetect[c] );

	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh     = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for(unsigned int c=0; c<vImg.size(); ++c) {
		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	// get pointer to output image
	int stepDet;
	float** ptDet = new float*[imgDetect.size()];
	for(unsigned int c=0; c<imgDetect.size(); ++c)
		cvGetRawData( imgDetect[c], (uchar**)&(ptDet[c]), &stepDet);
	stepDet /= sizeof(ptDet[0][0]);

	int xoffset = width/2;
	int yoffset = height/2;

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;

	for(y=0; y<img->height-height; ++y, ++cy) {

		// Get start of row
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset;

		for(x=0; x<img->width-width; ++x, ++cx) {

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row, stepImg);

			// vote for all trees (leafs)
			for(vector<const LeafNode*>::const_iterator itL = result.begin(); itL!=result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches
			        // with a probability for foreground > 0.5
			        //
                 if((*itL)->pfg>0.5) {

					// voting weight for leaf

					if(positive_like == 0)
					{
						// Normal Hough Voting
						float w = (*itL)->pfg / float( (*itL)->vCenter.size() * result.size() );

						// vote for all points stored in the leaf
						for(vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it) {

							for(int c=0; c<(int)imgDetect.size(); ++c) {
							  int x = int(cx - (*it)[0].x * ratios[c] + 0.5);
							  int y = cy-(*it)[0].y;
							  if(y>=0 && y<imgDetect[c]->height && x>=0 && x<imgDetect[c]->width) {
							    *(ptDet[c]+x+y*stepDet) += w;
							  }
							}
						}
					}
					else if(positive_like == 1)
					{
						// Classification Map
						float w = (*itL)->pfg / float(result.size());

						for(int c=0; c<(int)imgDetect.size(); ++c) {
						  if(y>=0 && y<imgDetect[c]->height && x>=0 && x<imgDetect[c]->width) {
						    *(ptDet[c]+cx+cy*stepDet) += w;
						  }
						}
					}
					else
					{
						// Regression Map

						float xA = 0.0;
						float yA = 0.0;
						float count = 0.0;
						for(vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it)
						{
							xA += (*it)[0].x;
							yA += (*it)[0].y;
							count++;
						}

						xA /= count;
						yA /= count;

						float D = 0.0;

						for(vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it)
						{
							D += sqrt((((*it)[0].x - xA) * ((*it)[0].x - xA)) + (((*it)[0].y - yA) * ((*it)[0].y - yA)));
						}

						// normalize D
						if(count != 0.0)
						{
							D /= count;
						}

						D /= 57.27; // Hard coded distance from furthest patch to center of training image

						// scale D
						D /= float(result.size());

						for(int c=0; c<(int)imgDetect.size(); ++c) {
						  if(y>=0 && y<imgDetect[c]->height && x>=0 && x<imgDetect[c]->width) {
						    *(ptDet[c]+cx+cy*stepDet) += D;
						  }
						}


					}

                  } // end if

			}

			// increase pointer - x
			for(unsigned int c=0; c<vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y

	if(positive_like == 0)
	{
		// smooth result image
		for(int c=0; c<(int)imgDetect.size(); ++c)
			cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);
	}

	// release feature channels
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);

	delete[] ptFCh;
	delete[] ptFCh_row;
	delete[] ptDet;

}

void CRForestDetector::detectPyramid(IplImage *img, vector<vector<IplImage*> >& vImgDetect, std::vector<float>& ratios, int positive_like, bool legacy) {

	if(img->nChannels==1) {

		std::cerr << "Gray color images are not supported." << std::endl;

	} else { // color

		#pragma omp parallel for
		for(int i=0; i<int(vImgDetect.size()); ++i)
		{
			// cout << "   Start " << i << endl;
			IplImage* cLevel = cvCreateImage( cvSize(vImgDetect[i][0]->width,vImgDetect[i][0]->height) , IPL_DEPTH_8U , 3);
			cvResize( img, cLevel, CV_INTER_LINEAR );

			// detection
			detectColor(cLevel,vImgDetect[i],ratios, positive_like, legacy);

			cvReleaseImage(&cLevel);
			// cout << "   End " << i << endl;
		}
	}

}








