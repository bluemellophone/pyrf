/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void CRForestDetector::detectColor(IplImage *img, IplImage *imgDetect,
                                   vector<vector<vector<CvPoint > > > &manifest,
                                   int mode, float multiplier, bool serial)
{
    // extract features
    vector<IplImage * > vImg;
    CRPatch::extractFeatureChannels(img, vImg);

    // reset output image
    cvSetZero( imgDetect );

    // get pointers to feature channels
    int stepImg;
    uchar **ptFCh     = new uchar*[vImg.size()];
    uchar **ptFCh_row = new uchar*[vImg.size()];
    for (unsigned int c = 0; c < vImg.size(); ++c) {
        cvGetRawData( vImg[c], (uchar **) & (ptFCh[c]), &stepImg);
    }
    stepImg /= sizeof(ptFCh[0][0]);

    // get pointer to output image
    int stepDet;
    float *ptDet;
    cvGetRawData( imgDetect, (uchar **) & (ptDet), &stepDet);
    stepDet /= sizeof(ptDet[0]);

    int x, y, cx, cy; // nx, ny; // x, y top left; cx, cy center of patch
    int z;
    float weight;
    int xoffset = width / 2;
    int yoffset = height / 2;

    cy = yoffset;
    // int global_counter = 0;
    for (y = 0; y < img->height - height; ++y, ++cy)
    {
        // Get start of row
        for (unsigned int c = 0; c < vImg.size(); ++c)
        {
            ptFCh_row[c] = &ptFCh[c][0];
        }
        cx = xoffset;

        for (x = 0; x < img->width - width; ++x, ++cx)
        {
            // regression for a single patch
            vector<const LeafNode *> result;
            crForest->regression(result, ptFCh_row, stepImg);

            // vote for all trees (leafs)
            for (vector<const LeafNode *>::const_iterator itL = result.begin(); itL != result.end(); ++itL)
            {
                // To speed up the voting, one can vote only for patches
                // with a probability for foreground > 0.5
                if (mode == 0 && (*itL)->pfg > 0.5) // Mode: Hough Voting (default)
                {
                    weight = (*itL)->pfg / (result.size() * (*itL)->vCenter.size());
                    // Apply the scale multiplier to the vote
                    weight *= multiplier;
                    // Vote for all center offsets stored in the leaf
                    #pragma omp parallel for if(serial)
                    for (z = 0; z < (*itL)->vCenter.size(); ++z)
                    {   
                        CvPoint it = (*itL)->vCenter[z];
                        int nx = cx - it.x;
                        int ny = cy - it.y;
                        // TODO: Let votes vote outside of the image
                        if (ny >= 0 && ny < imgDetect->height && nx >= 0 && nx < imgDetect->width)
                        {
                            // Normalize the weight by the number of leaves in the result and by number of centers voting
                            *(ptDet + nx + ny * stepDet) += weight;
                            // If perfect confidence, add point to the manifest
                            if ((*itL)->pfg == 1.00)
                            {
                                #pragma omp critical(voting)
                                {
                                    manifest[ny][nx].push_back(cvPoint(cx, cy));
                                    // Give perfect patches an additional vote as a reward
                                    *(ptDet + nx + ny * stepDet) += weight;
                                    // global_counter++;
                                }
                            }
                        }
                    }
                }
                else if (mode == 1) // Mode: Classification / Weight Map
                {
                    // Normalize the weight by the number of trees
                    weight = (*itL)->pfg / result.size();
                    *(ptDet + cx + cy * stepDet) += weight;
                }
            }
            // increase pointer - x
            for (unsigned int c = 0; c < vImg.size(); ++c)
            {
                ++ptFCh_row[c];
            }
        }
        // increase pointer - y
        for (unsigned int c = 0; c < vImg.size(); ++c)
        {
            ptFCh[c] += stepImg;
        }
    }
    // cout << "MANIFEST SIZE (MB): " << global_counter << " " << sizeof(CvPoint*) << "  " << (global_counter * sizeof(CvPoint*)) / (1024.0 * 1024.0) << endl; 
    
    // release feature channels
    for (unsigned int c = 0; c < vImg.size(); ++c)
    {
        cvReleaseImage(&vImg[c]);
    }
    delete[] ptFCh;
    delete[] ptFCh_row;
}

void CRForestDetector::detectPyramid(IplImage *img, vector<IplImage * > &vImgDetect,
                                     vector<vector<vector<vector<CvPoint > > > > &vManifests,
                                     vector<float> &scale_vector, int mode, bool serial)
{

    if (img->nChannels == 1)
    {
        cerr << "Gray color images are not supported." << endl;
    }
    else
    {
        #pragma omp parallel for if(serial)
        for (int i = 0; i < int(vImgDetect.size()); ++i)
        {
            IplImage *scale = cvCreateImage( cvSize(vImgDetect[i]->width, vImgDetect[i]->height) , IPL_DEPTH_8U , 3);
            cvResize( img, scale, CV_INTER_LANCZOS4 );
            // cvResize( img, scale, CV_INTER_LINEAR );
            // detection
            float multiplier = sqrt(1.0 / scale_vector[i]);
            detectColor(scale, vImgDetect[i], vManifests[i], mode, multiplier, serial);
            // release
            cvReleaseImage(&scale);
        }
    }

}
