/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#define _copysign copysign

#include <opencv/cxcore.h>
#include <opencv/cv.h>

#include <vector>
#include <iostream>

#include "HoG.h"

using namespace std;

// structure for image patch
struct PatchFeature {
    PatchFeature() {}

    CvRect roi;
    CvPoint center;

    vector<CvMat *> vPatch;
    void print() const {
        cout << "[pyrf c++] " << roi.x << " " << roi.y << " " << roi.width << " " << roi.height << " " << center.x << " " << center.y << endl;
    }
    void show(int delay) const;
};

static HoG hog;

class CRPatch {
public:
    CRPatch(CvRNG *pRNG, int w, int h, int num_l) : cvRNG(pRNG), width(w), height(h) {
        vLPatches.resize(num_l);
    }

    // Extract patches from image
    int extractPatches(IplImage *img, unsigned int n, int label);

    // Extract features from image
    static void extractFeatureChannels(IplImage *img, vector<IplImage *> &vImg);

    // min/max filter
    static void maxfilt(uchar *data, uchar *maxvalues, unsigned int step, unsigned int size, unsigned int width);
    static void maxfilt(uchar *data, unsigned int step, unsigned int size, unsigned int width);
    static void minfilt(uchar *data, uchar *minvalues, unsigned int step, unsigned int size, unsigned int width);
    static void minfilt(uchar *data, unsigned int step, unsigned int size, unsigned int width);
    static void maxminfilt(uchar *data, uchar *maxvalues, uchar *minvalues, unsigned int step, unsigned int size, unsigned int width);
    static void maxfilt(IplImage *src, unsigned int width);
    static void maxfilt(IplImage *src, IplImage *dst, unsigned int width);
    static void minfilt(IplImage *src, unsigned int width);
    static void minfilt(IplImage *src, IplImage *dst, unsigned int width);

    vector<vector<PatchFeature > > vLPatches;
private:
    CvRNG *cvRNG;
    int width;
    int height;
};

