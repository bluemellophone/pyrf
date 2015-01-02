/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"

using namespace std;

class CRForestDetector {
public:
    // Constructor
    CRForestDetector(const CRForest *pRF) : crForest(pRF) {
        width  = crForest->getPatchHeight();
        height = crForest->getPatchHeight();
    }

    // detect multi scale
    void detectPyramid(IplImage *img, vector<IplImage * > &imgDetect,
                       vector<vector<vector<vector<CvPoint > > > > &vmenifests, int mode, bool serial);

    // Get/Set functions
    unsigned int GetNumCenter() const {
        return crForest->GetNumCenter();
    }

private:
    void detectColor(IplImage *img, IplImage *imgDetect,
                     vector<vector<vector<CvPoint > > > &manifest, int mode);

    const CRForest *crForest;
    int width;
    int height;
};
