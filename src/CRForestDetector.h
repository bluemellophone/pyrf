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
	CRForestDetector(const CRForest* pRF, int w, int h) : crForest(pRF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(
		IplImage *img, 
		vector<vector<IplImage*> >& imgDetect, 
		vector<vector<vector<vector<vector<vector<int> > > > > >& vmenifests,
		vector<float>& ratios, 
		int positive_like, 
		bool legacy = true
	);

	// Get/Set functions
	unsigned int GetNumCenter() const {return crForest->GetNumCenter();}

private:
	void detectColor(
		IplImage *img, 
		vector<IplImage*>& imgDetect, 
		vector<vector<vector<vector<vector<int> > > > >& manifests,
		vector<float>& ratios, 
		int positive_like, 
		bool legacy
	);

	const CRForest* crForest;
	int width;
	int height;
};
