/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRTree.h"

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

class CRForest {
public:
    // Constructors
    CRForest(int trees = 0) {
        vTrees.resize(trees);
        patch_width  = -1;
        patch_height = -1;
    }
    ~CRForest() {
        for (vector<CRTree *>::iterator it = vTrees.begin(); it != vTrees.end(); ++it) delete *it;
        vTrees.clear();
    }

    // Set/Get functions
    void SetTrees(int n) {
        vTrees.resize(n);
    }
    int GetSize() const {
        return vTrees.size();
    }
    unsigned int GetDepth() const {
        return vTrees[0]->GetDepth();
    }
    unsigned int GetNumCenter() const {
        return vTrees[0]->GetNumCenter();
    }
    int getPatchWidth() const {
        return patch_width;
    }
    int getPatchHeight() const {
        return patch_height;
    }

    // Setters
    void verifyPatchWidth(int val, int tree_num) {
        if(patch_width < 0)
        {
            patch_width = val;   
        }
        else if(patch_width != val)
        {
            cout << "[pyrf c++] \tTree " << tree_num << " patch width mismatch and incompatible" << endl;
            exit(1);
        }
    }
    void verifyPatchHeight(int val, int tree_num) {
        if(patch_height < 0)
        {
            patch_height = val;   
        }
        else if(patch_height != val)
        {
            cout << "[pyrf c++] \tTree " << tree_num << " patch height mismatch and incompatible" << endl;
            exit(1);
        }
    }

    // Regression
    void regression(vector<const LeafNode *> &result, uchar **ptFCh, int stepImg) const;

    // Training
    void trainForest(int min_s, int max_d, CvRNG *pRNG, const CRPatch &TrData, int samples, float split, const char *filename, unsigned int offset, int patch_width, int patch_height, bool serial, bool verbose);

    // IO functions
    void saveForest(const char *filename, unsigned int offset = 0);
    void loadForest(const char *filename);
    void loadForest(vector<string> &tree_path_vector, bool serial, bool verbose);
    void show(int w, int h) const {
        vTrees[0]->showLeaves(w, h);
    }

    // Trees
    vector<CRTree *> vTrees;
private:

    int patch_width;
    int patch_height;
};

inline void CRForest::regression(vector<const LeafNode *> &result, uchar **ptFCh, int stepImg) const {
    result.resize( vTrees.size() );

    #pragma omp parallel for
    for (int i = 0; i < vTrees.size(); ++i)
    {
        result[i] = vTrees[i]->regression(ptFCh, stepImg);
    }
}

//Training
inline void CRForest::trainForest(int min_s, int max_d, CvRNG *pRNG, const CRPatch &TrData, int samples, float split, const char *filename, unsigned int offset, int patch_width, int patch_height, bool serial, bool verbose) {
    char buffer[200];

    max_d -= 1; // Decrease the max_d (max depth) by one to accomodate the interpretation of root is 0
    #pragma omp parallel for if(!serial)
    for (int i = 0; i < vTrees.size(); ++i)
    {
        vTrees[i] = new CRTree(min_s, max_d, patch_width, patch_height, pRNG);
        vTrees[i]->growTree(TrData, samples, split, verbose);
        // Save as trees are grown
        sprintf(buffer, "%s/tree-%03d.txt", filename, i + offset);
        vTrees[i]->saveTree(buffer);
    }
}

// IO Functions
inline void CRForest::saveForest(const char *filename, unsigned int offset) {
    char buffer[200];

    for (unsigned int i = 0; i < vTrees.size(); ++i) 
    {
        sprintf(buffer, "%s/tree-%03d.txt", filename, i + offset);
        vTrees[i]->saveTree(buffer);
    }
}

inline void CRForest::loadForest(const char *filename) {
    char buffer[200];

    for (unsigned int i = 0; i < vTrees.size(); ++i)
    {
        sprintf(buffer, "%s/tree-%03d.txt", filename, i);
        vTrees[i] = new CRTree(buffer);
        verifyPatchWidth(vTrees[i]->getPatchWidth(), i);
        verifyPatchHeight(vTrees[i]->getPatchHeight(), i);
    }
}

inline void CRForest::loadForest(vector<string> &tree_path_vector, bool serial, bool verbose) {
    #pragma omp parallel for if(!serial)
    for (unsigned int i = 0; i < vTrees.size(); ++i)
    {
        if(verbose)
        {
            #pragma omp critical(treeLoadInit)
            {
                cout << "[pyrf c++] Loading tree: " <<  tree_path_vector[i] << endl;
            }
        }
        // Load tree
        vTrees[i] = new CRTree(tree_path_vector[i].c_str());
        // Consitency check
        #pragma omp critical(treeLoadInit)
        {
            verifyPatchWidth(vTrees[i]->getPatchWidth(), i);
            verifyPatchHeight(vTrees[i]->getPatchHeight(), i);
        }
    }
}
