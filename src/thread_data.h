#pragma once
#include "matrix.h"

struct ThreadData {
    vector<Matrix> A_deltas;
    vector<vector<double>> b_deltas;
    vector<vector<double>> activations;

    ThreadData(const vector<int>& dims) {
        A_deltas.reserve(dims.size()-1);
        b_deltas.reserve(dims.size()-1);
        activations.reserve(dims.size());
        for (int i=0; i<dims.size()-1; i++) {
            A_deltas.emplace_back(dims[i+1], dims[i]);
            b_deltas.emplace_back(dims[i+1]);
            activations.emplace_back(dims[i]);
        }
        activations.emplace_back(dims.back());
    }

    void clear(){
        for(auto& A_delta : A_deltas){
            Matrix::clear(A_delta);
        }
        for(auto& b_delta : b_deltas){
            Matrix::clear(b_delta);
        }
        for(auto& layer : activations){
            Matrix::clear(layer);
        }
    }

};
