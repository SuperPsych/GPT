#pragma once
#include "matrix.h"

struct FFNScratch {
    Matrix W_gate_delta, W_up_delta, W_down_delta;
    vector<double> x, z_gate, a_gate, z_up, h;

    FFNScratch(int dim_model, int hidden_dim) :
        W_gate_delta(hidden_dim, dim_model), W_up_delta(hidden_dim, dim_model), W_down_delta(dim_model, hidden_dim),
        x(dim_model), z_gate(hidden_dim), a_gate(hidden_dim), z_up(hidden_dim), h(hidden_dim) {}

    void clear(){
        Matrix::clear(W_gate_delta);
        Matrix::clear(W_up_delta);
        Matrix::clear(W_down_delta);
    }
};
