#pragma once
#include "matrix.h"

struct AttentionScratch {
    int n;
    Matrix Q, K, V, P;
    Matrix dW_q, dW_k, dW_v;
    vector<vector<double>> x;

    AttentionScratch(int n, int dim_head, int dim_model) :
        n(n), Q(n, dim_head), K(n, dim_head), V(n, dim_head), P(n, n),
        dW_q(dim_head, dim_model), dW_k(dim_head, dim_model), dW_v(dim_head, dim_model),
        x(n) {}

    void clear() {
        Matrix::clear(dW_q);
        Matrix::clear(dW_k);
        Matrix::clear(dW_v);
    }
};
