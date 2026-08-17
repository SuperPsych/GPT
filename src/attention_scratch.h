#pragma once
#include "matrix.h"

struct AttentionScratch {
    int n;
    Matrix Q, K, V, P;
    Matrix dW_q, dW_k, dW_v;
    Matrix x;

    Matrix out;
    Matrix dQ, dK, dV, dx_out;
    Matrix grad_out_buf;
    vector<float> scores, dP;

    AttentionScratch(int n, int dim_head, int dim_model) :
        n(n), Q(n, dim_head), K(n, dim_head), V(n, dim_head), P(n, n),
        dW_q(dim_head, dim_model), dW_k(dim_head, dim_model), dW_v(dim_head, dim_model),
        x(n, dim_model),
        out(n, dim_head), dQ(n, dim_head), dK(n, dim_head), dV(n, dim_head), dx_out(n, dim_model),
        grad_out_buf(n, dim_head), scores(n), dP(n) {}

    void clear() {
        Matrix::clear(dW_q);
        Matrix::clear(dW_k);
        Matrix::clear(dW_v);
    }
};
