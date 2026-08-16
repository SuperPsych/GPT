#pragma once
#include "matrix.h"

// Whole-sequence activations for one layer's FFN. One of these is kept per
// layer (not per token, as it used to be): batching the projections into a
// single GEMM per layer needs the input/output laid out contiguously across
// the whole sequence, which vector<vector<float>>-per-token can't give you.
struct FFNActivations {
    Matrix x, z_gate, a_gate, z_up, h, out;

    FFNActivations(int max_seq_len, int dim_model, int hidden_dim) :
        x(max_seq_len, dim_model), z_gate(max_seq_len, hidden_dim), a_gate(max_seq_len, hidden_dim),
        z_up(max_seq_len, hidden_dim), h(max_seq_len, hidden_dim), out(max_seq_len, dim_model) {}
};

// Gradient accumulator for one layer's FFN::backward() call (whole sequence
// at once). Cleared once per layer, not once per token.
struct FFNDelta {
    Matrix W_gate_delta, W_up_delta, W_down_delta;
    Matrix dh, dz_gate, dz_up, dx;

    FFNDelta(int max_seq_len, int dim_model, int hidden_dim) :
        W_gate_delta(hidden_dim, dim_model), W_up_delta(hidden_dim, dim_model), W_down_delta(dim_model, hidden_dim),
        dh(max_seq_len, hidden_dim), dz_gate(max_seq_len, hidden_dim), dz_up(max_seq_len, hidden_dim), dx(max_seq_len, dim_model) {}

    void clear() {
        Matrix::clear(W_gate_delta);
        Matrix::clear(W_up_delta);
        Matrix::clear(W_down_delta);
    }
};
