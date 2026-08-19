#pragma once
#include "matrix.h"
#include "ffn_scratch.h"

struct FFN {
    Matrix W_gate, W_up, W_down;
    int dim_model, hidden_dim;

    FFN(int dim_model, int hidden_dim, int num_layers) :
        W_gate(hidden_dim, dim_model), W_up(hidden_dim, dim_model), W_down(dim_model, hidden_dim),
        dim_model(dim_model), hidden_dim(hidden_dim) {
        W_gate.randomize_xavier();
        W_up.randomize_xavier();
        float down_std = (float)(sqrt(1.0 / W_down.num_cols) / sqrt(2.0 * num_layers));
        W_down.randomize(down_std);
    }

    static void SiLU(vector<float> &x) {
        for (float& v : x) {
            v = v / (1.0f + exp(-v));
        }
    }
    static void SiLU(float* data, int count) {
        for (int i = 0; i < count; i++) {
            float v = data[i];
            data[i] = v / (1.0f + exp(-v));
        }
    }
    static float silu_grad(float z) {
        float s = 1.0f / (1.0f + exp(-z));
        return s * (1.0f + z * (1.0f - s));
    }
    template <typename Vec>
    static void output_activation(Vec& x) {
        double max = *max_element(x.begin(), x.end());
        double sum = 0;
        for(auto& i : x){
            i = exp(i - max);
            sum += i;
        }
        for(auto& i : x){
            i/=sum;
        }
    }

    vector<float> forward(const vector<float>& x) const {
        vector<float> a_gate = W_gate.times(x);
        SiLU(a_gate);
        vector<float> z_up = W_up.times(x);
        vector<float> h(hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            h[i] = a_gate[i] * z_up[i];
        }
        return W_down.times(h);
    }

    Matrix& forward_train(const Matrix& X, int n, FFNActivations& cache) {
        copy(X.data.begin(), X.data.begin() + (size_t)n * dim_model, cache.x.data.begin());
        W_gate.times(X, cache.z_gate, n);
        copy(cache.z_gate.data.begin(), cache.z_gate.data.begin() + (size_t)n * hidden_dim, cache.a_gate.data.begin());
        SiLU(cache.a_gate.data.data(), n * hidden_dim);
        W_up.times(X, cache.z_up, n);
        for (int idx = 0; idx < n * hidden_dim; idx++) {
            cache.h.data[idx] = cache.a_gate.data[idx] * cache.z_up.data[idx];
        }
        W_down.times(cache.h, cache.out, n);
        return cache.out;
    }

    Matrix& backward(const Matrix& grad_out, int n, const FFNActivations& cache, FFNDelta& delta) {
        Matrix::accumulate_weight_grad(grad_out, cache.h, delta.W_down_delta, n);
        W_down.backward_input(grad_out, delta.dh, n);

        for (int idx = 0; idx < n * hidden_dim; idx++) {
            float da_gate = delta.dh.data[idx] * cache.z_up.data[idx];
            delta.dz_up.data[idx] = delta.dh.data[idx] * cache.a_gate.data[idx];
            delta.dz_gate.data[idx] = da_gate * silu_grad(cache.z_gate.data[idx]);
        }

        Matrix::accumulate_weight_grad(delta.dz_gate, cache.x, delta.W_gate_delta, n);
        Matrix::accumulate_weight_grad(delta.dz_up, cache.x, delta.W_up_delta, n);

        W_gate.backward_input(delta.dz_gate, delta.dx, n, false);
        W_up.backward_input(delta.dz_up, delta.dx, n, true);
        return delta.dx;
    }
};
