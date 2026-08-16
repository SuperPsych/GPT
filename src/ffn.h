#pragma once
#include "matrix.h"
#include "ffn_scratch.h"

struct FFN {
    Matrix W_gate, W_up, W_down;
    int dim_model, hidden_dim;

    FFN(int dim_model, int hidden_dim) :
        W_gate(hidden_dim, dim_model), W_up(hidden_dim, dim_model), W_down(dim_model, hidden_dim),
        dim_model(dim_model), hidden_dim(hidden_dim) {
        W_gate.randomize();
        W_up.randomize();
        W_down.randomize();
    }

    static void SiLU(vector<double> &x) {
        for (double& v : x) {
            v = v / (1.0 + exp(-v));
        }
    }
    static double silu_grad(double z) {
        double s = 1.0 / (1.0 + exp(-z));
        return s * (1.0 + z * (1.0 - s));
    }
    static void output_activation(vector<double> &x) {
        double max = *max_element(x.begin(), x.end());
        double sum = 0;
        for(double& i : x){
            i = exp(i - max);
            sum += i;
        }
        for(double& i : x){
            i/=sum;
        }
    }

    vector<double> forward(const vector<double>& x) const {
        vector<double> a_gate = W_gate.times(x);
        SiLU(a_gate);
        vector<double> z_up = W_up.times(x);
        vector<double> h(hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            h[i] = a_gate[i] * z_up[i];
        }
        return W_down.times(h);
    }

    vector<double> forward_train(const vector<double>& x, FFNScratch& td) {
        td.x = x;
        W_gate.times(x, td.z_gate);
        td.a_gate = td.z_gate;
        SiLU(td.a_gate);
        W_up.times(x, td.z_up);
        for (int i = 0; i < hidden_dim; i++) {
            td.h[i] = td.a_gate[i] * td.z_up[i];
        }
        return W_down.times(td.h);
    }

    vector<double> backward(const vector<double>& grad_out, FFNScratch& td) {
        for (int r = 0; r < W_down.num_rows; r++) {
            for (int c = 0; c < W_down.num_cols; c++) {
                td.W_down_delta.at(r,c) += grad_out[r] * td.h[c];
            }
        }
        vector<double> dh(hidden_dim, 0.0);
        for (int r = 0; r < W_down.num_rows; r++) {
            for (int c = 0; c < W_down.num_cols; c++) {
                dh[c] += W_down.at(r,c) * grad_out[r];
            }
        }

        vector<double> dz_gate(hidden_dim), dz_up(hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            double da_gate = dh[i] * td.z_up[i];
            dz_up[i] = dh[i] * td.a_gate[i];
            dz_gate[i] = da_gate * silu_grad(td.z_gate[i]);
        }

        for (int r = 0; r < W_gate.num_rows; r++) {
            for (int c = 0; c < W_gate.num_cols; c++) {
                td.W_gate_delta.at(r,c) += dz_gate[r] * td.x[c];
            }
        }
        for (int r = 0; r < W_up.num_rows; r++) {
            for (int c = 0; c < W_up.num_cols; c++) {
                td.W_up_delta.at(r,c) += dz_up[r] * td.x[c];
            }
        }

        vector<double> dx(dim_model, 0.0);
        for (int r = 0; r < W_gate.num_rows; r++) {
            for (int c = 0; c < W_gate.num_cols; c++) {
                dx[c] += W_gate.at(r,c) * dz_gate[r];
            }
        }
        for (int r = 0; r < W_up.num_rows; r++) {
            for (int c = 0; c < W_up.num_cols; c++) {
                dx[c] += W_up.at(r,c) * dz_up[r];
            }
        }
        return dx;
    }
};
