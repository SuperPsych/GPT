#pragma once
#include <bits/stdc++.h>
#include <random>
#include <span>
using namespace std;

struct Matrix {
    vector<float> data;
    int num_rows;
    int num_cols;

    static void print(const vector<float> &x) {
        for (const float i : x) {
            cout << i << " ";
        }
        cout << endl;
    }

    Matrix(int r, int c) : data(r * c), num_rows(r), num_cols(c) {}

    inline float& at(int r, int c) {
        return data[r * num_cols + c];
    }
    inline float at(int r, int c) const {
        return data[r * num_cols + c];
    }

    span<float> add_row() {
        int offset = data.size();
        data.resize(data.size() + num_cols);
        num_rows++;
        return span<float>(data.data() + offset, num_cols);
    }

    span<float> row(int r) {
        return span<float>(data.data() + r * num_cols, num_cols);
    }

    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        float std_dev = (float)sqrt(2.0 / num_cols);
        normal_distribution<float> dist(0.0f, std_dev);
        for (float& value : data) {
            value = dist(gen);
        }
    }

    static void add(vector<float> &a, const vector<float> &b) {
        for (int i = 0; i < a.size(); i++) {
            a[i] += b[i];
        }
    }
    static void add(Matrix &a, const Matrix &b) {
        add(a.data, b.data);
    }
    static void add(vector<float> &a, const vector<float> &b, double mult) {
        for (int i = 0; i < a.size(); i++) {
            a[i] += b[i] * mult;
        }
    }
    static void add(Matrix &a, const Matrix &b, double mult) {
        add(a.data, b.data, mult);
    }
    static void clear(vector<float>& a) {
        fill(a.begin(), a.end(), 0.0f);
    }
    static void clear(Matrix& a) {
        clear(a.data);
    }

    // Moments (m, v) are kept as float too (this is a from-scratch fp32
    // training setup, not a mixed-precision one), but the hyperparameters
    // stay double since they're scalars evaluated once per weight per step,
    // nowhere near the hot per-token loops.
    static void adamw_update(vector<float>& W, const vector<float>& dW, vector<float>& m, vector<float>& v,
                              double lr, double beta1, double beta2, double eps, double weight_decay,
                              int t, double grad_scale) {
        double bc1 = 1.0 - pow(beta1, t);
        double bc2 = 1.0 - pow(beta2, t);
        for (size_t i = 0; i < W.size(); i++) {
            double g = dW[i] * grad_scale;
            double m_new = beta1 * m[i] + (1 - beta1) * g;
            double v_new = beta2 * v[i] + (1 - beta2) * g * g;
            m[i] = (float)m_new;
            v[i] = (float)v_new;
            double m_hat = m_new / bc1;
            double v_hat = v_new / bc2;
            W[i] -= (float)(lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * W[i]));
        }
    }
    static void adamw_update(Matrix& W, const Matrix& dW, Matrix& m, Matrix& v,
                              double lr, double beta1, double beta2, double eps, double weight_decay,
                              int t, double grad_scale) {
        adamw_update(W.data, dW.data, m.data, v.data, lr, beta1, beta2, eps, weight_decay, t, grad_scale);
    }
    void times(const vector<float>& x, span<float> y) const {
        for (int r = 0; r < num_rows; r++) {
            int base = r * num_cols;
            float sum = 0;
            for (int c = 0; c < num_cols; c++) {
                sum += data[base + c] * x[c];
            }
            y[r] = sum;
        }
    }

    vector<float> times(const vector<float>& x) const {
        vector<float> y(num_rows);
        times(x, y);
        return y;
    }

    // Batched projection: Y[0:n] = X[0:n] @ W^T, where `this` is W (dim_out x
    // dim_in, PyTorch-style), X is (n x dim_in), Y is (n x dim_out). Unlike a
    // per-token times() call, this computes 4 output rows at once per token
    // using 4 independent accumulators, which breaks the single-accumulator
    // FMA dependency chain a plain dot product has and lets the CPU actually
    // issue back-to-back FMAs instead of stalling on latency. X/Y only need
    // their first n rows valid; extra capacity rows are left untouched.
    void times(const Matrix& X, Matrix& Y, int n) const {
        int dim_in = num_cols;
        int dim_out = num_rows;
        for (int i = 0; i < n; i++) {
            const float* xi = &X.data[(size_t)i * dim_in];
            float* yi = &Y.data[(size_t)i * Y.num_cols];
            int r = 0;
            for (; r + 4 <= dim_out; r += 4) {
                const float* w0 = &data[(size_t)(r + 0) * dim_in];
                const float* w1 = &data[(size_t)(r + 1) * dim_in];
                const float* w2 = &data[(size_t)(r + 2) * dim_in];
                const float* w3 = &data[(size_t)(r + 3) * dim_in];
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (int c = 0; c < dim_in; c++) {
                    float xc = xi[c];
                    s0 += w0[c] * xc;
                    s1 += w1[c] * xc;
                    s2 += w2[c] * xc;
                    s3 += w3[c] * xc;
                }
                yi[r + 0] = s0; yi[r + 1] = s1; yi[r + 2] = s2; yi[r + 3] = s3;
            }
            for (; r < dim_out; r++) {
                const float* wr = &data[(size_t)r * dim_in];
                float s = 0;
                for (int c = 0; c < dim_in; c++) s += wr[c] * xi[c];
                yi[r] = s;
            }
        }
    }

    // Batched backprop-through-weights: dX[0:n] = dY[0:n] @ W, where `this`
    // is W (dim_out x dim_in), dY is (n x dim_out), dX is (n x dim_in). For
    // each token this is a sum of dim_out scaled copies of W's rows, which is
    // already contiguous/vectorizable in c without extra blocking.
    // accumulate=true adds onto dX's existing contents instead of overwriting
    // it, for callers (like the FFN) whose dx is the sum of two weights' input
    // gradients.
    void backward_input(const Matrix& dY, Matrix& dX, int n, bool accumulate = false) const {
        int dim_in = num_cols;
        int dim_out = num_rows;
        for (int i = 0; i < n; i++) {
            float* dxi = &dX.data[(size_t)i * dim_in];
            if (!accumulate) fill(dxi, dxi + dim_in, 0.0f);
            const float* dyi = &dY.data[(size_t)i * dY.num_cols];
            for (int r = 0; r < dim_out; r++) {
                float g = dyi[r];
                const float* wr = &data[(size_t)r * dim_in];
                for (int c = 0; c < dim_in; c++) dxi[c] += g * wr[c];
            }
        }
    }

    // Batched weight-gradient accumulation: dW[0:n rows contribute] += dY^T @ X,
    // i.e. dW(dim_out x dim_in) += sum_i outer(dY[i], X[i]). Same contiguous
    // accumulation pattern as backward_input, just accumulating into the
    // weight gradient instead of the input gradient.
    static void accumulate_weight_grad(const Matrix& dY, const Matrix& X, Matrix& dW, int n) {
        int dim_out = dW.num_rows;
        int dim_in = dW.num_cols;
        for (int i = 0; i < n; i++) {
            const float* dyi = &dY.data[(size_t)i * dY.num_cols];
            const float* xi = &X.data[(size_t)i * X.num_cols];
            for (int r = 0; r < dim_out; r++) {
                float g = dyi[r];
                float* wr = &dW.data[(size_t)r * dim_in];
                for (int c = 0; c < dim_in; c++) wr[c] += g * xi[c];
            }
        }
    }

    static float dot(const vector<float> &a, const vector<float> &b) {
        float res = 0;
        for (int i = 0; i < a.size(); i++) {
            res += a[i] * b[i];
        }
        return res;
    }

    void save(ostream& out) const {
        out << num_rows << " " << num_cols << "\n";
        out << scientific << setprecision(9);
        for (size_t i = 0; i < data.size(); i++) {
            out << data[i];
            if (i + 1 < data.size()) out << " ";
        }
        out << "\n";
    }

    void load(istream& in) {
        int r, c;
        in >> r >> c;
        num_rows = r;
        num_cols = c;
        data.resize((size_t)r * c);
        for (float& v : data) in >> v;
    }

    static void save_vec(ostream& out, const vector<float>& v) {
        out << v.size() << "\n";
        out << scientific << setprecision(9);
        for (size_t i = 0; i < v.size(); i++) {
            out << v[i];
            if (i + 1 < v.size()) out << " ";
        }
        out << "\n";
    }

    static void load_vec(istream& in, vector<float>& v) {
        size_t n;
        in >> n;
        v.resize(n);
        for (float& x : v) in >> x;
    }
};
