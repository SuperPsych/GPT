#pragma once
#include <bits/stdc++.h>
using namespace std;

struct RMSNorm {
    vector<float> g;
    static constexpr double eps = 1e-6;

    RMSNorm(int dim_model) : g(dim_model, 1.0f) {}

    template <typename VecX, typename VecOut>
    void forward(const VecX& x, VecOut& out) const {
        double ms = 0;
        for (float v : x) ms += (double)v * v;
        ms /= x.size();
        double scale = 1.0 / sqrt(ms + eps);
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = (float)(x[i] * scale * g[i]);
        }
    }

    vector<float> forward(const vector<float>& x) const {
        vector<float> out(x.size());
        forward(x, out);
        return out;
    }

    template <typename VecX, typename VecDOut, typename VecDx>
    void backward(const VecX& x, const VecDOut& dOut, vector<float>& dg, VecDx& dx) const {
        int n = x.size();
        double ms = 0;
        for (float v : x) ms += (double)v * v;
        ms /= n;
        double s = 1.0 / sqrt(ms + eps);
        double s3 = s * s * s;

        double weighted = 0;
        for (int i = 0; i < n; i++) {
            dg[i] += (float)(dOut[i] * x[i] * s);
            weighted += (double)dOut[i] * g[i] * x[i];
        }
        for (int i = 0; i < n; i++) {
            dx[i] = (float)(s * g[i] * dOut[i] - (s3 / n) * x[i] * weighted);
        }
    }

    vector<float> backward(const vector<float>& x, const vector<float>& dOut, vector<float>& dg) const {
        vector<float> dx(x.size());
        backward(x, dOut, dg, dx);
        return dx;
    }
};
