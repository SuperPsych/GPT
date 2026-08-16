#pragma once
#include <bits/stdc++.h>
using namespace std;

struct RMSNorm {
    vector<double> g;
    static constexpr double eps = 1e-6;

    RMSNorm(int dim_model) : g(dim_model, 1.0) {}

    vector<double> forward(const vector<double>& x) const {
        double ms = 0;
        for (double v : x) ms += v * v;
        ms /= x.size();
        double scale = 1.0 / sqrt(ms + eps);
        vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = x[i] * scale * g[i];
        }
        return out;
    }

    vector<double> backward(const vector<double>& x, const vector<double>& dOut, vector<double>& dg) const {
        int n = x.size();
        double ms = 0;
        for (double v : x) ms += v * v;
        ms /= n;
        double s = 1.0 / sqrt(ms + eps);
        double s3 = s * s * s;

        double weighted = 0;
        for (int i = 0; i < n; i++) {
            dg[i] += dOut[i] * x[i] * s;
            weighted += dOut[i] * g[i] * x[i];
        }
        vector<double> dx(n);
        for (int i = 0; i < n; i++) {
            dx[i] = s * g[i] * dOut[i] - (s3 / n) * x[i] * weighted;
        }
        return dx;
    }
};
