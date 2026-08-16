#pragma once
#include <cmath>
using namespace std;

struct RoPE {
    template <typename Vec>
    static void rotate(Vec& v, int pos, double sign = 1.0) {
        int dim = (int)v.size();
        for (int i = 0; i + 1 < dim; i += 2) {
            double theta = pos / pow(10000.0, (double)i / dim);
            double c = cos(theta);
            double s = sin(theta) * sign;
            double a = v[i], b = v[i + 1];
            v[i]     = a * c - b * s;
            v[i + 1] = a * s + b * c;
        }
    }
};
