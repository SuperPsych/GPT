#pragma once
#include <cmath>
#include <vector>
#include <unordered_map>
#include <utility>
using namespace std;

struct RoPE {
    using Angles = vector<pair<float, float>>; // (cos, sin) per frequency pair, indexed by position

    static unordered_map<int, vector<Angles>>& tables() {
        static unordered_map<int, vector<Angles>> t;
        return t;
    }

    static Angles compute(int dim, int pos) {
        int half = dim / 2;
        Angles angles(half);
        for (int k = 0; k < half; k++) {
            double theta = pos / pow(10000.0, (double)(2 * k) / dim);
            angles[k] = { (float)cos(theta), (float)sin(theta) };
        }
        return angles;
    }

    // Precomputes cos/sin for positions [0, max_pos) for a given head dimension so that
    // rotate() becomes a pure table lookup instead of recomputing pow/cos/sin every call.
    // Must be called once, single-threaded, before any concurrent (OpenMP) use, since it
    // mutates the shared table; rotate() itself only reads it once populated.
    static void precompute(int dim, int max_pos) {
        auto& table = tables()[dim];
        if ((int)table.size() < max_pos) table.resize(max_pos);
        for (int pos = 0; pos < max_pos; pos++) {
            if (table[pos].empty()) table[pos] = compute(dim, pos);
        }
    }

    template <typename Vec>
    static void rotate(Vec& v, int pos, double sign = 1.0) {
        int dim = (int)v.size();
        const Angles* angles = nullptr;

        auto it = tables().find(dim);
        if (it != tables().end() && pos < (int)it->second.size() && !it->second[pos].empty()) {
            angles = &it->second[pos];
        }

        Angles fallback;
        if (!angles) {
            // Position beyond the precomputed range (e.g. generation past max_seq_len).
            // Computed into a local so this stays safe even if called concurrently.
            fallback = compute(dim, pos);
            angles = &fallback;
        }

        int half = dim / 2;
        for (int k = 0; k < half; k++) {
            float c = (*angles)[k].first;
            float s = (*angles)[k].second * (float)sign;
            int i = 2 * k;
            float a = v[i], b = v[i + 1];
            v[i]     = a * c - b * s;
            v[i + 1] = a * s + b * c;
        }
    }
};
