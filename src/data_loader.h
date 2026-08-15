#pragma once
#include <charconv>
#include "sample.h"

vector<Sample> load_samples(const string& filename, int input_dim, int output_dim){
    cout << "Loading in data..." << endl;
    ifstream file(filename, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> buf(size);
    file.read(buf.data(), size);

    vector<Sample> samples;
    samples.reserve(count(buf.begin(), buf.end(), '\n') + 1);

    const char* p = buf.data();
    const char* end = p + size;
    auto skip_space = [&](){ while(p < end && isspace((unsigned char)*p)) p++; };

    while(true){
        skip_space();
        if(p >= end) return samples;

        vector<double> x(input_dim);
        vector<double> y(output_dim);
        for(int i=0; i<input_dim; i++){
            auto res = from_chars(p, end, x[i]);
            if(res.ec != errc()) return samples;
            p = res.ptr;
            skip_space();
        }
        for(int i=0; i<output_dim; i++){
            auto res = from_chars(p, end, y[i]);
            if(res.ec != errc()) return samples;
            p = res.ptr;
            skip_space();
        }
        samples.emplace_back(x, y);
    }
}
