#pragma once
#include "matrix.h"

struct Transform {
    Matrix A;
    vector<double> b;
    int dim_in;
    int dim_out;
    Transform(int dim_in, int dim_out) : A(dim_out, dim_in), b(dim_out){
        A.randomize();
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }
    static void ReLU(vector<double> &x) {
        for (int i = 0; i < x.size(); i++) {
            if (x[i]<0) x[i]=0;
        }
    }
    void transform(const vector<double>& x, vector<double>& y) {
        A.times(x, y);
        Matrix::add(y,b);
    }
    vector<double> transform(const vector<double>& x) {
        vector<double> y(dim_out);
        A.times(x, y);
        Matrix::add(y,b);
        return y;
    }
    void step(Matrix& A_delta, vector<double>& b_delta, double lr) {
        Matrix::add(A, A_delta, -lr);
        Matrix::add(b, b_delta, -lr);
    }
};
