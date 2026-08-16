#pragma once
#include <bits/stdc++.h>
#include <random>
using namespace std;

template <typename T>
struct Span {
    T* ptr;
    size_t len;
    Span(T* p, size_t n) : ptr(p), len(n) {}
    Span(vector<T>& v) : ptr(v.data()), len(v.size()) {}
    T& operator[](size_t i) const { return ptr[i]; }
    size_t size() const { return len; }
    T* data() const { return ptr; }
};

struct Matrix {
    vector<double> data;
    int num_rows;
    int num_cols;

    static void print(const vector<double> &x) {
        for (const double i : x) {
            cout << i << " ";
        }
        cout << endl;
    }

    Matrix(int r, int c) : data(r * c), num_rows(r), num_cols(c) {}

    inline double& at(int r, int c) {
        return data[r * num_cols + c];
    }
    inline double at(int r, int c) const {
        return data[r * num_cols + c];
    }

    Span<double> add_row() {
        int offset = data.size();
        data.resize(data.size() + num_cols);
        num_rows++;
        return Span<double>(data.data() + offset, num_cols);
    }

    Span<double> row(int r) {
        return Span<double>(data.data() + r * num_cols, num_cols);
    }

    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        double std_dev = sqrt(2.0 / num_cols);
        normal_distribution<double> dist(0.0, std_dev);
        for (double& value : data) {
            value = dist(gen);
        }
    }

    static void add(vector<double> &a, const vector<double> &b) {
        for (int i = 0; i < a.size(); i++) {
            a[i] += b[i];
        }
    }
    static void add(Matrix &a, const Matrix &b) {
        add(a.data, b.data);
    }
    static void add(vector<double> &a, const vector<double> &b, double mult) {
        for (int i = 0; i < a.size(); i++) {
            a[i] += b[i] * mult;
        }
    }
    static void add(Matrix &a, const Matrix &b, double mult) {
        add(a.data, b.data, mult);
    }
    static void clear(vector<double>& a) {
        fill(a.begin(), a.end(), 0.0);
    }
    static void clear(Matrix& a) {
        clear(a.data);
    }

    static void adamw_update(vector<double>& W, const vector<double>& dW, vector<double>& m, vector<double>& v,
                              double lr, double beta1, double beta2, double eps, double weight_decay,
                              int t, double grad_scale) {
        double bc1 = 1.0 - pow(beta1, t);
        double bc2 = 1.0 - pow(beta2, t);
        for (size_t i = 0; i < W.size(); i++) {
            double g = dW[i] * grad_scale;
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;
            double m_hat = m[i] / bc1;
            double v_hat = v[i] / bc2;
            W[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * W[i]);
        }
    }
    static void adamw_update(Matrix& W, const Matrix& dW, Matrix& m, Matrix& v,
                              double lr, double beta1, double beta2, double eps, double weight_decay,
                              int t, double grad_scale) {
        adamw_update(W.data, dW.data, m.data, v.data, lr, beta1, beta2, eps, weight_decay, t, grad_scale);
    }
    void times(const vector<double>& x, Span<double> y) const {
        for (int r = 0; r < num_rows; r++) {
            int base = r * num_cols;
            double sum = 0;
            for (int c = 0; c < num_cols; c++) {
                sum += data[base + c] * x[c];
            }
            y[r] = sum;
        }
    }

    vector<double> times(const vector<double>& x) const {
        vector<double> y(num_rows);
        times(x, y);
        return y;
    }
    static double dot(const vector<double> &a, const vector<double> &b) {
        double res = 0;
        for (int i = 0; i < a.size(); i++) {
            res += a[i] * b[i];
        }
        return res;
    }

    void save(ostream& out) const {
        out << num_rows << " " << num_cols << "\n";
        out << scientific << setprecision(17);
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
        for (double& v : data) in >> v;
    }

    static void save_vec(ostream& out, const vector<double>& v) {
        out << v.size() << "\n";
        out << scientific << setprecision(17);
        for (size_t i = 0; i < v.size(); i++) {
            out << v[i];
            if (i + 1 < v.size()) out << " ";
        }
        out << "\n";
    }

    static void load_vec(istream& in, vector<double>& v) {
        size_t n;
        in >> n;
        v.resize(n);
        for (double& x : v) in >> x;
    }
};
