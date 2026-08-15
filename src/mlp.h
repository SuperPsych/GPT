#pragma once
#include <omp.h>
#include "sample.h"
#include "transform.h"
#include "thread_data.h"
#include "stats.h"

struct MLP {
    vector<Transform> transforms;
    vector<double> x_means;
    vector<double> x_stdevs;
    MLP(const vector<int>& dims) {
        for (int i=0; i<dims.size()-1; i++) {
            transforms.emplace_back(dims[i], dims[i+1]);
        }
    }
    static void activation(vector<double> &x) {
        Transform::ReLU(x);
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

    void normalize(vector<double>& x){
        for(int i=0; i<x.size(); i++){
            x[i] -= x_means[i];
            if(x_stdevs[i] !=0) x[i] /= x_stdevs[i];
        }
    }

    vector<double> infer(vector<double> x){
        for (int i = 0; i < transforms.size()-1; i++) {
            x = transforms[i].transform(x);
            activation(x);
        }
        x = transforms.back().transform(x);
        output_activation(x);
        return x;
    }

    vector<double> predict(vector<double> x) {
        normalize(x);
        auto y = infer(x);
        return y;
    }

    void forward_pass(const vector<double> &x, vector<vector<double>>& activations) {
        activations[0] = x;
        for (int i = 0; i < transforms.size()-1; i++) {
            transforms[i].transform(activations[i], activations[i+1]);
            activation(activations[i+1]);
        }
        transforms.back().transform(activations[activations.size()-2], activations.back());
        output_activation(activations.back());
    }
    void backprop(const Sample& sample, ThreadData& td) {
        const vector<double>& x = sample.x;
        const vector<double>& y = sample.y;
        forward_pass(x, td.activations);
        vector<double> grad(transforms.back().dim_out);

        for (int i=0; i<y.size(); i++) {
            grad[i] = td.activations.back()[i] - y[i];
        }

        for (int i=0; i<transforms.size(); i++) {
            int idx = transforms.size()-i-1;
            Transform& transform = transforms[idx];
            Matrix& A_delta = td.A_deltas[idx];
            vector<double>& b_delta = td.b_deltas[idx];
            vector<double>& activation = td.activations[idx];

            Matrix::add(b_delta, grad);

            for (int r = 0; r < transform.dim_out; r++) {
                for (int c = 0; c < transform.dim_in; c++) {
                    A_delta.at(r,c) += grad[r]*activation[c];
                }
            }
            vector<double> new_grad(transform.dim_in);
            Matrix& A = transform.A;
            for (int r=0; r<transform.dim_out; r++) {
                for (int c=0; c<transform.dim_in; c++) {
                    new_grad[c] += A.at(r,c) * grad[r];
                }
            }
            grad.swap(new_grad);
            for (int j=0; j<grad.size(); j++) {
                if (activation[j] == 0) grad[j] = 0;
            }
        }
    }

    void normalize(vector<Sample>& samples){
        for(int i=0; i<samples[0].x.size(); i++){
            for(int j=0; j<samples.size(); j++){
                samples[j].x[i] -= x_means[i];
                if(x_stdevs[i]!=0) samples[j].x[i] /= x_stdevs[i];
            }
        }
    }

    void normalize_init(vector<Sample>& samples){
        vector<double> arr(samples.size());
        for(int i=0; i<samples[0].x.size(); i++){
            for(int j=0; j<arr.size(); j++){
                arr[j] = samples[j].x[i];
            }
            double m = mean(arr);
            double s = stdev(arr);
            x_means.emplace_back(m);
            x_stdevs.emplace_back(s);
            for(int j=0; j<samples.size(); j++){
                samples[j].x[i] -= m;
                if(s!=0) samples[j].x[i] /= s;
            }
        }
    }

    void train(vector<Sample>& train_samples, vector<Sample>& test_samples, int epochs, int batch_size, double lr) {
        normalize_init(train_samples);
        omp_set_dynamic(0);
        int thread_num = omp_get_max_threads();
        vector<ThreadData> thread_data(thread_num, ThreadData(dims()));

        cout << "Beginning training..." << endl;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0; b < train_samples.size()/batch_size; b++) {
                int start = b * batch_size;
                for(ThreadData& td : thread_data){
                    td.clear();
                }
                #pragma omp parallel for
                for(int i = 0; i < batch_size; i++){
                    backprop(train_samples[start + i], thread_data[omp_get_thread_num()]);
                }

                #pragma omp parallel for
                for(int layer = 0; layer < transforms.size(); layer++){
                    for(int t = 1; t < thread_num; t++){
                        Matrix::add(thread_data[0].A_deltas[layer], thread_data[t].A_deltas[layer]);
                        Matrix::add(thread_data[0].b_deltas[layer], thread_data[t].b_deltas[layer]);
                    }
                }

                for(int t = 0; t < transforms.size(); t++){
                    transforms[t].step(thread_data[0].A_deltas[t], thread_data[0].b_deltas[t], lr / batch_size);
                }
            }
            if((epoch+1) % 5 == 0){
                cout << "Epoch " << epoch+1 << "/" << epochs << 
                " | Train Loss: " << loss(train_samples) <<
                " | Accuracy: " << accuracy(test_samples) << endl;
            }
        }
    }

    static double loss(const vector<double>& a, const vector<double>& p) {
        double res = 0.0;
        for (int i = 0; i < a.size(); i++) {
            res -= a[i] * log(p[i]);
        }
        return res;
    }

    double loss(const vector<Sample>& samples){
        double res = 0.0;
        for(const Sample& sample : samples){
            res += loss(sample.y, infer(sample.x))/samples.size();
        }
        return res;
    }

    int collapse(vector<double>& x){
        double max = INT_MIN;
        int res = -1;
        for(int i=0; i<x.size(); i++){
            if(x[i] > max){
                max = x[i];
                res = i;
            }
        }
        return res;
    }

    double accuracy(vector<Sample>& samples){
        vector<double> y(samples[0].y.size());
        int pred, actual;
        int correct = 0;
        for(Sample& sample : samples){
            y = predict(sample.x);
            pred = collapse(y);
            actual = collapse(sample.y);
            if(pred==actual) correct++;
        }
        return (double) correct / samples.size();
    }

    void eval(vector<Sample>& samples){
        cout << "Evaluating mlp..." << endl;
        cout << "Accuracy: " << accuracy(samples) << endl;
    }

    static MLP initialize(int input_dim, int output_dim, int hidden_dim, int hidden_num) {
        cout << "Initializing mlp..." << endl;
        vector<int> dims;
        dims.emplace_back(input_dim);
        for (int i=0; i<hidden_num; i++) {
            dims.emplace_back(hidden_dim);
        }
        dims.emplace_back(output_dim);
        return MLP(dims);
    }

    vector<int> dims() {
        vector<int> dims;
        for (Transform& t : transforms) {
            dims.emplace_back(t.dim_in);
        }
        dims.emplace_back(transforms.back().dim_out);
        return dims;
    }
};
