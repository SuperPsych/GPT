#pragma once
#include <bits/stdc++.h>
using namespace std;

struct Sample {
    vector<double> x;
    vector<double> y;
    Sample() = default;
    Sample(vector<double>& x, vector<double>& y) : x(move(x)), y(move(y)) {}
    void print(){
        cout << "x: ";
        for(double i : x) cout << i << ", ";
        cout << "y: ";
        for(double i : y) cout << i << ", ";
    }
};
