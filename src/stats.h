#pragma once
#include <bits/stdc++.h>
using namespace std;

double mean(const vector<double>& nums){
    double res = 0;
    for(double i : nums) res += i;
    return res/nums.size();
}

double stdev(const vector<double>& nums){
    double m = mean(nums);
    double res = 0;
    for(double i : nums) res += (i-m)*(i-m);
    return sqrt(res/(nums.size()-1));
}
