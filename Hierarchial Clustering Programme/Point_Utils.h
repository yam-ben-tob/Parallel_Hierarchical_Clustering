#ifndef POINT_UTILS_H
#define POINT_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

struct Point {
    vector<double> features;
};

double euclidean_distance(const Point& a, const Point& b);

double gower_distance(const Point& a, const Point& b, const vector<double>* weights = nullptr);

vector<double> create_gower_weights(double quant_weight, double binary_weight);

vector<Point> load_csv(const string& filename);

vector<Point> choose_k_random_points(const vector<Point>& data, size_t k);

#endif 
