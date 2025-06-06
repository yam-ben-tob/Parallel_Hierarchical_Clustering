#include "Point_Utils.h"

double euclidean_distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.features.size(); ++i)
        sum += (a.features[i] - b.features[i]) * (a.features[i] - b.features[i]);
    return sqrt(sum);
}

double gower_distance(const Point& a, const Point& b, const vector<double>* weights) {
    double weighted_sum = 0.0;
    double total_weight = 0.0;
    size_t n = a.features.size();

    for (size_t i = 0; i < n; ++i) {
        double w = (weights != nullptr && i < weights->size()) ? (*weights)[i] : 1.0;
        double diff = std::abs(a.features[i] - b.features[i]);
        weighted_sum += w * diff;
        total_weight += w;
    }

    return (total_weight == 0.0) ? 0.0 : weighted_sum / total_weight;
}

vector<double> create_gower_weights(double quant_weight, double binary_weight) {
    int total_features = 54;
    int quant_count = 10;
    int binary_count = total_features - quant_count;

    vector<double> weights(total_features, 0.0);

    double quant_weight_per_feature = quant_weight / quant_count;
    double binary_weight_per_feature = binary_count > 0 ? binary_weight / binary_count : 0;

    for (int i = 0; i < quant_count; ++i) {
        weights[i] = quant_weight_per_feature;
    }
    for (int i = quant_count; i < total_features; ++i) {
        weights[i] = binary_weight_per_feature;
    }

    return weights;
}

vector<Point> load_csv(const string& filename) {
    vector<Point> data;
    ifstream file(filename);
    string line;

    if (!getline(file, line)) {
        return data;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        Point p;
        while (getline(ss, value, ',')) {
            p.features.push_back(stod(value));
        }
        data.push_back(p);
    }

    return data;
}

vector<Point> choose_k_random_points(const vector<Point>& data, size_t k) {
    vector<Point> result;
    if (k > data.size()) k = data.size();

    vector<size_t> indices(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        indices[i] = i;
    }

    random_device rd;
    mt19937 gen(rd());

    shuffle(indices.begin(), indices.end(), gen);

    for (size_t i = 0; i < k; i++) {
        result.push_back(data[indices[i]]);
    }

    return result;
}
