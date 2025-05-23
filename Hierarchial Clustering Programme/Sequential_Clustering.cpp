#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <set>
#include <algorithm>
#include <random>
#include <ctime> 

using namespace std;

struct Point {
    vector<double> features;
};

double euclidean_distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.features.size(); ++i)
        sum += (a.features[i] - b.features[i]) * (a.features[i] - b.features[i]);
    return sqrt(sum);
}

double gower_distance(
    const Point& a,
    const Point& b,
    const std::vector<double>* weights = nullptr
) {
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

std::vector<double> create_gower_weights(double quant_weight, double binary_weight) {
    int total_features = 54;  // fixed total number of features
    int quant_count = 10;     // fixed number of quantitative features
    int binary_count = total_features - quant_count;

    std::vector<double> weights(total_features, 0.0);

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

// Load CSV into vector<Point>
vector<Point> load_csv(const string& filename) {
    vector<Point> data;
    ifstream file(filename);
    string line;

    // Skip header line
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

    // Create a copy of indices
    vector<size_t> indices(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        indices[i] = i;
    }

    // Random engine
    random_device rd;
    mt19937 gen(rd());

    // Shuffle indices
    shuffle(indices.begin(), indices.end(), gen);

    // Pick first k indices
    for (size_t i = 0; i < k; i++) {
        result.push_back(data[indices[i]]);
    }

    return result;
}

// Agglomerative clustering: single-link
void hierarchical_clustering_sl(vector<Point>& data, int target_clusters) {
    int n = data.size();
    vector<set<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i].insert(i);

    vector<vector<double>> dist(n, vector<double>(n, numeric_limits<double>::max()));
    std::vector<double> weights = create_gower_weights(0.7, 0.3);

    // Precompute distances
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            dist[i][j] = gower_distance(data[i], data[j], &weights);
            dist[j][i] = dist[i][j];
        }

    while (clusters.size() > target_clusters) {
        int a = -1, b = -1;
        double min_dist = numeric_limits<double>::max();

        // Find closest pair of clusters
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                double d = numeric_limits<double>::max();
                for (int x : clusters[i]) {
                    for (int y : clusters[j]) {
                        d = min(d, dist[x][y]);
                    }
                }
                if (d < min_dist) {
                    min_dist = d;
                    a = i;
                    b = j;
                }
            }
        }

        // Merge clusters a and b
        clusters[a].insert(clusters[b].begin(), clusters[b].end());
        clusters.erase(clusters.begin() + b); // b > a always since j > i
    }

    // Output cluster assignments
    int label = 0;
    for (const auto& cluster : clusters) {
        for (int idx : cluster) {
            cout << "Point " << idx << " -> Cluster " << label << endl;
        }
        label++;
    }
}

// Agglomerative clustering: average-link
vector<set<int>> hierarchical_clustering(vector<Point>& data, int target_clusters) {
    int n = data.size();
    vector<set<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i].insert(i);

    vector<vector<double>> dist(n, vector<double>(n, numeric_limits<double>::max()));

    // Precompute distances
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            dist[i][j] = gower_distance(data[i], data[j]);
            dist[j][i] = dist[i][j];
        }

    while ((int)clusters.size() > target_clusters) {
        int a = -1, b = -1;
        double min_dist = numeric_limits<double>::max();

        // Find closest pair of clusters using average linkage
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                double sum_dist = 0.0;
                int count = 0;
                for (int x : clusters[i]) {
                    for (int y : clusters[j]) {
                        sum_dist += dist[x][y];
                        count++;
                    }
                }
                double d = sum_dist / count;

                if (d < min_dist) {
                    min_dist = d;
                    a = i;
                    b = j;
                }
            }
        }

        // Merge clusters a and b
        clusters[a].insert(clusters[b].begin(), clusters[b].end());
        clusters.erase(clusters.begin() + b); // b > a always since j > i
    }

    return clusters;
}


int main() {
    clock_t start = clock();

    vector<Point> data = load_csv("covtype_processed.csv");

    int k = 2000; 
    int target_clusters = 7; // There are 7 cover types, according to the dataset information

    vector<Point> sampled_data = choose_k_random_points(data, k);
    vector<set<int>> clusters = hierarchical_clustering(sampled_data, target_clusters);

    // Print cluster sizes
    cout << "Cluster sizes:\n";
    for (size_t i = 0; i < clusters.size(); ++i) {
        cout << "Cluster " << i + 1 << ": " << clusters[i].size() << " points\n";
    }

    clock_t end = clock();

    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed_secs << " seconds\n";

    return 0;
}


