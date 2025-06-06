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
#include <tuple>
#include <queue>
#include <unordered_set>
#include <cstdio>

#include "Point_Utils.h"

using namespace std;

// Single-link: O(n^3) time complexity
// Insuring result correctness by using simple algorithm
vector<set<int>> hierarchical_clustering_sl_slow(vector<Point>& data, int target_clusters) {
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

    return clusters;
}

// Single-link: O(n^2) time complexity
// Taken from the "Parallel algorithms for hierarchical clustering" article
vector<set<int>> hierarchical_clustering_sl_fast(vector<Point>& data, int target_clusters) {
    int n = data.size();

    // Precompute distance matrix (upper triangle)
    vector<vector<double>> dist(n, vector<double>(n, numeric_limits<double>::max()));
    std::vector<double> weights = create_gower_weights(0.7, 0.3);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            dist[i][j] = gower_distance(data[i], data[j], &weights);
            // only using upper triangle of the matrix dist (i < j)
        }

    // Track number of active clusters and sizes
    int current_clusters = n;

    // For each cluster, store nearest neighbor cluster and distance
    vector<int> nearest_neighbor(n, -1);
    vector<double> min_dist(n, numeric_limits<double>::max());

    // Initialize nearest neighbors for each point
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int u = min(i, j), v = max(i, j); // only using upper triangle of dist
            double d = dist[u][v];
            if (d < min_dist[i]) {
                min_dist[i] = d;
                nearest_neighbor[i] = j;
            }
        }
    }

    // Store cluster membership
    vector<set<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i].insert(i);

    while (current_clusters > target_clusters) {
        // Find closest pair of clusters (a,b)
        double global_min = numeric_limits<double>::max();
        int a = -1, b = -1;
        for (int i = 0; i < n; ++i) {
            if (clusters[i].empty()) continue;  // cluster inactive
            if (min_dist[i] < global_min) {
                global_min = min_dist[i];
                a = i;
                b = nearest_neighbor[i];
            }
        }

        if (a == -1 || b == -1) break; // no valid merge found

        // Merge cluster b into cluster a
        clusters[a].insert(clusters[b].begin(), clusters[b].end());
        clusters[b].clear();
        // Reset a's nearest neighbor
        min_dist[a] = numeric_limits<double>::max();

        // Update distances and nearest neighbors after merge
        for (int i = 0; i < n; ++i) {
            if (clusters[i].empty() || i == a) continue;
            
            int u_ai = min(a, i), v_ai = max(a, i);
            int u_bi = min(b, i), v_bi = max(b, i);

            dist[u_ai][v_ai] = min(dist[u_ai][v_ai], dist[u_bi][v_bi]);

            double new_dist = dist[u_ai][v_ai];

            // Only if i's nearest neighbor was b (now merged into a)
            if (nearest_neighbor[i] == b) {
                min_dist[i] = new_dist;
                nearest_neighbor[i] = a;
            }

            // Only if a now sees a better neighbor in i
            if (new_dist < min_dist[a]) {
                min_dist[a] = new_dist;
                nearest_neighbor[a] = i;
            }
        }

        current_clusters--;
    }

    // Extract active clusters only
    vector<set<int>> result;
    for (int i = 0; i < n; ++i) {
        if (!clusters[i].empty())
            result.push_back(move(clusters[i]));
    }

    return result;
}


// Average-link: O(n^2*log(n)) time complexity
// Examining a different metric over the data
vector<set<int>> hierarchical_clustering_al(vector<Point>& data, int target_clusters) {
    int n = data.size();
    vector<set<int>> clusters(n);
    vector<int> sizes(n, 1);
    for (int i = 0; i < n; ++i) clusters[i].insert(i);

    vector<vector<double>> dist(n, vector<double>(n, numeric_limits<double>::max()));
    std::vector<double> weights = create_gower_weights(0.7, 0.3);

    // Precompute distances
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            dist[i][j] = gower_distance(data[i], data[j], &weights);
            dist[j][i] = dist[i][j];
        }
    
    // Active cluster indicator
    vector<bool> active(n, true);

    // Priority queue for min distance
    std::priority_queue<
        std::tuple<double, int, int>,
        std::vector<std::tuple<double, int, int>>,
        std::greater<std::tuple<double, int, int>>
    > min_heap;    
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            min_heap.emplace(dist[i][j], i, j);

    int current_clusters = n;

    while (current_clusters > target_clusters) {
        // Pop until valid pair
        double d;
        int a, b;
        do {
            tie(d, a, b) = min_heap.top();
            min_heap.pop();
        } while (!active[a] || !active[b]);

        // Merge b into a
        clusters[a].insert(clusters[b].begin(), clusters[b].end());
        sizes[a] += sizes[b];
        active[b] = false;
        current_clusters--;

        // Update distances from cluster a to others
        for (int k = 0; k < n; ++k) {
            if (k == a || !active[k]) continue;

            double d_ak = dist[a][k];
            double d_bk = dist[b][k];
            double updated = (sizes[a] - sizes[b]) * d_ak + sizes[b] * d_bk;
            updated /= sizes[a];

            dist[a][k] = dist[k][a] = updated;
            min_heap.emplace(updated, min(a, k), max(a, k));
        }
    }

    // Extract only active clusters
    vector<set<int>> result;
    for (int i = 0; i < n; ++i)
        if (active[i])
            result.push_back(move(clusters[i]));

    return result;
    
}


int main() {
    vector<Point> data = load_csv("covtype_processed.csv");

    int target_clusters = 7; // having 7 types of forests in the dataset

    // k sequence, capped at 30000 max
    std::vector<int> k_values = {
    100, 200, 400, 800, 1600, 3200, 
    5000, 7000, 9000, 11000, 13000, 
    16000, 19000, 22000, 25000, 28000, 30000
    };


    printf("k\tTime (seconds)\n");
    printf("-------------------------\n");

    for (int k : k_values) {
    vector<Point> sampled_data = choose_k_random_points(data, k);

    clock_t start = clock();
    vector<set<int>> clusters = hierarchical_clustering_sl_fast(sampled_data, target_clusters);
    clock_t end = clock();

    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    // Print time
    printf("%d\t%.3f\n", k, elapsed_secs);

    // Print cluster sizes after clustering
    printf("Cluster sizes for k=%d: ", k);
    for (const auto& cluster : clusters) {
        printf("%lu ", cluster.size());
    }
    printf("\n");
    }

}


