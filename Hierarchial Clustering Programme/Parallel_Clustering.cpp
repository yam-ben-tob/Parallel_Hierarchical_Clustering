#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <algorithm>

#include "Point_Utils.h"

#define THREADS_PER_BLOCK 256

__device__ double gower_distance_device(const double* a, const double* b, const double* weights, int dim) {
    double weighted_sum = 0.0;
    double total_weight = 0.0;

    for (int i = 0; i < dim; ++i) {
        double w = (weights != nullptr) ? weights[i] : 1.0;
        double diff = fabs(a[i] - b[i]);
        weighted_sum += w * diff;
        total_weight += w;
    }

    return (total_weight == 0.0) ? 0.0 : (weighted_sum / total_weight);
}

__global__ void compute_distances_gower(const double* points, const double* weights, double* distances, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx < total) {
        int i = idx / n;
        int j = idx % n;
        if (i == j) {
            distances[i * n + j] = 0.0;
            return;
        }

        // pointers to feature vectors for points i and j
        const double* pi = points + i * d;
        const double* pj = points + j * d;

        distances[i * n + j] = gower_distance_device(pi, pj, weights, d);
    }
}

__global__ void find_nearest_neighbors(const double* distances, int* nearest_neighbors, double* min_distances, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double min_d = 1e20;  
    int best_j = -1;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        double d = distances[i * n + j];
        if (d < min_d) {
            min_d = d;
            best_j = j;
        }
    }

    nearest_neighbors[i] = best_j;
    min_distances[i] = min_d;
}

__global__ void initialize_clusters_and_activity(int* d_clusters, bool* d_is_active, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_clusters[idx] = idx;    // Each point starts as its own cluster
        d_is_active[idx] = true;  // All clusters initially active
    }
}



// Struct to hold GPU data pointers
struct GpuData {
    double* d_points = nullptr;
    double* d_distances = nullptr;
    int* d_nearest_neighbors = nullptr;  
    double* d_min_distances = nullptr; 
    int* d_clusters;
    bool* d_is_active;  
    int n = 0;
    int d = 0;
};


// Initialization function that returns GPU pointers
GpuData initialize_on_gpu(const std::vector<Point>& points) {
    GpuData gpu_data;
    gpu_data.n = points.size();
    if (gpu_data.n == 0) return gpu_data;
    gpu_data.d = points[0].features.size();

    // Flatten points to 1D array
    std::vector<double> flat_points(gpu_data.n * gpu_data.d);
    for (int i = 0; i < gpu_data.n; ++i) {
        for (int dim = 0; dim < gpu_data.d; ++dim) {
            flat_points[i * gpu_data.d + dim] = points[i].features[dim];
        }
    }

    // Allocate device memory
    cudaMalloc(&gpu_data.d_points, gpu_data.n * gpu_data.d * sizeof(double));
    cudaMalloc(&gpu_data.d_distances, gpu_data.n * gpu_data.n * sizeof(double));

    // Copy points to device
    cudaMemcpy(gpu_data.d_points, flat_points.data(), gpu_data.n * gpu_data.d * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute distance matrix
    double* d_weights = nullptr; 
    int totalThreads = gpu_data.n * gpu_data.n; // n^2 threads, unlike article's method (log(n) threads)
    int blocks = (totalThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_distances_gower<<<blocks, THREADS_PER_BLOCK>>>(
        gpu_data.d_points,
        d_weights,  
        gpu_data.d_distances,
        gpu_data.n,
        gpu_data.d
    );

    // Device arrays for nearest neighbor index and distance
    int* d_nearest_neighbors;
    double* d_min_distances;

    cudaMalloc(&d_nearest_neighbors, gpu_data.n * sizeof(int));
    cudaMalloc(&d_min_distances, gpu_data.n * sizeof(double));

    int total_threads = gpu_data.n;
    int num_blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch one thread per point to find its nearest neighbor
    find_nearest_neighbors_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        gpu_data.d_distances,
        d_nearest_neighbors,
        d_min_distances,
        gpu_data.n
    );

    // Allocate clusters and activity arrays
    cudaMalloc(&gpu_data.d_clusters, gpu_data.n * sizeof(int));
    cudaMalloc(&gpu_data.d_is_active, gpu_data.n * sizeof(bool));

    initialize_clusters_and_activity<<<num_blocks, THREADS_PER_BLOCK>>>(
        gpu_data.d_clusters,
        gpu_data.d_is_active,
        gpu_data.n
    );

    cudaDeviceSynchronize();
    return gpu_data;
}

void run_single_linkage_clustering(GPUData& gpu_data, int p) {
    GpuData gpu_data = initialize_on_gpu(points);

    for (int iter = 0; iter < gpu_data.n - 1; ++iter) {
        // Step 1: Find global minimum of min_distances
        int min_index;
        double min_value;
        find_global_minimum(gpu_data.d_min_distances, gpu_data.d_is_active, gpu_data.n, min_index, min_value);

        // Step 2: Merge clusters
        merge_clusters<<<1, 1>>>(
            gpu_data.d_clusters,
            gpu_data.d_is_active,
            gpu_data.d_nearest_neighbors,
            min_index,
            gpu_data.n
        );
        cudaDeviceSynchronize();

        // Step 3: Update distances to the new cluster
        update_distances<<<(gpu_data.n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            gpu_data.d_distances,
            gpu_data.d_is_active,
            min_index,
            gpu_data.n
        );
        cudaDeviceSynchronize();

        // Step 4: Update nearest neighbors
        update_nearest_neighbors<<<(gpu_data.n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            gpu_data.d_distances,
            gpu_data.d_is_active,
            gpu_data.d_nearest_neighbors,
            gpu_data.d_min_distances,
            gpu_data.n
        );
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();

     // Step 3: Copy final clusters and activity arrays back to host if needed
    std::vector<int> clusters = copy_clusters_to_host(gpu_data);
    std::vector<bool> activity = copy_activity_to_host(gpu_data);

    // Step 4: Free GPU memory (implement a cleanup function)
    cleanup_gpu_data(gpu_data);
}


// Call this when done with data to free GPU memory
void free_gpu_data(GpuData& gpu_data) {
    if (gpu_data.d_points) cudaFree(gpu_data.d_points);
    if (gpu_data.d_distances) cudaFree(gpu_data.d_distances);
    gpu_data.d_points = nullptr;
    gpu_data.d_distances = nullptr;
}

