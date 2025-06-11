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

__global__ void find_local_min_kernel(
    const double* min_distances,  // [n]
    const bool* is_active,        // [n]
    double* d_local_mins,         // [p]
    int* d_local_indices,         // [p]
    int n,
    int p                         // total threads = p
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= p) return;

    // Divide n clusters among p threads
    int clusters_per_thread = (n + p - 1) / p;
    int start_cluster = thread_id * clusters_per_thread;
    int end_cluster = min(start_cluster + clusters_per_thread, n);

    double local_min = DBL_MAX;
    int local_index = -1;

    for (int i = start_cluster; i < end_cluster; ++i) {
        if (is_active[i] && min_distances[i] < local_min) {
            local_min = min_distances[i];
            local_index = i;
        }
    }

    d_local_mins[thread_id] = local_min;
    d_local_indices[thread_id] = local_index;
}

__global__ void reduce_min_kernel(
    const double* in_vals,
    const int* in_indices,
    double* out_vals,
    int* out_indices,
    int N  // total number of input values
) {
    // __shared__: shared among all threads in a block. 
    // Resides in on-chip memory, much faster than global memory
    // Preforming reduction using a block of threads over a block of value,
    // Time complexity: O(log(THREADS_PER_BLOCK)) = O(1) 
    __shared__ double shared_vals[THREADS_PER_BLOCK];
    __shared__ int shared_idxs[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Load input to shared memory
    if (global_id < N) {
        shared_vals[tid] = in_vals[global_id];
        shared_idxs[tid] = in_indices[global_id];
    } else {
        shared_vals[tid] = DBL_MAX;
        shared_idxs[tid] = -1;
    }

    // Waits until all threads in a block reach this point
    // Ensures shared memory is fully updated and visible to all threads
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] < shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        out_vals[blockIdx.x] = shared_vals[0];
        out_indices[blockIdx.x] = shared_idxs[0];
    }
}

__global__ void merge_clusters_kernel(
    int* d_clusters,     // [n]
    bool* d_is_active,   // [n]
    int old_cluster,
    int new_cluster,
    int n,
    int p                // total number of threads
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= p) return;

    // Compute the range this thread is responsible for
    int clusters_per_thread = (n + p - 1) / p;  
    int start = thread_id * clusters_per_thread;
    int end = min(start + clusters_per_thread, n);

    // Each thread processes its portion
    for (int i = start; i < end; ++i) {
        if (d_clusters[i] == old_cluster) {
            d_clusters[i] = new_cluster;
        }
    }

    // Let one thread mark old_cluster as inactive
    if (thread_id == 0) {
        d_is_active[old_cluster] = false;
    }
}


__global__ void update_distances(
    double* d_distances,        // [n * n] flattened distance matrix
    bool* d_is_active,          // [n]
    int* d_nearest_neighbors,   // [n]
    double* d_min_distances,    // [n]
    int old_cluster,
    int new_cluster,
    int n,
    int p                       // total number of threads
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= p) return;

    // Calculate the range this thread should process
    int clusters_per_thread = (n + p - 1) / p;
    int start = thread_id * clusters_per_thread;
    int end = min(start + clusters_per_thread, n);

    for (int i = start; i < end; ++i) {
        if (!d_is_active[i] || i == new_cluster) continue;

        double dist_old = d_distances[i * n + old_cluster];
        double dist_new = d_distances[i * n + new_cluster];
        double updated_dist = min(dist_old, dist_new);

        d_distances[i * n + new_cluster] = updated_dist;
        d_distances[new_cluster * n + i] = updated_dist;

        if (d_nearest_neighbors[i] == old_cluster) {
            d_nearest_neighbors[i] = new_cluster;
            d_min_distances[i] = updated_dist;
        }

        // Update nearest neighbor of new_cluster
        if (d_is_active[i] && i != new_cluster) {
            double new_dist = d_distances[i * n + new_cluster];
            if (new_dist < d_min_distances[new_cluster]) {
                d_min_distances[new_cluster] = new_dist;
                d_nearest_neighbors[new_cluster] = i;
            }
        }
    }
}

// Struct to hold GPU data pointers
struct GpuData {
    double* d_points = nullptr;         // Flattened input points array (size: n * d)
    double* d_distances = nullptr;      // distance matrix (size: n * n)
    int* d_nearest_neighbors = nullptr; // Index of nearest neighbor for each active cluster (size: n) 
    double* d_min_distances = nullptr;  // distance to nearest neighbor per cluster (size: n)   
    int* d_clusters;                    // cluster ID for each point (size: n)          
    bool* d_is_active;                  // flags indicating active clusters (size: n)    
    int n = 0;                          // Number of points
    int d = 0;                          // Dimensionality of each point           
};


// Initialization function that returns GPU pointers
GpuData initialize_on_gpu(const std::vector<Point>& points, int p) {
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
    // p threads leading to O(n^2/p) time complexity, 
    // for p=n/logn we get O(nlog(n)) time complexity as needed.
    int totalThreads = p; // adjust to n^2 and see effect
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

    // n^2 threads, O(1) time complexity
    int total_threads = gpu_data.n * gpu_data.n;
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

    // Blocks the CPU until all previously launched kernels and memory operations finish
    // across all blocks and kernels
    cudaDeviceSynchronize();
    return gpu_data;
}

int find_global_min(
    const double* d_min_distances,   // [n]
    const bool* d_is_active,         // [n]
    int n,
    int p,
) {
    // Allocate intermediate buffers
    double* d_local_mins;
    int* d_local_indices;
    cudaMalloc(&d_local_mins, p * sizeof(double));
    cudaMalloc(&d_local_indices, p * sizeof(int));

    // Launch local min kernel (p threads)
    int blocks = (p + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    find_local_min_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_min_distances, d_is_active,
        d_local_mins, d_local_indices,
        n, p
    );

    // Stage 1 reduction
    int blocks_stage1 = (p + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double* d_stage1_mins;
    int* d_stage1_indices;
    cudaMalloc(&d_stage1_mins, blocks_stage1 * sizeof(double));
    cudaMalloc(&d_stage1_indices, blocks_stage1 * sizeof(int));

    reduce_min_kernel<<<blocks_stage1, THREADS_PER_BLOCK>>>(
        d_local_mins, d_local_indices,
        d_stage1_mins, d_stage1_indices,
        p
    );

    // Stage 2 reduction
    double* d_global_min_val;
    int* d_global_min_idx;
    cudaMalloc(&d_global_min_val, sizeof(double));
    cudaMalloc(&d_global_min_idx, sizeof(int));

    int stage2_threads = 1;
    while (stage2_threads < blocks_stage1 && stage2_threads < threads_stage1)
        stage2_threads *= 2;

    reduce_min_kernel<<<1, stage2_threads>>>(
        d_stage1_mins, d_stage1_indices,
        d_global_min_val, d_global_min_idx,
        blocks_stage1
    );

    // Copy result to host
    int h_global_min_idx;
    cudaMemcpy(&h_global_min_idx, d_global_min_idx, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_local_mins);
    cudaFree(d_local_indices);
    cudaFree(d_stage1_mins);
    cudaFree(d_stage1_indices);
    cudaFree(d_global_min_val);
    cudaFree(d_global_min_idx);

    return h_global_min_idx; 
}

void cleanup_gpu_data(GpuData& gpu_data) {
    cudaFree(gpu_data.d_points);
    cudaFree(gpu_data.d_distances);
    cudaFree(gpu_data.d_nearest_neighbors);
    cudaFree(gpu_data.d_min_distances);
    cudaFree(gpu_data.d_clusters);
    cudaFree(gpu_data.d_is_active);
}


int* run_single_linkage_clustering(GPUData& gpu_data, int p) {
    GpuData gpu_data = initialize_on_gpu(points, p);
    
    int blocks = (p + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // continue here, 2 rounds of reduction ,max 3, depends on num of threads in a block which have shared memory.
    // how to split clusters between threads?
    
    for (int iter = 0; iter < gpu_data.n - 1; ++iter) {
        // Step 1: Find global minimum of min_distances (still on CPU for now)
        int min_index = find_global_min(gpu_data.d_min_distances, gpu_data.d_is_active, gpu_data.n, p);
        
        // Step 2: Merge clusters
        int new_cluster;
        cudaMemcpy(&new_cluster, &gpu_data.d_nearest_neighbors[min_index], sizeof(int), cudaMemcpyDeviceToHost);
        merge_clusters_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_clusters,
            d_is_active,
            min_index,
            new_cluster,
            gpu_data.n,
            p
        );
        udaDeviceSynchronize();

        // Step 3: Update distances to the new cluster (parallel with p threads)
        update_distances<<<blocks, THREADS_PER_BLOCK>>>(
            gpu_data.d_distances,
            gpu_data.d_is_active,
            gpu_data.d_nearest_neighbors,
            gpu_data.d_min_distances,
            min_index,
            new_cluster,
            gpu_data.n,
            p  
        );
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

     // Step 4: Copy final clusters and activity arrays back to host if needed
    int* h_clusters = new int[gpu_data.n];
    cudaMemcpy(h_clusters, gpu_data.d_clusters, gpu_data.n * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 5: Free GPU memory 
    cleanup_gpu_data(gpu_data);

    return h_clusters;
}

