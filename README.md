# Parallel Hierarchical Clustering Project

## Introduction
Hierarchical clustering is a method of organizing data into a tree-like structure based on similarities. This project focuses on the single-link hierarchical clustering method, which iteratively merges the closest clusters until all data points form one cluster, represented as a dendrogram.

Hierarchical clustering is widely used for customer segmentation, gene expression analysis, document classification, and social network analysis. Unlike other clustering methods, it does not require specifying the number of clusters upfront and reveals relationships at multiple levels.

## Project Goals
- Implement a sequential single-link hierarchical clustering algorithm.
- Design and implement a parallel version using CUDA GPU to accelerate distance computations.
- Compare runtime performance across different numbers of GPU threads and benchmark datasets of varying sizes.

## Key Highlights
- The project addresses the O(n²) complexity challenge of hierarchical clustering by parallelizing inter-cluster distance updates.
- Parallelization improves scalability and reduces runtime without altering clustering results.
- The CUDA GPU implementation is the main focus, showcasing significant speedup over the serial approach.

## Build and Run

### Compilation

This project includes a utility file `Point_Utils.h` and its implementation `Point_Utils.cpp` which are required by the main source file `sequential_clustering.cpp`.

To compile the sequential clustering program, run the following command in your terminal:

```bash
g++ sequential_clustering.cpp Point_Utils.cpp -o sequential_clustering -O2

```
This command compiles both source files and links them into a single executable named sequential_clustering.

### Running

After successfully compiling the program, you can run the executable as follows:

```bash
./sequential_clustering

```
By default, the program expects the input CSV file covtype_processed.csv to be located in the same directory as the executable.
---


*CSE305 Project: Parallel Hierarchical Clustering*
