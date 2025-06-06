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

---

*CSE305 Project: Parallel Hierarchical Clustering*
