# Clustering Performance Study

This project provides a comparative analysis of three clustering algorithms using different pre-processing techniques. Clustering is an unsupervised machine learning technique used to group similar data points into clusters, and this study evaluates their performance using standard evaluation metrics.

---

## Overview

The project compares the performance of the following clustering algorithms:

1. **K-Means Clustering**: A popular partitional clustering algorithm that groups data into k clusters.
2. **Hierarchical Clustering**: An agglomerative clustering technique that builds a hierarchy of clusters.
3. **Mean Shift Clustering**: A non-parametric clustering algorithm that does not require the number of clusters to be predefined.

Various pre-processing techniques were applied to the dataset, and the performance of these algorithms was evaluated using clustering metrics.

---

## Dataset

The dataset used for this project is a small dataset from the UCI Machine Learning Repository. Any suitable small dataset for clustering analysis can be used.

---

## Pre-Processing Techniques

We applied the following pre-processing techniques:

1. **No Data Processing**: Raw data without modifications.
2. **Normalization**: Feature scaling to bring all values within a specific range (e.g., 0 to 1).
3. **Transformation**: Applying log or square root transformations to reduce skewness in data.
4. **PCA (Principal Component Analysis)**: Dimensionality reduction to reduce the number of features while preserving variance.
5. **T+N (Transformation + Normalization)**: Combining transformation and normalization techniques.

---

## Clustering Algorithms

The following clustering algorithms were evaluated:

1. **K-Means Clustering**: Groups data into a predefined number of clusters.
2. **Hierarchical Clustering**: Builds a hierarchy of clusters using an agglomerative approach.
3. **Mean Shift Clustering**: Automatically determines the number of clusters based on density estimation.

---

## Evaluation Metrics

The clustering algorithms were evaluated using the following metrics:

1. **Silhouette Score**: Measures how similar a data point is to its own cluster compared to other clusters.
2. **Calinski-Harabasz Index**: Ratio of the between-cluster dispersion to within-cluster dispersion.
3. **Davies-Bouldin Score**: Measures the average similarity ratio of each cluster with its most similar cluster.

---

## How to Run the Code

### Prerequisites

- **Python 3.x**
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

Install the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bhavya312-bit/Clustering-Performance-Study
   cd Clustering-Performance-Study
   ```

2. **Open the Notebook**
   Open `clustering_analysis.ipynb` in Jupyter Notebook or Google Colab.

3. **Run the Notebook**
   Follow the steps in the notebook to:
   - Preprocess the data
   - Apply clustering algorithms
   - Evaluate the results

4. **View Results**
   Results are displayed as tables and plots within the notebook, comparing the performance of the algorithms.

---

## Results

- **Mean Shift Clustering** performed better on metrics such as Silhouette Score and Davies-Bouldin Score.
- **K-Means Clustering** showed superior performance based on the Calinski-Harabasz Index.

---

## Conclusion

This project provided insights into the performance of different clustering algorithms under various pre-processing techniques. The results highlighted the strengths of each algorithm under different metrics.

### Future Work

- Explore other clustering techniques, such as **DBSCAN**.
- Use larger and more diverse datasets for a comprehensive evaluation.

---

