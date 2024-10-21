# Clustering
Clustering is an unsupervised machine learning technique that groups similar objects or data points into clusters.
# Overview
This project compares the performance of three clustering algorithms:

K-Means Clustering
Hierarchical Clustering
Mean Shift Clustering
Different pre-processing techniques were applied, including normalization, transformation, PCA, and combinations of these techniques. The performance of the algorithms was evaluated using various clustering evaluation metrics.
# Dataset
We used a small dataset from the UCI Machine Learning Repository. You can choose any small dataset suitable for clustering analysis.
# Pre Processing Techniques
We used the following pre-processing techniques:

No Data Processing: Raw data without any modifications.

Normalization: Feature scaling to bring all values between a specific range (e.g., 0 to 1).

Transformation: Applying log or square root transformation to data to reduce skewness.

PCA (Principal Component Analysis): Dimensionality reduction to reduce the number of features while preserving variance.

T+N (Transformation + Normalization): Combining transformation and normalization techniques.
# Clustering algorithms
We compared the following clustering algorithms:

K-Means Clustering: A popular partitional clustering algorithm that groups data into k clusters.

Hierarchical Clustering: An agglomerative clustering technique that builds a hierarchy of clusters.

Mean Shift Clustering: A non-parametric clustering algorithm that does not require the number of clusters to be predefined.
# Evaluation metrics
We used the following metrics to evaluate the clustering performance:

Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters.

Calinski-Harabasz Index: Measures the ratio of the sum of the between-cluster dispersion and the within-cluster dispersion.

Davies-Bouldin Score: Measures the average similarity ratio of each cluster with its most similar cluster.
# How to run the code
To run this project on your local machine or Colab, follow these steps:

Prerequisites
Python 3.x

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Install the required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn
# Steps
Clone the repository:
git clone https://github.com/bhavya312-bit/Clustering-Performance-Study

Open the Jupyter Notebook or Colab: Open the clustering_analysis.ipynb notebook in Jupyter or Google Colab.

Run the notebook: Follow the steps in the notebook to preprocess the data, apply clustering algorithms, and evaluate the results

View the results: The results will be displayed in the notebook as tables and plots, comparing the performance of the algorithms.
# Conclusion
The project provided a comparative analysis of different clustering algorithms using various pre-processing techniques. The study found that Mean Shift performed better in terms of the Silhouette Score and Davies-Bouldin Score, while K-Means showed better results in terms of the Calinski-Harabasz Index.

Future work may include testing other clustering techniques (like DBSCAN) and using larger datasets for more comprehensive evaluation.
