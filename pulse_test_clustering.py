import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the pulse test data from CSV
data = pd.read_csv("pulse_test_data_per_cycle.csv")

# Pivot the data to form a matrix where each row represents a cycle, and each column is a voltage at a specific time
pulse_data = data.pivot_table(index="Cycle", columns="Time [s]", values="Terminal voltage [V]")

# Fill missing values (if any) and standardize the data (scaling)
pulse_data_filled = pulse_data.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pulse_data_filled)

# Apply PCA for dimensionality reduction (to 2 components for visualization)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Apply KMeans clustering (you can adjust the number of clusters as needed)
num_clusters = 4  # Example: set the number of clusters to 4
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(scaled_data)

# Plot the PCA-transformed data and color by cluster
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap="viridis", s=50)
plt.title("PCA of Pulse Test Cycles with K-Means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Output the cycle numbers for each cluster
for cluster in range(num_clusters):
    cycle_numbers = pulse_data_filled.index[clusters == cluster].tolist()
    print(f"Cluster {cluster}: Cycles {cycle_numbers}")
