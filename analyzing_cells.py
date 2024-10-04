import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the CSV file into a DataFrame
df = pd.read_csv('cycle_variables_data_5.csv')

# Select all numerical features excluding 'time' (replace 'time' with your actual time column name)
X = df.select_dtypes(include=[float, int]).drop(columns=["Time [s]","Change in time [s]","Time [h]","Change in time [h]", "Cycle number",])  # Replace 'time' with the actual column name
# Handle missing data if necessary (e.g., by dropping or filling NaN values)
X = X.dropna()  # You can also use fillna() to replace missing values if needed
print(X.columns)
# Normalize the data using StandardScaler (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering to all features
kmeans = KMeans(n_clusters=10, random_state=42)  # Adjust n_clusters based on your data
labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original dataframe
df['Cluster'] = labels

# Evaluate the clustering with silhouette score
sil_score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {sil_score}')

# Visualize the clusters (for high-dimensional data, use PCA or t-SNE)
from sklearn.decomposition import PCA

# Reduce the dimensionality for visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA-reduced data with cluster labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering with All Features (Excluding Time) - PCA Reduced')
plt.show()

# Example of detecting outliers in the first cluster
outlier = df[df['Cluster'] == 0]  # Adjust the cluster number as needed
print(outlier)
