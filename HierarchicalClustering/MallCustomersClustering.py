# import kagglehub

# path = kagglehub.dataset_download("shwetabh123/mall-customers", 
#                                   output_dir = "D:/MachineLearning/HierarchicalClustering/data")

# print("Path to dataset files:", path)

import pandas as pd

df = pd.read_csv("data/Mall_Customers.csv")

print(df.shape)
print(df.head(10))
print(df.isnull().sum())

df = df.drop("CustomerID", axis = 1)

df["Genre"] = df["Genre"].map({
    "Male": 1,
    "Female": 0
})

from sklearn.preprocessing import StandardScaler

x_scaled = StandardScaler().fit_transform(df)

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering

linked = linkage(x_scaled, method = "complete")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

dendrogram(linked)

plt.xlabel("Customers")
plt.ylabel("Distance")
plt.title("Hierarchical clustering - complete linkage")

plt.show()

# by observing the plot , k = 2 is best for single linkage

model = AgglomerativeClustering(n_clusters=2,linkage="complete")

clusters = model.fit_predict(x_scaled)

df["clusters"] = clusters

print(df.head(20))
print(df["clusters"].value_counts())

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["clusters"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Hierarchical Clustering - Single Linkage")
plt.show()

print(
    df.groupby("clusters")[
        ["Age", "Genre", "Annual Income (k$)", "Spending Score (1-100)"]
    ].mean()
)

# observation
""" 
K=2 → broad segmentation into two primary cluster(cutting using large vertical gap)
K=4 → detailed segmentation 
and single linkage approach overlaps and forms closed chains because gender values 0 and 1
so complete and average is best 
"""