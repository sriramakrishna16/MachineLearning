import pandas as pd

# url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"

# df = pd.read_csv(url)

# df.to_csv("D:/MachineLearning/Density-BasedSCAN/data/earthquakes.csv", index = True)

df = pd.read_csv("data/earthquakes.csv")

# print(df.head(10))
# print(df.shape)

df = df.drop("Unnamed: 0", axis = 1)

x = df[["latitude", "longitude"]]

print(x.head(10))
print(x.describe())

import matplotlib.pyplot as plt

# plt.scatter(x["latitude"], x["longitude"], s = 5)

# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Earthquake Locations")

# plt.show()

# print(x.isnull().sum())

import numpy as np

x = x.to_numpy()

x_rad = np.radians(x)

print(x_rad[:5])

eps_km = 50

earth_radius_km = 6371

eps = eps_km / earth_radius_km

print("EPS in radians: ", eps)

from sklearn.cluster import DBSCAN

# dbscan = DBSCAN(
#     eps=eps,
#     min_samples=5,
#     metric="haversine"
# )

# labels = dbscan.fit_predict(x_rad)

# df["cluster"] = labels

# unique, counts = np.unique(labels, return_counts=True)

# print("Cluster counts: \n")

# for label, count in zip(unique, counts):
#     print(f"Cluster {label}: {count} points")

eps_values = [10, 25, 50, 75, 100]

for eps_km in eps_values:

    eps = eps_km / 6371

    dbscan = DBSCAN(
        eps=eps,
        min_samples=5,
        metric="haversine"
    )

    labels = dbscan.fit_predict(x_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(
        f"eps = {eps_km} km | "
        f"clusters = {n_clusters} | "
        f"noise = {n_noise}"
    )

# plt.figure(figsize=(12, 6))

# plt.scatter(
#     df["longitude"],
#     df["latitude"],
#     c=df["cluster"],
#     s=5
# )

# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("DBSCAN Clustering of Earthquakes")

# plt.show()
