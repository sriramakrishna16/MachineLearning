import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = np.array([
    # Cluster 1
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],

    # Cluster 2
    [8, 8],
    [8, 9],
    [9, 8],
    [9, 9],

    # Noise
    [5, 5]
])

dbscan = DBSCAN(
    eps=3,
    min_samples=3
)

labels = dbscan.fit_predict(X)

print(labels)

print("Labels:", labels)

for point, label in zip(X, labels):
    print(point, "→ Cluster", label)

# observation 
# with epsilon = 1.5 there is a continous chain of nearby points and forms single cluster and and 
#  one point as outlier

