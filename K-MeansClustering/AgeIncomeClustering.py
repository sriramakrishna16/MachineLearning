import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Age" : [20, 22, 21, 23, 25, 45, 47, 46, 48, 50],
    "Income" : [20000, 22000, 21000, 24000, 25000, 70000, 72000, 68000, 75000, 73000]
}

df = pd.DataFrame(data)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(df)

x = df[["Age", "Income"]]

x_scaled = scaler.fit_transform(x)

# finding best k using elbow method
wcss = []
for k in range(1,11):
    model = KMeans(n_clusters = k, random_state = 42)
    model.fit(x_scaled)
    wcss.append(model.inertia_)

# plt.plot(range(1,11), wcss, marker="o")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()

model = KMeans(n_clusters = 2, random_state=42)

model.fit(x_scaled)

clusters = model.predict(x_scaled)

df["clusters"] = clusters

print(df)

plt.scatter(
    df["Age"],
    df["Income"],
    c=df["clusters"]
)

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering")
plt.show()