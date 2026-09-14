# import kagglehub

# path = kagglehub.dataset_download("debarghamitraroy/mall-customers-segmentation-dataset", 
#                                   output_dir = "D:/MachineLearning/K-MeansClustering/data")

# print("Path to dataset files:", path)

import pandas as pd

df = pd.read_csv("data/train.csv")

df["Gender"] = df["Gender"].map({
    "Male" : 1,
    "Female" : 0
})

print(df.head(10))

print(df.isnull().sum())
print(df.describe())

df = df.drop("CustomerID", axis = 1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(df)

print(x_scaled[:10])

wcss = []

from sklearn.cluster import KMeans

for k in range(1,11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(x_scaled)
    wcss.append(model.inertia_)

import matplotlib.pyplot as plt

# plt.plot(range(1,11), wcss, marker= "o")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("wcss")
# plt.title("Elbow method")
# plt.show()

# graph looks harder to identify 
# printing actual values and observing the difference and behavior
for k, value in zip(range(1, 11), wcss):
    print(k, value)

# optimised k = 4

model = KMeans(n_clusters=4, random_state=42)

model.fit(x_scaled)

clusters = model.predict(x_scaled)

df["clusters"] = clusters

print(df.head())

new_customer = [[25,0, 40, 80]]

new_customer_scaled = scaler.transform(new_customer)

cluster = model.predict(new_customer_scaled)

print(cluster)

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["clusters"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")
plt.show()
