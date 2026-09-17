from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target

print(x.shape)
print(y.shape)

# print(x[:5])
# print(y[:5])

# our goal is to make 30 features into 2 principal components

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

x_scaled = scalar.fit_transform(x)

import numpy as np

cov_matrix = np.cov(x_scaled, rowvar=False)
print(cov_matrix.shape)

# calculating eigen values and vectors

eigenval, eigenvec = np.linalg.eigh(cov_matrix)

idx = np.argsort(eigenval)[::-1]


eigenval = eigenval[idx]
eigenvec = eigenvec[:, idx]

# calculating the information in each pc it has

explained_variance = eigenval/np.sum(eigenval)
print(explained_variance)

w = eigenvec[:, :2]
print(w.shape)

x_pca = x_scaled @ w

print(x_pca.shape)

# we dont need to calculate theese manually , we can do it using PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
xx_pca = pca.fit_transform(x_scaled)
print(xx_pca.shape) 
print(pca.explained_variance_ratio_)

# import matplotlib.pyplot as plt

# plt.scatter(
#     x_pca[:, 0],
#     x_pca[:, 1],
#     c=y
# )

# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA - Breast Cancer Dataset")
# plt.show()

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state = 42, stratify=y)

svm_without_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1))
])

svm_without_pca.fit(x_train, y_train)

y_pred = svm_without_pca.predict(x_test)

print("SVM WITHOUT PCA")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

# svm with pca

svm_with_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("svm", SVC(kernel="rbf", C=1))
])

svm_with_pca.fit(x_train, y_train)

y_pred_with_pca = svm_with_pca.predict(x_test)


print("SVM WITH PCA")

print("Accuracy:", accuracy_score(y_test, y_pred_with_pca))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_with_pca))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_with_pca,
    target_names=data.target_names
))

# with pca , svm accuracy is 97.37%
# without pca , svm accuracy is 98.25%

# nearly 0.88% loss while using pca with reduced dimenstions (10 pca components)

pca = svm_with_pca.named_steps["pca"]

print("PCA INFORMATION")

print("Original number of features:", x.shape[1])

print("Number of PCA components:", pca.n_components_)

print("Explained variance:", pca.explained_variance_ratio_)

print("Total explained variance:", pca.explained_variance_ratio_.sum())