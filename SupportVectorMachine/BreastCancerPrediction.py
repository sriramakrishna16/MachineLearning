from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel = "rbf", C=1, gamma = 0.004)) #c is penality
])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# for c = 1 , accuracy = 95%
# for c = 0.1 , accurcay = 98% (best)
# for c = 0.5 , accuracy = 96%
# for c = 0.01 , accuracy = 97%

# for kernel = rbf  and c = 1 = 98% (best)
# for kernel = poly and c = 10 = 95% (best)