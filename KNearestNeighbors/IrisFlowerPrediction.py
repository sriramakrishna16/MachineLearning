from sklearn.datasets import load_iris
data = load_iris()

x = data.data
y = data.target

print(x[:5])
print(y[:5])

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 32)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
model = Pipeline([
    ("scale", StandardScaler()),
    ("knf", KNeighborsClassifier(n_neighbors = 5,
                                 weights = "distance",
                                 metric = "minkowski",
                                 p = 5))  #minkowski is distance method where p = 1 is manhattan and 2 is euclidean
                                #  we can directly take metric = "euclidean"
])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
cn = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("accuracy :" , accuracy)
print("confusion matrix \n" , cn)
print("classification report \n", cr)

print("training accurcay :", model.score(x_train, y_train))
print("testing accurcay :", model.score(x_test, y_test))

# for k in [1, 3, 5, 7, 9, 11, 13, 15]:
#     model.set_params(knn__n_neighbors=k)
#     model.fit(x_train, y_train)
#     print("K:", k)
#     print("Training accuracy:", model.score(x_train, y_train))
#     print("Testing accuracy:", model.score(x_test, y_test))
#     print()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    x,
    y,
    cv=5
)

print(scores)
print("Average accuracy:", scores.mean())


# Observations
""" 
when k = 1 and random_state = 42 -> accuracy is 1 training accuracy - 1 , testing accuracy - 1
when k = 3 and random_state = 42 -> accuracy is 1 but trian -> 0.94 and test -> 1.0
                so for k = 3, there will be some misclassification for end points, and testing data is very close
                to class thats why 100% testing accuracy

                now changing random state

when k = 1 and random_state = 32 -> accuracy is 0.96 and train -> 1.0 and test -> 0.96
                may be the close samples are shuffled to train data so result in 100%

so i have done cross validation with 5 folds and k = 1 and k = 3 and k = 5
    and metric = euclidean(better than 1 and 3) and weights = distance(better than uniform)
    so average accuracy = 96% whith k = 5 (best)
    at k = 3 , 95% and at k = 1 , 94%
                
"""