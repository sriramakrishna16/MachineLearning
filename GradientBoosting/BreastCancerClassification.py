from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()

# print(data)
print(data.data.shape)
print(data.target.shape)

df = pd.DataFrame(data.data, columns = data.feature_names)
df["target"] = data.target

# print(df.head(10))

x = df.drop("target", axis = 1)
y = df["target"]

print(x.shape)
print(y.shape)

# the input features has several measurements describing the shape and characateristics of the tumor
# and the target column is the 0 and 1 classification where 0 is malignant and 1 is benign 
# malignant mean cancerous tumor and benign is non cancerous tumor
# and the dataset is preprocessed by sklearn

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state=42
)

print("===========")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate = 0.01,
    max_depth = 3,
    random_state = 42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("predicted :" , y_pred)
print("actual    :", y_test.values)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_pred, y_test)
cm = confusion_matrix(y_pred, y_test)
print("accuracy")
print(accuracy)
print("confusion matrix")
print(cm)

print("training accuracy: ", model.score(x_train, y_train))
print("testing accuracy : ", model.score(x_test, y_test))



# this is the result , when n_estimators = 100 , 100 sequential decision trees

""" accuracy
0.956140350877193
confusion matrix
[[40  2]
 [ 3 69]]
training accuracy:  1.0(100)
testing accuracy :  0.956140350877193(95.6) """

# when n_estimators = 10

""" accuracy
0.956140350877193
confusion matrix
[[40  2]
 [ 3 69]]
training accuracy:  0.9846153846153847
testing accuracy :  0.956140350877193 """


# observation
""" More trees can make the model increasingly good at fitting the training data,
without improving its ability to generalize. """