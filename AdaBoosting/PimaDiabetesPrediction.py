import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# this dataset has no headers , so we manually create the columns titles
# we are training features to model for predicting the patient has diabetes or not.

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names = columns)

print("========" , df["Outcome"].value_counts())

print(df.head(10))
print(df.shape)

x = df.drop("Outcome", axis = 1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)


x_train, x_test,y_train, y_test = train_test_split(x, y,test_size = 0.2, random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


weak_learner = DecisionTreeClassifier(max_depth =1, random_state= 42)
model = AdaBoostClassifier(
    estimator = weak_learner,
    n_estimators=100,
    learning_rate = 0.1,
    random_state=42
    )

model.fit(x_train, y_train)

# model.fit(x_train_smote, y_train_smote) #applying smothe

y_pred = model.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(cr)
print(cm)
print("accuracy :" , accuracy)

print("training accuracy : " , model.score(x_train, y_train))
print("testing accuracy : " , model.score(x_test, y_test))

# when i use max_depth = 2 or 3 in weak leaner , its getting better recall and precision on 
# class 1 data, but slightly increasing overfitting

# output
""" [[87 12]
 [22 33]]
accuracy : 0.7792207792207793
training accuracy :  0.7785016286644951
testing accuracy :  0.7792207792207793 """

# applying smothe increases the class 1 recall

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

auc = roc_auc_score(y_test, y_pred)

print("AOC", auc)

