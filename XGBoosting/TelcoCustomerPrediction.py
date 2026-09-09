import kagglehub
import pandas as pd

# path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir = "D:/MachineLearning/XGBoosting/data")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
# print(df.head(20))

print(df.isnull().sum())
print(df.dtypes)
print(df.select_dtypes(include="object").columns)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df.isnull().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop("customerID", axis = 1)
print(df.shape)

df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

print(df["Churn"].value_counts())

df["gender"] = df["gender"].map({
    "Male":1,
    "Female":0
})
df["Partner"] = df["Partner"].map({
    "Yes":1,
    "No":0
})
df["Dependents"] = df["Dependents"].map({
    "Yes":1,
    "No":0
})
df["PhoneService"] = df["PhoneService"].map({
    "Yes":1,
    "No":0
})
df["PaperlessBilling"] = df["PaperlessBilling"].map({
    "Yes":1,
    "No":0
})

print(df["PaymentMethod"].value_counts())
x = df.drop("Churn", axis=1)
y = df["Churn"]

x = pd.get_dummies(x, drop_first= True, dtype = int)
print(x.dtypes)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators = 100,
    learning_rate=0.1,
    max_depth=3,
    gamma=5,
    scale_pos_weight=2.0,
    random_state=42
)
# model has very lower recall because of imbalanced data, so i applied scale_pos_weight
# number of negative samples / no of pos samples -> 4140/1494 = 2.77

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


print(accuracy)
print("confusion matrix")
print(cm)
print("precision : ", precision)
print("recall : ", recall)

print("training accuracy : ", model.score(x_train, y_train))
print("testing accuracy : ", model.score(x_test, y_test))



# beacuse of imbalance data i applied scale_pos_weight so it jumps recall from 54% to nearly 78% and
#  recall drops from 82 ro 79

# output
# 0.794180269694819
# confusion matrix
# [[831 205]
#  [ 85 288]]
# training accuracy :  0.7850550230741924
# testing accuracy :  0.794180269694819
