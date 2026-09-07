import kagglehub
import pandas as pd

# path = kagglehub.dataset_download("rikdifos/credit-card-approval-prediction", output_dir="D:/MachineLearning/DecisionTreesProblems/data")
# print("Path to dataset files:", path)

df1 = pd.read_csv("data/application_record.csv")
df2 = pd.read_csv("data/credit_record.csv")

print(df1.shape)
print(df2.shape)

#checking null vales
# print(df1.isnull().sum())
# print(df2.isnull().sum())

#Checking duplicate values
# print(df1.duplicated().sum())
# print(df2.duplicated().sum())

#remove duplicate applicants
df1 = df1.drop_duplicates(subset="ID", keep="first")

#there is more than one lakh missing values in occupation type
df1["OCCUPATION_TYPE"] = df1["OCCUPATION_TYPE"].fillna("unknown")

print(df1["ID"].duplicated().sum())
print(df2["STATUS"].value_counts())

#creating TARGET
df2["TARGET"] = df2["STATUS"].apply(lambda x: 1 if x in ["1","2","3","4","5"] else 0)

#grouping credit records by ID
credit_target = df2.groupby("ID")["TARGET"].max().reset_index()

#merging credit_target values with the application details
df = df1.merge(credit_target, on="ID", how="inner")

print(df.shape)
print(df["TARGET"].value_counts())

df = df.drop("ID", axis=1)

#feature engineering
df["AGE"] = df["DAYS_BIRTH"].abs() / 365
df["EMPLOYED"] = (df["DAYS_EMPLOYED"] != 365243).astype(int)
df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, 0).abs() / 365
df = df.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1)

#mapping binary categorical columns
df["CODE_GENDER"] = df["CODE_GENDER"].map({"M":0, "F":1})
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map({"Y":0, "N":1})
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"Y":0, "N":1})

#checking for NaN values after mapping
print(df[["CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY"]].isnull().sum())

#creating X and y
x = df.drop("TARGET", axis=1)
y = df["TARGET"]

#splitting data before encoding
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import OneHotEncoder

#categorical columns
categorical_cols = ["NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE"]

#OneHot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_train = encoder.fit_transform(x_train[categorical_cols])
encoded_test = encoder.transform(x_test[categorical_cols])

#creating encoded DataFrames
encoded_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=x_train.index)
encoded_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=x_test.index)

#dropping original categorical columns
x_train = x_train.drop(columns=categorical_cols)
x_test = x_test.drop(columns=categorical_cols)

#adding encoded columns
x_train = pd.concat([x_train, encoded_train], axis=1)
x_test = pd.concat([x_test, encoded_test], axis=1)

print(x_train.shape)
print(x_test.shape)

#checking remaining NaN values
print("Train NaN:", x_train.isnull().sum().sum())
print("Test NaN:", x_test.isnull().sum().sum())

#training Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
model.fit(x_train, y_train)

#making predictions
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

#evaluating the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Training Accuracy:", model.score(x_train, y_train))
print("Testing Accuracy:", model.score(x_test, y_test))