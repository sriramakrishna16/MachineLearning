import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

application = pd.read_csv("data/application_record.csv")
credit = pd.read_csv("data/credit_record.csv")

print("Application shape:", application.shape)
print("Credit shape:", credit.shape)

credit["target"] = credit["STATUS"].apply(
    lambda x: 1 if x in ["1", "2", "3", "4", "5"] else 0
)

target = credit.groupby("ID")["target"].max().reset_index()

print("\nTarget distribution:")
print(target["target"].value_counts())

application = application.drop_duplicates(subset="ID", keep="first")

df = application.merge(target, on="ID", how="inner")

print("\nMerged dataset:")
print(df.shape)

df = df.drop("ID", axis=1)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

X = df.drop("target", axis=1)
y = df["target"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Number of trees:", len(model.estimators_))

print("First 10 tree predictions:")

print("training accuracy: ", model.score(X_train, y_train))
print("testing accuracy : ", model.score(X_test, y_test))

for i, tree in enumerate(model.estimators_[:10]):
    prediction = tree.predict(X_test.iloc[[0]])[0]
    print(f"Tree {i+1} prediction:", prediction)

final_prediction = model.predict(
    X_test.iloc[[0]]
)[0]

print("\nRandom Forest final prediction:")
print(final_prediction)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 important features:")
print(feature_importance.head(15))