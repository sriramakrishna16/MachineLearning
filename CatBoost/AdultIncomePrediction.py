from catboost.datasets import adult

# train, test = adult()

# train.to_csv("data/adult_train.csv", index = True)
# test.to_csv("data/adust_test.csv", index = True)

import pandas as pd

train = pd.read_csv("data/adult_train.csv")
test = pd.read_csv("data/adult_test.csv")

# print(train.head())
# print("======")
# print(test.head())

# i have used index = true thats why i have extra column 

train.drop("Unnamed: 0", axis = 1, inplace = True)
test.drop("Unnamed: 0", axis = 1, inplace = True)

print(train.shape)
print(test.shape)

# catboot internally splits train and test 66:33 ratio
# if we wanted to do it manually then 

data = pd.concat([train, test], ignore_index = True)

print(data.shape)

x = train.drop("income", axis = True)
y = train["income"]

from sklearn.model_selection import train_test_split

x_train = x
x_test = test.drop("income", axis = True)
y_train = y
y_test = test["income"]

print(x_train.dtypes)

cat_features = ["workclass", "education", "marital-status",
                "occupation", "relationship", "race", "sex", "native-country"]

x_train[cat_features] = x_train[cat_features].fillna("Missing")
x_test[cat_features] = x_test[cat_features].fillna("Missing")

# print(x_train[cat_features].isnull().sum())
# print(x_test[cat_features].isnull().sum())

from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations= 500,
    learning_rate = 0.05,
    depth = 6,
    loss_function = "Logloss",
    eval_metric = "Accuracy",
    random_seed = 42,
    verbose = 100
)

model.fit(x_train, y_train, cat_features = cat_features,
          eval_set = (x_test, y_test),
          early_stopping_rounds = 50,
          verbose = 100
          )

y_pred = model.predict(x_test)

print(y_pred[:10])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


print("accurcay score :", accuracy)
print("confusion_matrix")
print(cm)
print("classification report")
print(cr)

print("trainig accuracy : ", model.score(x_train, y_train))
print("testing accuracy : ", model.score(x_test, y_test))

# print(model.get_feature_importance())

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# observations

""" depth = 6, learning_rate = 0.05 → Accuracy = 87.49 - best
depth = 4, learning_rate = 0.05 → Accuracy = 87.36
depth = 8, learning_rate = 0.05 → Accuracy = 87.33

depth = 6, learning_rate = 0.10 → Accuracy = 87.48
depth = 6, learning_rate = 0.01, itergation = 1500 → Accuracy = 86.84 """

from catboost import Pool

text_pool = Pool(x_test, cat_features= cat_features)

shap_values = model.get_feature_importance(
    type="ShapValues",
    data=text_pool
)

print(shap_values.shape)
print(shap_values[0])
print(x_test.iloc[0])
print(model.predict(x_test.iloc[[0]]))
print(model.predict_proba(x_test.iloc[[0]]))