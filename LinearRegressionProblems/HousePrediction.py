import kagglehub
import pandas as pd

# path = kagglehub.dataset_download("yasserh/housing-prices-dataset", output_dir = "D:/MachineLearning/LinearRegressionProblems/data")
# print(path)

df = pd.read_csv("data/Housing.csv")
# print(df)

# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.isnull().sum())

x = df.drop("price", axis=1)
y = df["price"]

# print(x.select_dtypes(include="str").columns)

# for col in x.select_dtypes(include="str").columns:
#     print(col, x[col].unique())

binary_cols = ["mainroad","guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
x[binary_cols] = x[binary_cols].replace({
    "yes":1,
    "no":0
})

# print(x.head())
# print(x.dtypes)
# print(x)

x = pd.get_dummies(x, columns=["furnishingstatus"], drop_first=True,dtype=int)

from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(x_test)

# print([f"{x:.2f}" for x in y_pred[:10]])

# measuring how good it performs
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) #how close model predictions to actual values

# print("MAE", mae)
# print("r2", r2)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2)
})
# print(results.head(10))

coefficients = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_
})

print(coefficients.round(2))
print("Intercept:", model.intercept_)