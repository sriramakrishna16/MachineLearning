from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame = True)

df = data.frame

print(df.head(10))

x = df.drop("MedHouseVal", axis =1)
y = df["MedHouseVal"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x , y , test_size=0.2, random_state=42
)

import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators = 200,
    learning_rate = 0.1,
    max_depth = -1,
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("mae : ", mae)
print("mse : ", mse)
print(rmse)
print("r2 score : ",r2)

print(y_test[:10].values)
print(y_pred[:10])

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("Training RMSE:", train_rmse)
print("Testing RMSE :", test_rmse)

print("Training R2:", train_r2)
print("Testing R2 :", test_r2)
