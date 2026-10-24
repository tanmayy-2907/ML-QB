#Q7. Effect of vehicle age and kilometres driven; Derive age column, use kms_driven alone and along with age to train and compare 2 Regression models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression


used_cars_df = pd.read_csv("mock_datasets/used_cars.csv")

#Deriving Age Attribute
used_cars_df["Age"] = 2026 - used_cars_df["Year"]

print(used_cars_df["Age"].head())

X = used_cars_df[["Kms_Driven","Age"]]
Y = used_cars_df["Selling_Price"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

features = [
    ["Kms_Driven"],
    ["Kms_Driven", "Age"]
]
results = []
for feature in features:
    model = LinearRegression()
    model.fit(X_train[feature], Y_train)
    Y_pred = model.predict(X_test[feature])
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)

    results.append({
        "Feature" : " + ".join(feature),
        "RMSE" : rmse,
        "R2" : r2

    })

print(pd.DataFrame(results).round(4))