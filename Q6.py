#Q6. Linear Regression, Decision Tree Regression & Random Forest Regression on used cars dataset

# Decision Tree - Makes predictions by splitting data into groups using feature-based conditions. For example, it might split cars by year or kilometres driven.
# Random Forest - ensemble technique; Combines predictions from many decision trees & avg their prediction for Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor #Decision Tree Model
from sklearn.ensemble import RandomForestRegressor #Random Forest Model
from sklearn.preprocessing import LabelEncoder

used_cars_df = pd.read_csv("mock_datasets/used_cars.csv")

# Transmission and Owner are text features.
# We must convert them into numeric values before using these Regression models.

# Encoding Categorical columns using Label Encoding method
le_transmission = LabelEncoder()
le_owner = LabelEncoder()

#fit_transform helps in learning the structure of data and then applying transformation on that data
used_cars_df["Transmission"] = le_transmission.fit_transform(used_cars_df["Transmission"])
used_cars_df["Owner"] = le_owner.fit_transform(used_cars_df["Owner"])

#Check encoded cols
print(used_cars_df.head(5)) #for Transmission: Automatic - 0, Manual - 1 ; for Owner: First - 0, Second - 1, Third - 2

X = used_cars_df[["Year", "Kms_Driven", "Transmission", "Owner"]]
Y = used_cars_df["Selling_Price"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#Training the three models
models = {
    "Linear Regression" : LinearRegression(),
    "Decision Tree" : DecisionTreeRegressor(random_state=42),
    "Random Forest" : RandomForestRegressor(n_estimators=100, random_state=42) #n_estimators here is basically no of decision trees
}

results = []
for name, model in models.items():

    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)

    results.append({
        "Model" : name,
        "RMSE" : rmse,
        "R2" : r2
    })

print(pd.DataFrame(results).round(4))