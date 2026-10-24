# Q10. Detecting and handling extreme car prices
# Determine whether extreme selling-price observations influence the performance of a regression model.
# Objectives:
# 1. Identify potential outliers in the selling-price column using the IQR method.
# 2. Train one model on the original training data and another after applying a documented
# outlier-handling method to the training data.
# 3. Compare both models on the same untouched test set using RMSE and R2 Score.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder


used_cars_df = pd.read_csv("../mock_datasets/used_cars.csv")

results = []
#Encoding
le_transmission = LabelEncoder()
le_owner = LabelEncoder()

used_cars_df["Transmission"] = le_transmission.fit_transform(used_cars_df["Transmission"])
used_cars_df["Owner"] = le_transmission.fit_transform(used_cars_df["Owner"])

#MODEL WITHOUT OUTLIER HANDLING
X = used_cars_df[["Year", "Kms_Driven", "Transmission", "Owner"]]
Y = used_cars_df["Selling_Price"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

RMSE = np.sqrt(mean_squared_error(Y_test,Y_pred))
R2 = r2_score(Y_test,Y_pred)

results.append({
    "Model" : "Without Outlier Handling",
    "RMSE" : RMSE,
    "R2" : R2
})

#MODEL WITH OUTLIER HANDLING
# IQR METHOD
# Q1: 25th percentile of the data.
# Q3: 75th percentile of the data.
# IQR: Interquartile Range, the spread of the middle 50% of observations.
# IQR = Q3 - Q1; Lower Bound = Q1 - 1.5 * IQR ; Upper Bound = Q3 + 1.5 * IQR
# Any value below LB or above UB is an outlier


#Outlier Handling performed on numerical columns
df_clean = used_cars_df.copy()

cols = ["Year","Kms_Driven"]

for col in cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1

    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR

    df_clean = df_clean[
        (df_clean[col] >= LB) &
        (df_clean[col] <= UB)
    ]
# Features and target
X = df_clean[["Year", "Kms_Driven","Transmission", "Owner"]]
y = df_clean["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

results.append({
    "Model" : "With Outlier Handling",
    "RMSE" : rmse,
    "R2" : r2
})

print(pd.DataFrame(results).round(4))
