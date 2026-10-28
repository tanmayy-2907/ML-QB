# Q9. Impact of categorical encoding on price prediction
# Investigate how the representation of transmission and owner affects regression performance.
# Objectives:
# 1. Encode categorical features using an appropriate encoding technique.
# 2. Train a regression model using the encoded features alongside numerical attributes and compare them

#Encoding Techniques: 2 Types
# 1. Label Encoding - Assigns an integer to each category
# 2. One Hot Encoding - Creates a separate binary column for each category


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

used_cars_df = pd.read_csv("../mock_datasets/used_cars.csv")

#Since we already did Label Encoding, for this Q, we will do One Hot Encoding
#One Hot Encoding snippet
used_cars_df = pd.get_dummies(
    used_cars_df,
    columns=["Transmission","Owner"],
    dtype=int
)

print(used_cars_df.head()) #Check

results = []
#Model with Encoded Features + Numerical Features
X_encoded = used_cars_df.drop(columns=["Selling_Price"])
# Y as it is
X_train_enc,X_test_enc,Y_train,Y_test = train_test_split(
    X_encoded,Y,
    test_size=0.2,
    random_state=42
)
encoded_model = LinearRegression()
encoded_model.fit(X_train_enc,Y_train)
Y_pred_enc = encoded_model.predict(X_test_enc)

EncRMSE = np.sqrt(mean_squared_error(Y_test,Y_pred_enc))
EncR2 = r2_score(Y_test,Y_pred_enc)

results.append({
    "Model" : "Numerical + Encoded Feature",
    "RMSE" : EncRMSE,
    "R2" : EncR2
})

#Model with only numerical features
X = used_cars_df[["Year", "Kms_Driven"]]
Y = used_cars_df["Selling_Price"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

NumRMSE = np.sqrt(mean_squared_error(Y_test,Y_pred))
NumR2 = r2_score(Y_test,Y_pred)

results.append({
    "Model" : "Numerical Feature Only",
    "RMSE" : NumRMSE,
    "R2" : NumR2
})

print(pd.DataFrame(results).round(4))
