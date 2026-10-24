# Q2. Linear versus polynomial regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.linear_model import LinearRegression

# Training a Linear Model on TV, Radio
advertisement_df = pd.read_csv("../mock_datasets/advertising.csv")

X = advertisement_df[["TV", "Radio"]]
Y = advertisement_df["Sales"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

results = pd.DataFrame({
    "Actual Target Values" : Y_test.to_numpy(),
    "Predicted Target Values" : Y_pred,
})
print("\nLinear Regression Model: ")
print(results.round(4))
print("\nRMSE: ", rmse)
print("R2 Score: ", r2)

#POLYNOMIAL REGRESSION MODEL
from sklearn.preprocessing import PolynomialFeatures

#1 Convert Original Features to Polynomial Features (2D)
#TV^2, TV * Radio, Radio^2
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#2 Create & Train Model using Polynomial Features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
Y_pred_poly = poly_model.predict(X_test_poly)

poly_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_poly))
poly_r2 = r2_score(Y_test, Y_pred_poly)

poly_results = pd.DataFrame({
    "Actual Target Values" : Y_test.to_numpy(),
    "Predicted Target Values" : Y_pred_poly,
})
print("\nPolynomial Regression Model (Degree 2): ")
print(poly_results.round(4))
print("\nRMSE: ", poly_rmse)
print("R2 Score: ", poly_r2)

print("\nComparison Results: ")
Comparison = pd.DataFrame({
    "Model" : ["Linear", "Polynomial"],
    "RMSE" : [rmse, poly_rmse],
    "R2 Score" : [r2, poly_r2]
})

print(Comparison.round(4))
