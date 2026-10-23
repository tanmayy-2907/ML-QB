# Regression is a type of supervised machine learning used to predict a continuous numerical value.
# tries to find the best-fitting straight-line relationship between inputs and the target.
# Linear Regression tries to find the best-fitting straight-line relationship between inputs and the target.
# For a single feature, the equation is: y = mx + c
# For multi features, the equation is : y = b0 + b1x1 + b2x2 + b3x3 + ....
#
# How does it choose the best line?
# It tries to minimize the difference between actual and predicted Target Variable by minimizing the sum of squared errors
# SSE = Summation [i=1 to n] (YiTest - YiPredicted)^2
# Squaring prevents positive and negative errors from cancelling each other and penalizes larger errors more heavily.

#Q. Training a Regression Model for advertising dataset using TV expenditure column only

#1 Loading Dataset
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import pdfRepr
from scipy.linalg import sqrtm

advertisement_df = pd.read_csv("../mock_datasets/advertising.csv")

#2 Importing models and metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#3 Extracting TV feature and Sales Target Col
X = advertisement_df[["TV"]] #DataFrame - 2-D labeled table (Like a Spreadsheet) (Contains Multiple Data Types)
Y = advertisement_df["Sales"] #Series -  1-D labeled array (Like a single column) (Contains Single Data Type)

#4 Split Training & Testing Data
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#5 Create & train model
model = LinearRegression()
model.fit(X_train,Y_train)

#6 Predict Sales
Y_pred = model.predict(X_test)

#7 Display learned Coefficients
print("Slope: ", model.coef_[0]) #Represents Slope (m) (Change in prediction for every 1 unit change in Feature)
print("Intercept: ", model.intercept_) #Represnts Intercept (c) (Value of Predicted variable at input feature = 0)

#8 Evaluate Model using metrics
# SSE (Sum of Squared Errors) — The total of all squared differences between actual and predicted values.
# MSE (Mean Squared Error) — SSE divided by n. Gives the average squared error. Penalizes large errors heavily due to squaring.
# RMSE (Root Mean Squared Error) — Square root of MSE. Same info as MSE but expressed in the original units of your target
# MAE (Mean Absolute Error) — Average of absolute errors. Treats all errors equally, so it's robust to outliers.
# R² (R-squared) — Proportion of variance in the target that the model explains.
# R² => 1 = perfect, 0 = no better than predicting the mean, can go negative if worse than the mean
#Thumb Rule
# MAE / RMSE → "How much error am I making?" (lower = better)
# R² → "How well does the model fit?" (higher = better)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R2 Score: ", r2)

#9 Compare actual & predicted sales value
results = pd.DataFrame({
    "Actual Target Values ": Y_test.to_numpy(),
    "Predicted Target Values ": Y_pred
})
###IMP NOTE:
# train_test_split preserves input types → y_test stays a pandas Series with its original shuffled index (e.g., [3, 7, 12, ...]), while y_pred from .predict() is a plain NumPy array with no index.
# If you put both into a pd.DataFrame({...}) directly, pandas aligns by index label, causing NaN misalignment.
# Always call .to_numpy() on y_test to strip the index so both columns align positionally (row-by-row).

print(results.head(10))