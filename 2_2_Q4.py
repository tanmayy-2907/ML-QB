# Q4. Multiple Regression Model for all Features, Residual Computation, Plot of Actual vs Predicted Values
# Residual = Actual - Predicted Value
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Training a Linear Model on TV, Radio, Newspaper
advertisement_df = pd.read_csv("mock_datasets/advertising.csv")

X = advertisement_df[["TV", "Radio","Newspaper"]]
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
    "Residual" : Y_test.to_numpy() - Y_pred
})
print(results.round(4))

print("\nRMSE: ", rmse)
print("R2 Score: ", r2)

results["Abs_Residual"] = results["Residual"].abs()

#To sort columns in a DataFrame, use "sort_values"
print("\nTop 5 Highest Absolute Residuals ")
top_5 = results.sort_values(by="Abs_Residual", ascending=False).head(5)
print(top_5)

#Plotting Actual vs Predicted Values

plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Vs Predicted Target Values")
plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    "r--")
plt.show()