# Q3. Evaluating the contribution of Newspaper advertising (Adjusted R2 Score)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.linear_model import LinearRegression

# Training a Linear Model on TV, Radio, Newspaper
advertisement_df = pd.read_csv("../mock_datasets/advertising.csv")

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


print("\nRMSE: ", rmse)
print("R2 Score: ", r2)

# Training a Linear Model on TV &  Radio

X_red = advertisement_df[["TV", "Radio"]]
Y_red = advertisement_df["Sales"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X_red,Y_red,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred_red = model.predict(X_test)

rmse_red = np.sqrt(mean_squared_error(Y_test, Y_pred_red))
r2_red = r2_score(Y_test, Y_pred_red)


print("\nReduced RMSE: ", rmse_red)
print("Reduced R2 Score: ", r2_red)

n = len(Y_test)

print("\nAdjusted R2 Score")
#FORMULA
#ADJUSTED R2 Score = 1 - (1 - R2 Score) * (No.of Obs (n) - 1)/ (n - No of features(p) - 1)
adj_r2_A = 1 - (1 - r2) * (n - 1) / (n - 3 - 1)
adj_r2_B = 1 - (1 - r2_red) * (n - 1) / (n - 2 - 1)
print("\nModel A: ", adj_r2_A)
print("Model B: ", adj_r2_B)