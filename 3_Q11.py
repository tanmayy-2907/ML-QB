# Q11. Feature importance in used-car price prediction
# Identify which available car attributes contribute most to predicting selling price.
# Objectives:
# 1. Train a Random Forest Regressor using all available features.
# 2. Calculate and rank the feature importance scores.
# 3. Retrain a model using the three most important features and compare its RMSE and
# R2 Score with the full-feature model.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

used_cars_df = pd.read_csv("mock_datasets/used_cars.csv")
results = []

#Encoding
used_cars_df = pd.get_dummies(
    used_cars_df,
    columns=["Transmission","Owner"],
    dtype=int
)

model = RandomForestRegressor(n_estimators=100, random_state=42)

X = used_cars_df.drop(columns="Selling_Price")
Y = used_cars_df["Selling_Price"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(Y_test,Y_pred))
R2 = r2_score(Y_test,Y_pred)

results.append({
    "Model" : "All features",
    "RMSE" : RMSE,
    "R2" : R2
})

# Feature Importance
# After training a Random Forest Regressor, feature_importances_ provides an importance score for each input feature.
# - Higher score → the feature contributed more to the forest's predictive splits.
# - Lower score → the feature contributed less.
# - The scores sum to approximately 1.

#Extracting Important features
impFeatures_df = pd.DataFrame({
    "Feature Name" : X.columns,
    "Importance" : model.feature_importances_
})

impFeatures_df = impFeatures_df.sort_values(
    by="Importance", ascending=False
)
print(impFeatures_df)

#Using Top 3 Important features to train another model
X_top3 = used_cars_df[["Year", "Kms_Driven", "Transmission_Automatic"]]

X_train_top3,X_test_top3,Y_train,Y_test = train_test_split(
    X_top3,Y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train_top3,Y_train)
Y_pred_top3 = model.predict(X_test_top3)
top3RMSE = np.sqrt(mean_squared_error(Y_test,Y_pred))
top3R2 = r2_score(Y_test,Y_pred)

results.append({
    "Model" : "Top 3 Important Features",
    "RMSE" : top3RMSE,
    "R2" : top3R2
})

print(pd.DataFrame(results).round(4))