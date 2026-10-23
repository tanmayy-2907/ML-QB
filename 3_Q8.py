#Q8. Hyperparameter tuning of a Decision Tree Regressor
# Improve used-car price prediction by identifying suitable Decision Tree parameters.

# Hyperparameters are settings you choose before training the model.
#For a Decision Tree Regressor, common hyperparameters include:
# - max_depth: Maximum depth of the tree.
# - min_samples_split: Minimum number of samples required to split a node.
# - min_samples_leaf: Minimum number of samples required in a leaf node.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor #Decision Tree Model
from sklearn.preprocessing import LabelEncoder

used_cars_df = pd.read_csv("mock_datasets/used_cars.csv")

# Encoding Categorical columns using Label Encoding method
le_transmission = LabelEncoder()
le_owner = LabelEncoder()

used_cars_df["Transmission"] = le_transmission.fit_transform(used_cars_df["Transmission"])
used_cars_df["Owner"] = le_transmission.fit_transform(used_cars_df["Owner"])


#Split
X = used_cars_df[["Year","Kms_Driven","Transmission","Owner"]]
Y = used_cars_df["Selling_Price"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)
#Training Model and Creating a Hyperparameters Grid to tune the model
model = DecisionTreeRegressor(random_state=42)

param_grid = {
    "max_depth" : [3,5,10, None],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator= model,
    param_grid=param_grid,
    cv= 5, #Cross fold validations = 5
    scoring="neg_mean_squared_error", #Since we need model with Lowest MSE
    n_jobs=-1 #No of cores , -1 indicates use all cores for parallel processing;Optional
)

grid_search.fit(X_train,Y_train)
print("Best Parameters: ", grid_search.best_params_) #Best parameter combination from param grid
print("Best CV RMSE: ", np.sqrt(-grid_search.best_score_)) #Lowest avg RMSE Score during cross fold validation
# - sign to make the value positive since its already negative

best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
r2 = r2_score(Y_test, Y_pred)

print("Tuned Test set RMSE: ", rmse)
print("Tuned R2 Score: ", r2)


#BASE MODEL WITHOUT TUNING HYPERPARAMETERS
base_model = DecisionTreeRegressor(random_state=42)
base_model.fit(X_train, Y_train)
Base_Y_pred = base_model.predict(X_test)

base_rmse = np.sqrt(mean_squared_error(Y_test,Base_Y_pred))
base_r2 = r2_score(Y_test,Base_Y_pred)

print("BASE RMSE: ", base_rmse)
print("BASE R2: ", base_r2)


# GridSearchCV: Tests combinations of hyperparameters using cross-validation to find the best combination.
# scoring="neg_mean_squared_error": Built-in scikit-learn scoring string; negative MSE is used because GridSearchCV maximizes scores.
# n_jobs=-1: Uses all available CPU cores for parallel processing; optional, default is None.
# best_params_: Returns the best hyperparameter combination.
# best_score_: Returns the best average cross-validation score;
# best_estimator_: Returns the model configured with the best hyperparameters.
# Important: Fit GridSearchCV only on training data, then evaluate the selected model on the separate test set.