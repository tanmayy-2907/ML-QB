# 14. SVM hyperparameter optimisation
# Improve the performance of an SVM classifier by tuning its hyperparameters.
# Objectives:
# 1. Standardise age and estimated salary.
# 2. Apply GridSearchCV to tune C and gamma for an RBF Kernel SVM.
# 3. Compare the default and tuned models using cross-validation scores, test accuracy,
# and F1-score.


# The main parameters to tune are:
# - C — Controls the penalty for misclassification. Higher C tries harder to classify training points correctly.
# - kernel — Determines the type of decision boundary: linear or rbf.
# - gamma — Controls how far the influence of each training point reaches. 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv("mock_datasets/social_network_ads.csv")

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#untuned SVM
untuned_model = Pipeline([
    ("scalar",StandardScaler()),
    ("svm", SVC())
])

untuned_model.fit(X_train, y_train)
untuned_pred = untuned_model.predict(X_test)

untuned_accuracy = accuracy_score(y_test, untuned_pred)

print("Untuned SVM Accuracy: ", untuned_accuracy)
#Tuned SVM using GridSearchCV

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", 0.1, 1]
}
tuned_model = Pipeline([
    ("scalar", StandardScaler()),
    ("svm", SVC())
])
gridsearch = GridSearchCV(
    estimator=tuned_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

gridsearch.fit(X_train,y_train)
Y_pred_tuned = gridsearch.predict(X_test)
tuned_acc = accuracy_score(y_test, Y_pred_tuned)

print("\nBest Parameters:", gridsearch.best_params_)
print("Best CV Accuracy:", gridsearch.best_score_)
print("Tuned SVM Test Accuracy:", tuned_acc)