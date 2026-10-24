# 18. Comparing SVM with Logistic Regression
# Evaluate whether a nonlinear SVM provides better predictions than Logistic Regression for
# customer purchase behaviour.
# Objectives:
# 1. Train Logistic Regression and RBF Kernel SVM classifiers using standardised
# features.
# 2. Use identical train-test partitions for both models.
# 3. Compare confusion matrices, accuracy, precision, recall, and F1-score and justify the
# preferred model


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

df = pd.read_csv("../mock_datasets/social_network_ads.csv")

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Same train-test split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train both models
models = {
    "Logistic Regression": LogisticRegression(),
    "RBF SVM": SVC(kernel="rbf")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n", name)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))


#Remember the structure of CM in scikit learn is diagonally swapped to what we draw on paper
# TN FP - scikit learn
# FN TP

# TP FN - what we usually write
# FP TN