# Q12. Comparing Linear and RBF Kernel SVM
# Predict whether a user purchases a car using age and estimated salary.
# Objectives:
# 1. Preprocess the dataset and standardise the numerical features.
# 2. Train Linear Kernel and RBF Kernel SVM classifiers using the same train-test split.
# 3. Compare confusion matrices, accuracy, precision, recall, and F1-score and identify
# the better-performing kernel.

# Linear SVM
# Separates the classes using a straight decision boundary.
# It works well when the classes can be separated approximately linearly.

# RBF SVM
# Uses the Radial Basis Function kernel to learn nonlinear decision boundaries when a straight line may not separate the classes well.

# Metric used in SVM Models -> Accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

social_df = pd.read_csv("../mock_datasets/social_network_ads.csv")

X = social_df[["Age", "EstimatedSalary"]]
Y = social_df["Purchased"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#Training Linear SVM Model
linear_model = SVC(kernel="linear")
linear_model.fit(X_train,Y_train)
Y_pred_linear = linear_model.predict(X_test)

#Metric
linear_acc = accuracy_score(Y_test,Y_pred_linear)

# RBF SVM
rbf_model = SVC(kernel="rbf")
rbf_model.fit(X_train, Y_train)
Y_pred_rbf = rbf_model.predict(X_test)

rbf_acc = accuracy_score(Y_test, Y_pred_rbf)

# Compare results
results = pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM"],
    "Accuracy": [linear_acc, rbf_acc]
})

print(results)

# #KEY POINTS
# SVC is scikit-learn's Support Vector Classification model.
# kernel="linear" creates a linear decision boundary.
# kernel="rbf" allows a nonlinear decision boundary.
# Accuracy = number of correct predictions divided by total predictions.