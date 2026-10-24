# 16. Decision boundary visualisation for car purchases
# Visualise how different SVM kernels separate customers who purchase a car from those who
# do not.
# Objectives:
# 1. Standardise age and estimated salary.
# 2. Train Linear and RBF Kernel SVM classifiers and plot their decision boundaries.
# 3. Compare their decision regions and test-set performance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("mock_datasets/social_network_ads.csv")

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train both models
linear = SVC(kernel="linear")
rbf = SVC(kernel="rbf")

linear.fit(X_train, y_train)
rbf.fit(X_train, y_train)

# Compare test accuracy
print("Linear SVM Accuracy:", accuracy_score(y_test, linear.predict(X_test)))
print("RBF SVM Accuracy:", accuracy_score(y_test, rbf.predict(X_test)))

# Create 100 evenly spaced Age and Salary values
age = np.linspace(X["Age"].min(), X["Age"].max(), 100)
salary = np.linspace(X["EstimatedSalary"].min(), X["EstimatedSalary"].max(), 100)

# Create a grid of all possible Age and Salary combinations
xx, yy = np.meshgrid(age, salary)

# Store grid coordinates as input features for prediction
grid = pd.DataFrame({
    "Age": xx.ravel(),
    "EstimatedSalary": yy.ravel()
})

# Scale grid using the same scaler
grid_scaled = scaler.transform(grid)

# Plot both decision boundaries
plt.figure(figsize=(12, 5))

for i, model in enumerate([linear, rbf], 1):
    Z = model.predict(grid_scaled).reshape(xx.shape)

    # Color the regions according to the model's predictions
    plt.subplot(1, 2, i)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X["Age"], X["EstimatedSalary"], c=y)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.title(["Linear SVM", "RBF SVM"][i-1])

plt.show()

