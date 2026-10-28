# 16. Decision boundary visualisation for car purchases
# Visualise how different SVM kernels separate customers who purchase a car from those who
# do not.
# Objectives:
# 1. Standardise age and estimated salary.
# 2. Train Linear and RBF Kernel SVM classifiers and plot their decision boundaries.
# 3. Compare their decision regions and test-set performance.

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import alpha
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv("../mock_datasets/social_network_ads.csv")

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
linear = SVC(kernel="linear", random_state=42)
rbf = SVC(kernel="rbf", random_state=42)

linear.fit(X_train, y_train)
rbf.fit(X_train, y_train)

# Compare test accuracy
print("Linear SVM Accuracy:", accuracy_score(y_test, linear.predict(X_test)))
print("RBF SVM Accuracy:", accuracy_score(y_test, rbf.predict(X_test)))

# Convert scaled arrays back to DataFrames for plotting
X_train_plot = pd.DataFrame(
    X_train, columns=["Age", "EstimatedSalary"]
)
X_test_plot = pd.DataFrame(
    X_test, columns=["Age", "EstimatedSalary"]
)

# Plot both decision boundaries
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# # Linear SVM
# DecisionBoundaryDisplay.from_estimator(
#     linear,
#     X_train,
#     alpha=0.4,
#     response_method="predict",
#     cmap="bwr",
#     ax=axes[0]
# )
#
# axes[0].scatter(
#     X_test_plot["Age"],
#     X_test_plot["EstimatedSalary"],
#     c=y_test,
#     edgecolors="k",
#     cmap="bwr"
# )
#
# axes[0].set_xlabel("Standardized Age")
# axes[0].set_ylabel("Standardized Estimated Salary")
# axes[0].set_title("Linear SVM")
#
# # RBF SVM
# DecisionBoundaryDisplay.from_estimator(
#     rbf,
#     X_train,
#     alpha=0.4,
#     response_method="predict",
#     cmap="bwr",
#     ax=axes[1]
# )
#
# axes[1].scatter(
#     X_test_plot["Age"],
#     X_test_plot["EstimatedSalary"],
#     c=y_test,
#     edgecolors="k",
#     cmap="bwr"
# )
#
# axes[1].set_xlabel("Standardized Age")
# axes[1].set_ylabel("Standardized Estimated Salary")
# axes[1].set_title("RBF SVM")
#
# plt.tight_layout()
# plt.show()


fig, axes = plt.subplots(1,2,figsize=(12,5))

#LInear svm
DecisionBoundaryDisplay.from_estimator(
    linear,
    X_train,
    alpha=0.4,
    response_method="predict",
    cmap = "bwr",
    ax = axes[0]
)

axes[0].scatter(
    X_test_plot["Age"],
    X_test_plot["EstimatedSalary"],
    c=y_test,
    edgecolors="k",
    cmap="bwr"
)

axes[0].set_xlabel("Age")
axes[0].set_ylabel("EstimatedSalary")
axes[0].set_title("Linear SVM")

DecisionBoundaryDisplay.from_estimator(
    rbf,
    X_train,
    alpha=0.4,
    response_method="predict",
    cmap="bwr",
    ax=axes[1]
)

axes[1].scatter(
    X_test_plot["Age"],
    X_test_plot["EstimatedSalary"],
    c=y_test,
    edgecolors="k",
    cmap="bwr"
)
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Estimated Salary")
axes[1].set_title("RBF SVM")

plt.tight_layout()
plt.show()


