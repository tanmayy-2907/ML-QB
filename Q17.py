# 17. Analysing classification errors
# Investigate which customers are incorrectly classified by an SVM model.
# Objectives:
# 1. Train an RBF Kernel SVM using the available customer features.
# 2. Identify false positives and false negatives from the test-set predictions.
# 3. Calculate precision and recall, and discuss the practical implications of the two types of errors.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("mock_datasets/social_network_ads.csv")

X = df[["Age", "EstimatedSalary"]]
Y = df["Purchased"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#Scaling
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

model = SVC(kernel="rbf")
model.fit(X_train_scaled,Y_train)
Y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(Y_test, Y_pred)

print("\nConfusion Matrix: \n", cm)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()


#KPR
# - Confusion matrix: Evaluates correct and incorrect classification predictions.
# - TP (True Positive): Actual 1, predicted 1.
# - TN (True Negative): Actual 0, predicted 0.
# - FP (False Positive): Actual 0, predicted 1.
# - FN (False Negative): Actual 1, predicted 0.
# - Scikit-learn's default matrix layout:
# TN & FP
# FN & TP
# - Rows represent actual classes; columns represent predicted classes.
# - confusion_matrix(y_test, y_pred) calculates the matrix.
# - ConfusionMatrixDisplay(confusion_matrix=cm).plot() visualizes it.
# - plt.show() displays the plot.