# 13. Effect of feature scaling on SVM classification
# Investigate whether feature scaling affects the prediction of car purchases.
# Objectives:
# 1. Train an RBF Kernel SVM using unscaled age and estimated salary.
# 2. Train another RBF Kernel SVM using standardised features.
# 3. Compare their confusion matrices, accuracy, and F1-score and explain the observed
# differences.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

social_df = pd.read_csv("mock_datasets/social_network_ads.csv")

X = social_df[["Age", "EstimatedSalary"]]
Y = social_df["Purchased"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)
# RBF SVM using unscaled features
rbf_model = SVC(kernel="rbf")
rbf_model.fit(X_train, Y_train)
Y_pred_rbf = rbf_model.predict(X_test)
rbf_acc = accuracy_score(Y_test, Y_pred_rbf)

# Why do we need feature scaling?
# Your dataset has two features with different numerical ranges:
# - Age: typically ranges from tens to around 60.
# - EstimatedSalary: can range into tens or hundreds of thousands.
# SVM uses distances and margins(distance-sensitive algo), so the larger numerical range of salary can disproportionately influence the decision boundary.

# StandardScaler transforms each feature so it has approximately a mean of 0 and standard deviation of 1.
# z = (x - myu) / sigma; x - original value, myu - training-set mean, sigma - training-set SD

#RBF SVM using Scaled features
scale = StandardScaler()

X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

scaled_model = SVC(kernel="rbf")
scaled_model.fit(X_train_scaled,Y_train)
Y_pred_scaled = scaled_model.predict(X_test_scaled)
scaled_acc = accuracy_score(Y_test,Y_pred_scaled)

results = pd.DataFrame({
    "Model" : ["Unscaled", "Scaled"],
    "Accuracy" : [rbf_acc,scaled_acc]
})

print(results.round(4))

# KPR
# StandardScaler() creates the scaler.
# Never fit the scaler on the test data; that causes data leakage.
