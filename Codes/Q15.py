# 15. Analysing the influence of age and salary
# Determine whether age or estimated salary provides more useful information for predicting
# car purchases.
# Objectives:
# 1. Train separate SVM classifiers using age alone, salary alone, and both features.
# 2. Use the same kernel, data split, and preprocessing approach for fair comparison.
# 3. Compare precision, recall, F1-score, and accuracy and identify the most effective
# feature combination.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


df = pd.read_csv("../mock_datasets/social_network_ads.csv")

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scaling
scale = StandardScaler()
X_train_scaled = pd.DataFrame(scale.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scale.transform(X_test), columns=X_test.columns)
#Since fit_transform returns a numpy array and we need Pandas DataFrame ahead for easier looping logic

#All combinations for training models
combinations = [
    ["Age"],
    ["EstimatedSalary"],
    ["Age", "EstimatedSalary"]
]
results = []
#Model training
for c in combinations:
    model = SVC(kernel="linear")

    model.fit(X_train_scaled[c],y_train)
    Y_pred = model.predict(X_test_scaled[c]) # Line 30 Comment

    model_acc = accuracy_score(y_test,Y_pred)
    model_precision = precision_score(y_test,Y_pred)
    model_recall = recall_score(y_test, Y_pred)
    model_F1 =f1_score(y_test, Y_pred)

    results.append({
        "Features " : " + ".join(c),
        "Accuracy" : model_acc,
        "Precision" : model_precision,
        "Recall" : model_recall,
        "F1 Score" : model_F1
    })

print(pd.DataFrame(results).round(4))

