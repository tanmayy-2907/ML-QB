# Q20. Comparing impurity criteria in Decision Trees
# Investigate how the splitting criterion influences Iris classification.
# Objectives:
# 1. Train Decision Tree classifiers using Gini and Entropy criteria with the same depth
# and split settings.
# 2. Compare their confusion matrices and classification accuracy.
# 3. Visualise the trees and discuss differences in their splitting decisions.

import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris #IRIS DATASET

# What is Impurity ?
# A Decision Tree makes decisions by splitting data into groups.
# Impurity measures how mixed the classes are within a group.
# A Decision Tree tries to select splits that reduce impurity.

# Gini vs Entropy
# 1. Gini impurity
# Measures the probability of incorrectly classifying a randomly selected sample if its class were assigned according to the class distribution in the group.
# In scikit-learn, use criterion="gini".
# 2. Entropy
# Measures the uncertainty in the class distribution within a group.
# In scikit-learn, use criterion="entropy".

iris_df = load_iris()
X = iris_df.data
Y = iris_df.target

#Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#Training gini and entropy models
gini_model = DecisionTreeClassifier(criterion="gini",max_depth=3,random_state=42)
entropy_model = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)

gini_model.fit(xtrain,ytrain)
entropy_model.fit(xtrain,ytrain)

#predictions
gini_pred = gini_model.predict(xtest)
entropy_pred = entropy_model.predict(xtest)

#Metrics
print("\nGini Model Accuracy: ",accuracy_score(ytest, gini_pred))
print("Gini Model Confusion Matrix: ", confusion_matrix(ytest, gini_pred))

print("\nEntropy Model Accuracy: ",accuracy_score(ytest, entropy_pred))
print("Entropy Model Confusion Matrix: ", confusion_matrix(ytest, entropy_pred))

#Plot trees
plt.figure(figsize=(12,5)) #Sets overall figure size

plt.subplot(1,2,1) #Creates first plot out of 2.
plot_tree(gini_pred, filled=True) #Draw gini tree with colored nodes
plt.title("Gini")

plt.subplot(1,2,1) #Creates second plot out of 2.
plot_tree(entropy_pred, filled=True) #Draw entropy tree with colored nodes
plt.title("Entropy")

plt.show()
