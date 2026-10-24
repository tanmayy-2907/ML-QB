# Q19. Analysing Decision Tree depth and overfitting
# Examine the effect of tree complexity on the classification of Iris species.
# Objectives:
# 1. Train Decision Tree classifiers with maximum depths of 2, 3, 4, and 5.
# 2. Calculate training and testing accuracy for each model.
# 3. Plot the results and identify the depth that offers a suitable balance between
# underfitting and overfitting.

import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler #optional here
from sklearn.datasets import load_iris #IRIS DATASET

# What is a Decision Tree?
# # A Decision Tree is a machine learning algorithm that makes predictions by asking a sequence of questions about the input features.
# For the Iris dataset, it uses flower measurements to classify a flower into one of three species:
# - Setosa
# - Versicolor
# - Virginica

#Loading Iris Dataset
iris_df = load_iris()
X = iris_df.data
Y = iris_df.target

#Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

depths = [2, 3, 4, 5]
train_acc = [] #Training accuracies for all depths
test_acc = [] #Testing accuracies for all depths

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    model.fit(xtrain,ytrain)

    y_train_pred = model.predict(xtrain) #predictions on training set
    y_test_pred = model.predict(xtest)  #predictions on testing set

    train_acc.append(accuracy_score(ytrain,y_train_pred)) #computing train acc
    test_acc.append(accuracy_score(ytest,y_test_pred)) #computing test acc

#Display acc
for d in range(len(depths)): #Accessing depth values in list by index
    print(
        "Depth: ", depths[d],
        "Training Accuracy", round(train_acc[d],4),
        "Testing Accuracy", round(test_acc[d],4)
    )

#Plotting
plt.plot(depths, train_acc, marker="o", label="Training Accuracy") #Draws a line graph of training accuracy against tree depth.
# Xaxis = depths, Yaxis = train_acc, marker = displays circle at each pt

plt.plot(depths, test_acc, marker="o", label="Testing Accuracy")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy Score")
plt.title("Accuracy against Depth of Tree")
plt.xticks(depths) #plt.xticks(depths) displays the tested depths on the x-axis.
plt.legend()
plt.show()