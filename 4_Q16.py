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
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


