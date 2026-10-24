# Day 1 Notes — Python, Pandas & Data Preprocessing

## 1. DataFrame Basics

- A DataFrame is a two-dimensional tabular data structure containing rows and columns.

## 2. Dataset Inspection

- `df.shape` returns a tuple containing `(number_of_rows, number_of_columns)`.
- `df.shape[0]` returns the number of rows.
- `df.shape[1]` returns the number of columns.
- `df.columns` returns the column names as a Pandas Index object.
- `list(df.columns)` converts the column names into a Python list.
- `df.info()` displays column names, data types, non-null counts, and memory usage. It prints the information directly and returns `None`.
- `df.describe()` displays descriptive statistics such as count, mean, standard deviation, minimum, maximum, and quartiles for numerical columns.

## 3. Missing Values

- `df.isnull()` checks each cell for missing values and returns `True` for missing entries and `False` otherwise.
- `df.isnull().sum()` counts missing values in each column.
- `df.count()` returns the number of non-null values in each column; it does not directly return the total number of rows.
- Use `df.shape[0]` or `len(df)` to count the total number of rows.

## 4. Selecting Columns

- `df["TV"]` selects a single column and returns a Series.
- `df[["TV", "Sales"]]` selects multiple columns and returns a DataFrame.
- `pd.read_csv("file.csv", usecols=["TV", "Sales"])` loads only the specified columns from a CSV file.
- `df.iloc[:, [0, 1]]` selects columns by integer position. The colon `:` selects all rows.
- Prefer selecting columns by name when possible because column positions may change.

## 5. Unique Values and Value Counts

- `df["Owner"].unique()` returns the distinct values in a column.
- `df["Purchased"].value_counts()` counts the occurrences of each category.
- `df["Purchased"].value_counts(normalize=True)` returns the proportion of each category instead of its raw count.
- Multiplying normalized counts by `100` converts proportions into percentages.
- `percent.iloc[0]` accesses the first element by position, regardless of its index label.
- `.loc[]` is used for label-based access, while `.iloc[]` is used for integer-position-based access.

## 6. Filtering Rows Using Conditions

- `df[df["Year"] >= 2020]` selects rows where the year is 2020 or later.
- `df[df["Kms_Driven"] > 50000]` selects rows where kilometres driven exceed 50,000.
- Use `&` to combine conditions with logical AND.
- Use `|` to combine conditions with logical OR.
- Put each condition in parentheses when combining multiple conditions.

Example:

```python
filtered_df = df[
    (df["Year"] >= 2020) &
    (df["Transmission"] == "Automatic")
]
```

## 7. Features and Target

- Features (`X`) are the input variables used by a machine learning model to make predictions.
- Target (`y`) is the variable the model is trained to predict.

## 8. Training and Testing Data

- `train_test_split()` divides a dataset into training and testing subsets.
- The training set is used to train the model.
- The testing set is used to evaluate the model on data not used for training.
- `random_state=42` makes the split reproducible so that repeated runs with the same data and settings produce the same split.
- `stratify=y` helps preserve the target class proportions in the training and testing sets. It does not guarantee perfectly identical proportions in every split.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

- `X_train` contains training features.
- `X_test` contains testing features.
- `y_train` contains training targets.
- `y_test` contains testing targets.

---

# Day 2 Notes — Advertising Regression (Q1–Q5)

## 1. Regression Basics

- Regression: Supervised machine learning used to predict continuous numerical values.
- Simple Linear Regression: Uses one input feature.
- Multiple Linear Regression: Uses two or more input features.
- Polynomial Regression: Adds powers and interactions of features to model nonlinear relationships.
- Linear regression equation: `y = m × x + c`
- Multiple regression equation: `y = b₀ + b₁x₁ + b₂x₂ + ... + bₚxₚ`
- Best-fit line: Linear Regression minimizes the sum of squared residuals (SSE).
- Coefficient/slope: Represents the change in prediction for a one-unit change in a feature, holding other features constant.
- Intercept: Predicted target value when all input features are zero.

## 2. Evaluation Metrics

- SSE: Sum of squared differences between actual and predicted values.
- MSE: SSE divided by the number of observations.
- RMSE: Square root of MSE; expressed in the target's original units.
- MAE: Average absolute prediction error.
- R²: Proportion of target variance explained by the model.
  - `R² = 1`: Perfect predictions.
  - `R² = 0`: Equivalent to predicting the test-set mean.
  - `R² < 0`: Worse than that baseline.
- Adjusted R²: R² adjusted for the number of input features; penalizes unnecessary features.

### Important Formulas

- SSE = Sum of (actual value − predicted value)²
- MSE = SSE / n
- RMSE = Square root of MSE
- MAE = Sum of absolute (actual value − predicted value) / n
- R² = 1 − (SSE / SST)
- Adjusted R² = 1 − [(1 − R²) × (n − 1) / (n − p − 1)]

Here, `n` is the number of observations and `p` is the number of input features.

Rule of thumb: Lower RMSE/MAE is better; higher R²/adjusted R² is better.

## 3. Train-Test Split and Model Evaluation

- `test_size=0.2` means 80% training data and 20% testing data.
- `random_state=42` makes the split reproducible.
- `fit()` trains the model; `predict()` generates predictions.
- Use the same train-test split when comparing models to make the comparison fair.
- `model.coef_` contains learned coefficients; `model.intercept_` contains the intercept.

## 4. Pandas and NumPy Key Points

- `pd.read_csv()` loads a CSV file into a DataFrame.
- Selecting `df[["TV"]]` returns a DataFrame; selecting `df["TV"]` returns a Series.
- `Series.to_numpy()` converts a Pandas Series to a NumPy array and removes its index.
- `y_test` retains original row indices after `train_test_split`, whereas predictions from `model.predict()` are a NumPy array.
- When combining actual values and predictions in a DataFrame, converting `y_test` to NumPy ensures positional alignment.
- `pd.DataFrame(list_of_dictionaries)` converts a list of dictionaries into a table.
- `sort_values()` sorts rows; `ascending=False` sorts from largest to smallest.
- `head(5)` selects the first five rows.
- `.abs()` calculates absolute values.
- `.idxmin()` returns the index label of the minimum value.
- `.loc[]` selects by index label; `.iloc[]` selects by integer position.
- `.round(4)` rounds displayed values to four decimal places.

## 5. Q1 — Feature Combination Comparison

- Train models using TV, Radio, Newspaper individually and in different combinations.
- Compare RMSE and R² for each combination.
- A single feature uses Simple Linear Regression; two or more features use Multiple Linear Regression.
- The model with the lowest RMSE has the smallest prediction error among the tested models.
- In your results, TV + Radio performed almost as well as all three features; Newspaper added negligible predictive improvement.

## 6. Q2 — Linear vs Polynomial Regression

- Linear regression using TV and Radio models a linear relationship with both features.
- Degree-2 polynomial regression adds TV², Radio² and TV × Radio.
- `PolynomialFeatures` expands the original feature set.
- `include_bias=False` avoids adding a constant column because Linear Regression already fits an intercept by default.
- Use `fit_transform()` on training data and `transform()` on test data.
- Compare both models using RMSE and R².
- In your results, polynomial regression performed slightly better on the test set.

## 7. Q3 — Newspaper Contribution and Adjusted R²

- Model A uses TV, Radio and Newspaper; Model B uses only TV and Radio.
- Compare RMSE, R² and adjusted R².
- For Model A, `p = 3`; for Model B, `p = 2`.
- Use the number of observations in the evaluation set for `n` when calculating adjusted R² from the corresponding test-set R².
- In your results, Model B had higher adjusted R², suggesting Newspaper did not justify the extra feature complexity.

## 8. Q4 — Residual Analysis

- Residual: Actual value − predicted value.
- Positive residual: Model underpredicted.
- Negative residual: Model overpredicted.
- Absolute residual: Absolute value of the residual; measures error magnitude regardless of direction.
- Sort by absolute residual in descending order to find the five largest prediction errors.
- Actual-vs-predicted scatter plot: Points closer to the reference line `y = x` indicate more accurate predictions.
- The reference line represents perfect predictions.

## 9. Q5 — Effect of Training Data Size

- Test size `0.5` → 50% training data, 50% testing data.
- Test size `0.3` → 70% training data, 30% testing data.
- Test size `0.2` → 80% training data, 20% testing data.
- Train a separate model for each split and calculate RMSE and R².
- Different splits can produce different test observations, so more training data does not guarantee better scores on every individual split.
- In your results, the 70% training split performed best among the three tested splits.

---

# Day 3 Notes — Used Cars Regression (Q6–Q11)

## Q6 — Regression Model Comparison

- Decision Tree Regressor: Predicts by splitting data using feature-based conditions.
- Random Forest Regressor: Combines predictions from multiple decision trees; `n_estimators=100` means 100 trees.
- Compare models using RMSE (lower is better) and R² (higher is better).

## Q7 — Feature Engineering

- Feature engineering means creating new features from existing data.
- Vehicle Age = Reference Year − Manufacturing Year.
- Compare models with and without the new feature using the same train-test split.

## Q8 — Hyperparameter Tuning

- Hyperparameters are settings chosen before training, such as `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- `GridSearchCV` tests parameter combinations using cross-validation.
- `param_grid` stores the parameter values to test.
- `cv=5` performs 5-fold cross-validation.
- `scoring="neg_mean_squared_error"` uses negative MSE because GridSearchCV maximizes scores.
- `n_jobs=-1` uses all available CPU cores; default is `None`.
- `best_params_` returns the best settings; `best_estimator_` returns the tuned model.
- `best_score_` returns the best average cross-validation score.
- CV RMSE = Square root of (−`best_score_`) when scoring uses negative MSE.
- Compare the default model with the tuned model using test RMSE and R².

## Q9 — Categorical Encoding

- Label Encoding: Assigns an integer label to each category using `LabelEncoder`.
- One-Hot Encoding: Creates binary columns for categories using `pd.get_dummies()`.
- One-Hot Encoding avoids implying an artificial order between nominal categories.
- Use separate encoders for separate columns when applying LabelEncoder.

## Q10 — Outlier Detection

- IQR measures the spread of the middle 50% of observations.
- IQR = Q3 − Q1
- Lower Bound = Q1 − (1.5 × IQR)
- Upper Bound = Q3 + (1.5 × IQR)
- Values outside these bounds are potential outliers.
- `quantile(0.25)` gives Q1; `quantile(0.75)` gives Q3.
- Detecting outliers does not automatically mean they should be removed.

## Q11 — Feature Importance

- Random Forest's `feature_importances_` provides importance scores for input features.
- Higher scores indicate greater contribution to the forest's predictive splits.
- Sort scores using `sort_values()` in descending order to rank features.
- `plt.bar()` visualizes feature importance.
- Feature importance indicates predictive contribution, not causation.
- One-Hot Encoding creates separate importance scores for each encoded column.

---

# Day 4 — Social Network Ads Classification (Q12–Q18)

## Q12. Linear SVM vs RBF SVM

- SVM (Support Vector Machine): Classification algorithm that finds a decision boundary to separate classes.
- Linear kernel: Creates a straight decision boundary.
- RBF kernel: Can create nonlinear, curved decision boundaries.
- `SVC(kernel="linear")` creates Linear SVM.
- `SVC(kernel="rbf")` creates RBF SVM.
- `accuracy_score(y_test, y_pred)` measures classification accuracy.
- Feature scaling is important for SVM because it is sensitive to feature magnitudes.

## Q13. Feature Scaling

- StandardScaler: Standardizes features to have approximately zero mean and unit standard deviation.
- Formula: `z = (x − μ) / σ`
- `fit_transform(X_train)` calculates scaling parameters from training data and scales it.
- `transform(X_test)` applies the same scaling to test data.
- Never fit the scaler on test data, as this causes data leakage.
- Scaling does not guarantee higher accuracy, but helps SVM handle features on different scales.

## Q14. SVM Hyperparameter Tuning

- GridSearchCV: Tests combinations of hyperparameters using cross-validation to find the best combination.
- `param_grid` stores the parameter values to test.
- `C`: Controls the penalty for misclassification; larger values penalize errors more strongly.
- `kernel`: Specifies the decision boundary type, such as `linear` or `rbf`.
- `gamma`: Controls the influence of training points for the RBF kernel.
- `cv=5` performs five-fold cross-validation.
- `scoring="accuracy"` selects parameters based on cross-validation accuracy.
- `grid.best_params_` returns the best parameters.
- `grid.best_score_` returns the best cross-validation score.
- `grid.best_estimator_` returns the best fitted model.
- `grid.predict(X_test)` can predict directly using the best estimator when `refit=True` (the default).
- Compare the tuned and untuned models on the same test set.

## Q15. Feature Comparison

- Compare three feature combinations: `Age`, `EstimatedSalary`, and both features together.
- Train a separate model for each combination.
- Use the same train-test split for a fair comparison.
- Use `accuracy_score()` to evaluate each model.
- The combination with the highest test accuracy performs best on that split.

## Q16. Decision Boundary Visualization

- A decision boundary separates regions where a classifier predicts different classes.
- Linear SVM creates a straight boundary.
- RBF SVM can create a nonlinear boundary.
- `np.meshgrid()` generates grid coordinates for plotting.
- `model.predict()` predicts the class for each grid point.
- `plt.contourf()` colors the predicted regions.
- `plt.scatter()` plots the actual data points.
- Both models should use the same dataset and feature scaling for comparison.

## Q17. Confusion Matrix

- A confusion matrix evaluates a classification model by showing correct and incorrect predictions.
- `cm` is the variable storing the matrix returned by `confusion_matrix(y_test, y_pred)`.
- It helps identify TP, TN, FP and FN.
- TP: Actual 1, predicted 1.
- TN: Actual 0, predicted 0.
- FP: Actual 0, predicted 1.
- FN: Actual 1, predicted 0.

Scikit-learn's default confusion matrix layout is:

```text
                Predicted 0    Predicted 1

Actual 0             TN             FP
Actual 1             FN             TP
```

- Rows represent actual classes; columns represent predicted classes.
- `ConfusionMatrixDisplay(confusion_matrix=cm).plot()` visualizes the matrix.
- `plt.show()` displays the plot.
- In our dataset, class `1` (Purchased) is treated as the positive class.

## Q18. SVM vs Logistic Regression

- Logistic Regression: Predicts class probabilities and generally learns a linear decision boundary.
- RBF SVM: Can learn nonlinear decision boundaries.
- Both models use standardized features and the same train-test split.
- `accuracy_score()` measures overall correctness.
- Precision: Of predicted positives, how many are actually positive?

Precision = TP / (TP + FP)

- Recall: Of actual positives, how many were correctly identified?

Recall = TP / (TP + FN)

- F1-score: Harmonic mean of precision and recall.

F1-score = 2 × (Precision × Recall) / (Precision + Recall)

- `confusion_matrix()` compares actual and predicted classes.
- `precision_score()`, `recall_score()` and `f1_score()` calculate their respective metrics.
- Prefer the model based on the actual metrics and the problem's priorities. A nonlinear SVM is not automatically better than Logistic Regression.

---

# Day 5 Notes — Iris Classification (Q19–Q20)

## Q19. Analysing Decision Tree Depth and Overfitting

- `DecisionTreeClassifier(max_depth=depth)` limits the maximum depth of the Decision Tree.
- A shallow tree may underfit because it cannot learn enough patterns.
- A deep tree may overfit by learning details specific to the training data.
- `train_pred = model.predict(X_train)` predicts labels for the training data.
- `test_pred = model.predict(X_test)` predicts labels for unseen test data.
- `accuracy_score()` calculates the proportion of correct predictions.

Accuracy = Number of correct predictions / Total number of predictions

- High training accuracy with much lower testing accuracy indicates possible overfitting.
- `plt.plot()` compares training and testing accuracy across different tree depths.
- The preferred depth balances training performance with generalization to unseen data.

## Q20. Comparing Impurity Criteria in Decision Trees

- `criterion="gini"` uses Gini impurity to choose Decision Tree splits.
- `criterion="entropy"` uses Entropy to measure class uncertainty when choosing splits.
- Gini impurity = 1 − Sum of (pᵢ)²
- Entropy = − Sum of [pᵢ × log₂(pᵢ)]
- Here, `pᵢ` is the proportion of samples belonging to class `i`.
- Lower impurity indicates a purer group of samples.
- Keep `max_depth` and `random_state` the same to compare the criteria fairly.
- `confusion_matrix(y_test, y_pred)` shows correct and incorrect classifications.
- `accuracy_score()` compares the classification accuracy of both models.
- `plot_tree(model, filled=True)` visualizes the learned Decision Tree.
- Gini and Entropy may produce different splits, but neither is guaranteed to achieve better accuracy.
