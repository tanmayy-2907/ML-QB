# Q1. Simple Linear Regression & Multiple Linear Regression
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("../mock_datasets/advertising.csv")

# 2. Separate all features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 3. Split data once for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Define feature combinations
combinations = [
    ["TV"],
    ["Radio"],
    ["Newspaper"],
    ["TV", "Radio"],
    ["TV", "Newspaper"],
    ["Radio", "Newspaper"],
    ["TV", "Radio", "Newspaper"]
]

results = []

# 5. Train and evaluate each model
for features in combinations:

    model = LinearRegression()

    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Features": " + ".join(features),
        "RMSE": rmse,
        "R2 Score": r2
    })

# 6. Display comparison
results_df = pd.DataFrame(results)

print(results_df.round(4))

# 7. Identify the best model by RMSE
best_model = results_df.loc[results_df["RMSE"].idxmin()]

print("\nBest feature combination by RMSE:")
print(best_model)

#REASONING BEHIND PT 6 & 7
# 1. results = [] + .append({...})
# Builds a list where each item is one dict (one model's results).
# After looping: results = [{"Features": "TV", "RMSE": 12.3, ...}, ...]

# 2. pd.DataFrame(results)
# Converts list of dicts → table (rows = models, columns = Features/RMSE/R²).
# Why: unlocks vectorized operations like .idxmin(), .round(), filtering — impossible on a raw list.

# 3. .idxmin()
# Returns the index label of the row with the minimum value.
# results_df["RMSE"].idxmin() → e.g., 2 (the row label where RMSE is smallest).

# 4. .loc vs .iloc
# .loc[label] → selects by index label (the name of the row).
# .iloc[position] → selects by integer position (0, 1, 2, ...).
# Slicing: .loc[0:2] includes 2; .iloc[0:2] excludes 2.

# 5. Index = Label
# "Index" and "label" are the same thing — the row identifier.
# Default is 0, 1, 2, ... (looks like position, but it's a label).
# After train_test_split, labels get shuffled; Look at this example
#  Default: labels happen to be 0, 1, 2
# df.loc[0]    # selects row with label 0
# df.iloc[0]   # selects row at position 0
#  → Same result here, but for different reasons
#
#  After a split, labels are shuffled:
# y_test = pd.Series([12.5, 8.3], index=[3, 7])
#     3    12.5
#     7     8.3
#
# y_test.loc[3]    # → 12.5 (label 3 exists)
# y_test.iloc[0]   # → 12.5 (position 0, which is label 3)
# y_test.loc[0]    # → KeyError! (no label 0)

# 6. In your code:
# best_model = results_df.loc[results_df["RMSE"].idxmin()]
# .idxmin() gives the label → .loc uses it → returns the full best-model row.