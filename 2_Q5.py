# Q5. Effect of training data size on sales prediction. Train models on varying Train-test split ratio(50,70,80)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression


# Training a Linear Model on TV, Radio, Newspaper
advertisement_df = pd.read_csv("mock_datasets/advertising.csv")

X = advertisement_df[["TV", "Radio","Newspaper"]]
Y = advertisement_df["Sales"]

test_size = [0.5, 0.3, 0.2]
results = []
for t in test_size:
    X_train,X_test,Y_train,Y_test = train_test_split(
        X,Y,
        test_size=t,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)

    results.append({
        "Test Size" : t,
        "RMSE" : rmse,
        "R2" : r2
    })

results_df = pd.DataFrame(results)
print(results_df)