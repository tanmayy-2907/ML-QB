import pandas as pd
import numpy as np
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

advertise_df = pd.read_csv("../mock_datasets/advertising.csv")
social_df = pd.read_csv("../mock_datasets/social_network_ads.csv")
cars_df = pd.read_csv("../mock_datasets/used_cars.csv")

#1. Finding Null Values in advertisement dataset
null_ct = advertise_df.isnull().sum()
print("Null Count: ", null_ct)

#2. Filling missing values with mean & median in Social dataset
social_df["Age"].fillna(social_df["Age"].mean())
social_df["EstimatedSalary"].fillna(social_df["EstimatedSalary"].median())

#3. Dropping Duplicates
advertise_df.drop_duplicates()
social_df.drop_duplicates()
cars_df.drop_duplicates()

#4. Standardisation (StandardScaler) Transforms features to have a mean of approximately 0 and
# a standard deviation of 1.
scale = StandardScaler()

X = advertise_df[["TV", "Radio", "Newspaper"]]
y = advertise_df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaled_X_train = scale.fit_transform(X_train)
scaled_X_test = scale.transform(X_test)

#5. Normalisation(MinMaxScaler): Scales features to a range of 0 to 1, typically.

normalize = MinMaxScaler()
normalized_X_train = normalize.fit_transform(X_train)
normalized_X_test = normalize.transform(X_test)

#Do either 4 or 5, both are not preferred at once
#Use these scaled attributes for fit and predict

#6. Encoding - Labelled Encoding & One-Hot Encoding
#Used on categorical columns to convert to numeric values

#Labelled
le_transmission = LabelEncoder()

cars_df["Tranmission"] = le_transmission.fit_transform(cars_df["Tranmission"]) #Ordered Encoding

#One-Hot

cars_df = pd.get_dummies(
    cars_df,
    columns=["Owner"],
    dtype=int
)

#7. IQR Method - Outlier Removal

cols = ["Year","Kms_Driven"]
for col in cols:
    Q1 = cars_df[col].quantile(0.25)
    Q3 = cars_df[col].quantile(0.75)

    IQR = Q3 - Q1

    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR

    cars_df = cars_df[(cars_df[col] >= LB) & (cars_df[col] <= UB)]


