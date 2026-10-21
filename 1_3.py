import pandas as pd
from sklearn.model_selection import train_test_split

social_networks_df = pd.read_csv("mock_datasets/social_network_ads.csv")

print(social_networks_df.head())
print(social_networks_df.tail())

print(social_networks_df.shape) #Rows and Columns count

print(social_networks_df.info()) #Metadata

print(list(social_networks_df.columns)) #Column names list

print(social_networks_df.isnull().sum()) #Count of null entries in each col

print(social_networks_df.describe()) #Statistical info about data

#feature & target extraction
X = social_networks_df.drop(columns = ["Purchased"])
Y = social_networks_df["Purchased"]

#Feature & Target Shapes
print(X.shape)
print(Y.shape)

# Q4. count how many customers purchased (Purchased = 1) and did not purchase (Purchased = 0).

#value_counts return count of all unique values present in a column
print("\nCount of Purchased Vs Non Purchased: ")
print(social_networks_df["Purchased"].value_counts())

#with normalize parameter, count is changed to proportion (Divides count of each value by total number of row values)
print("\nCount of Purchased Vs Non Purchased: (In Ratio/Percentage)")
print(social_networks_df["Purchased"].value_counts(normalize=True))

# Q5.Split the Social Network Ads data into training and test sets while preserving the class proportions using stratify=y.
#Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size= 0.2,
    random_state=42,
    stratify=Y #Only for Classification Dataset;
    #ensures that your training and testing sets have the exact same proportion of class labels as your original dataset.
)
#To verify stratification
print("\nOriginal class proportions:")
print(Y.value_counts(normalize=True))

print("\nTraining class proportions:")
print(Y_train.value_counts(normalize=True))

print("\nTesting class proportions:")
print(Y_test.value_counts(normalize=True))

print("\nSplit shapes:")
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)