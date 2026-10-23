import pandas as pd
from sklearn.model_selection import train_test_split

used_cars_df = pd.read_csv("../mock_datasets/used_cars.csv")

# print(used_cars_df.head())
# print(used_cars_df.tail())
#
# print(used_cars_df.shape) #Rows and Columns count
#
# print(used_cars_df.info()) #Metadata
#
# print(list(used_cars_df.columns)) #Column names list
#
# print(used_cars_df.isnull().sum()) #Count of null entries in each col
#
# print(used_cars_df.describe()) #Statistical info about data
#
# #feature & target extraction
# X = used_cars_df.drop(columns = ["Selling_Price"])
# Y = used_cars_df["Selling_Price"]
#
# #Feature & Target Shapes
# print(X.shape)
# print(Y.shape)
#
# #Split data
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y,
#     test_size= 0.2,
#     random_state=42,
# )
#
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


### QUESTIONS
#Q1. Print only the Year, Kms_Driven and Selling_Price columns.
#Approach1 - Using Names
carsdf1 = used_cars_df[["Year", "Kms_Driven", "Selling_Price"]]
print(carsdf1)

#Approach2 - Using only selective cols while loading csv itself
carsdf2 = pd.read_csv("../mock_datasets/used_cars.csv", usecols =["Year", "Kms_Driven", "Selling_Price"])
print(carsdf2)

#Approach3 - Using Index rather than names (iloc property)
#iloc -> Integer Location; Pandas Property to select Rows & Cols using numerical pos
#The basic syntax for .iloc always follows a [row, column] format:
#df.iloc[row_positions, col_positions]

print(used_cars_df.iloc[:,[0,1,4]]) # : -> Read all rows, [0,1,4] -> Select Column 0,1,4 (indexing starts from 0)


#Q2. Find the number of missing values in every column.
print(used_cars_df.isnull().sum())

#Q3. Display the unique values in the Transmission column.
print(used_cars_df["Transmission"].unique())
