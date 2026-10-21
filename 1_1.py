#Advertising Dataset
import pandas as pd
from pandas.core.common import random_state

#1. dataset loading
advertisement_df = pd.read_csv("mock_datasets/advertising.csv")

#2. dataset inspection
print("\nTop 5 rows for AD:")
print(advertisement_df.head())

print("\nBottom 5 rows for AD:")
print(advertisement_df.tail())

print("\nNo of R&Cs")
print(advertisement_df.shape)

print("\nColumn Names:")
print(advertisement_df.columns) #Returns an index object
#An index in pandas is a special object that acts as the address book or look-up key for your data.
#Two Types: Row Index(df.index) & Column Index(df.columns)
print(list(advertisement_df.columns)) #Conversion to list

print("\nData Types & Non-null counts:")
advertisement_df.info() #Prints Structure & Metadata(col names, non-null ct within each col, its DT & Memory Usage) & Returns None

print("\nCount of Null Entries:")
print(advertisement_df.isnull().sum())
# Output of df.isnull():       Output of df.isnull().sum():
#     Col1   Col2   Col3        Col1    0
#  0  False  False   True       Col2    1
#  1  False   True  False       Col3    1
#  2  False  False  False       dtype: int64

#isnull() by defaults runs element-wise to evaluate cell and swaps it with
# True -> if empty cell / NaN
# False -> if any value is present

#isnull().sum() by defaults set the axis to Vertical Axis(axis=0), So it adds the count vertically for each col and returns
print("\nStatistical Summary:")
print(advertisement_df.describe()) #Returns stats of actual data(min,max,mean,std etc...)

#3. Extracting features & target_var
X = advertisement_df.drop(columns = ["Sales"]) #Features
Y = advertisement_df["Sales"] #Target

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape: ", Y.shape)

#4. Split into Training & Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size = 0.2, #20% for Testing
    random_state = 42
)
print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", Y_train.shape)
print("Testing target:", Y_test.shape)









