import pandas as pd
from sklearn.model_selection import train_test_split

advertisement_df = pd.read_csv("mock_datasets/advertising.csv")
used_cars_df = pd.read_csv("mock_datasets/used_cars.csv")
social_networks_df = pd.read_csv("mock_datasets/social_network_ads.csv")

# #Filtering Rows Using Conditions
#
# # Cars manufactured in or after 2020
# recent_cars = used_cars_df[
#     used_cars_df["Year"] > 2020
# ]
# print(recent_cars)
#
# # Cars driven more than 50,000 km
# high_usage = used_cars_df[
#     used_cars_df["Kms_Driven"] > 50000
# ]
# print(high_usage)
#
# # Cars that are both recent and automatic
# filtered_df = used_cars_df[(used_cars_df["Year"] > 2020) & (used_cars_df["Kms_Driven"] > 50000)]
# print(filtered_df)

#FINAL QUESTIONS

#Q1. Load advertising.csv and display only TV and Sales.
selective_df = advertisement_df[["TV", "Sales"]]
print(selective_df)


#Q2. Find the number of rows in used_cars.csv.
# row_ct = used_cars_df.count() #Incorrect, count() returns no. of non-null values in each col
# print(row_ct)
print(len(used_cars_df))
#OR
print(used_cars_df.shape[0])

#Q3. Print the unique owner categories in the Used Cars dataset.
unique_owner_cats = used_cars_df["Owner"].unique()
print(unique_owner_cats)

#Q4. Find the percentage of customers who purchased a car.
percent = social_networks_df["Purchased"].value_counts(normalize=True)
print(percent*100) #Both purchased vs unpurchased percent
print()
print(percent[1]*100) #Only Purchased (only confused here what does 0 and 1 represent, if they are values within cols then how am i able toaccess it by[1] ?

Xadv = advertisement_df.drop(columns = ["Sales"])
Yadv = advertisement_df["Sales"]

#Q5. Split the Advertising data into 70% training and 30% testing.
X_train,X_test,Y_train,Y_test = train_test_split(
    Xadv,Yadv,
    test_size=0.3,
    random_state=42
)
