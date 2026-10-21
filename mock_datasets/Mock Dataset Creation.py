
import os
import numpy as np
import pandas as pd

# Fixed seed so results remain reproducible
rng = np.random.default_rng(42)


# ==========================================
# 1. ADVERTISING DATASET — REGRESSION
# ==========================================

n = 200

tv = rng.uniform(0, 300, n)
radio = rng.uniform(0, 50, n)
newspaper = rng.uniform(0, 120, n)

sales = (
    3
    + 0.045 * tv
    + 0.18 * radio
    + 0.0005 * tv * radio
    + rng.normal(0, 2, n)
)

advertising = pd.DataFrame({
    "TV": tv.round(2),
    "Radio": radio.round(2),
    "Newspaper": newspaper.round(2),
    "Sales": sales.round(2)
})

advertising.to_csv(
    "mock_datasets/advertising.csv", index=False
)


# ==========================================
# 2. SOCIAL NETWORK ADS — CLASSIFICATION
# ==========================================

n = 400

age = rng.integers(18, 61, n)
salary = rng.integers(15000, 150001, n)

logits = -6 + 0.09 * age + 0.000035 * salary
probability = 1 / (1 + np.exp(-logits))

purchased = rng.binomial(1, probability)

social = pd.DataFrame({
    "Age": age,
    "EstimatedSalary": salary,
    "Purchased": purchased
})

social.to_csv(
    "mock_datasets/social_network_ads.csv", index=False
)


# ==========================================
# 3. USED CARS — REGRESSION
# ==========================================

n = 250

year = rng.integers(2005, 2026, n)
kms = rng.integers(5000, 180001, n)

transmission = rng.choice(
    ["Manual", "Automatic"], n
)

owner = rng.choice(
    ["First", "Second", "Third"], n
)

price = (
    800000
    - (2026 - year) * 25000
    - kms * 1.4
    + np.where(transmission == "Automatic", 70000, 0)
    + rng.normal(0, 45000, n)
)

cars = pd.DataFrame({
    "Year": year,
    "Kms_Driven": kms,
    "Transmission": transmission,
    "Owner": owner,
    "Selling_Price": np.maximum(price, 20000).round(2)
})

cars.to_csv(
    "mock_datasets/used_cars.csv", index=False
)


# ==========================================
# CONFIRM DATASET CREATION
# ==========================================

print("All mock datasets created successfully!\n")

for filename in [
    "advertising.csv",
    "social_network_ads.csv",
    "used_cars.csv"
]:
    df = pd.read_csv(f"mock_datasets/{filename}")
    print(f"\n{filename}")
    print("Shape:", df.shape)
    print(df.head(3))
