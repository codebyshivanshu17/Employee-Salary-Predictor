import pandas as pd

# Load dataset
df = pd.read_csv("data/salary_data.csv")

print("Original dataset shape:", df.shape)

# Remove unnecessary target-related columns
df = df.drop(
    columns=["salary", "salary_currency"],
    errors="ignore"
)

# Remove rows where salary target is missing
df = df.dropna(subset=["salary_in_usd"])

# Features
X = df.drop(columns=["salary_in_usd"])

# Target
y = df["salary_in_usd"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("salary_in_usd")

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nSample features:")
print(X.head())

print("\nSample salaries:")
print(y.head())




