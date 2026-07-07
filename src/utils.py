from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Load dataset
breast_cancer = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)

# Add target column
df["target"] = breast_cancer.target

# Create data/raw folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save dataset
df.to_csv("data/raw/breast_cancer.csv", index=False)

print("Dataset exported successfully!")
print("Shape:", df.shape)