import pandas as pd

# Load the training data
train_data = pd.read_csv('data/train_subsample.csv')

# Display basic information about the dataset
print("Dataset shape:", train_data.shape)
print("\nColumn names:")
print(train_data.columns.tolist())
print("\nFirst few rows:")
print(train_data.head())
print("\nData types:")
print(train_data.dtypes)
print("\nBasic statistics:")
print(train_data.describe())