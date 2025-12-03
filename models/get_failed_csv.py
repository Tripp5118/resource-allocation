import pandas as pd

# Read the CSV files
training_data = pd.read_csv('models/training_data_cleaned.csv')
compositions = pd.read_csv('models/compositions.csv')

# Define the x columns (the input features)
x_cols = ['Fe', 'Co', 'Cr', 'Ni', 'V']

# Find compositions that were NOT in the training data
# by doing an anti-join (left merge with indicator, keep only 'left_only')
merged = compositions.merge(
    training_data[x_cols],
    on=x_cols,
    how='left',
    indicator=True
)

# Keep only rows that are in compositions but NOT in training_data
failed_predictions = merged[merged['_merge'] == 'left_only']

# Drop the merge indicator column
failed_predictions = failed_predictions.drop(columns=['_merge'])

# Save to new CSV
failed_predictions.to_csv('failed_predictions.csv', index=False)

print(f"Found {len(failed_predictions)} compositions that failed to generate predictions")
print(f"Saved to 'failed_predictions.csv'")