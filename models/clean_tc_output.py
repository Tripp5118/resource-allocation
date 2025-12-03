import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Combine all CSV files from CalcFiles
print("Loading CSV files from CalcFiles/...")
csv_files = glob.glob('CalcFiles/Results_Set_*.csv')
print(f"Found {len(csv_files)} files")

dfs = []
for f in csv_files:
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {df.shape}")

# Remove duplicates
df = df.drop_duplicates(subset=['Fe', 'Co', 'Cr', 'Ni', 'V'])
print(f"After deduplication: {df.shape}")

# Drop rows with NaN
df = df.dropna(subset=['CTE (1/K)', 'K (W/(mK))'])
print(f"After dropping NaN: {df.shape}")

# Drop rows where K = 1e-20 (bad default value)
df = df[df['K (W/(mK))'] > abs(1e-10)]
print(f"After dropping K=1e-20: {df.shape}")

# Check for other suspicious values
print(f"\nCTE range: [{df['CTE (1/K)'].min():.6e}, {df['CTE (1/K)'].max():.6e}]")
print(f"K range: [{df['K (W/(mK))'].min():.6e}, {df['K (W/(mK))'].max():.6e}]")


# Visualizations
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# CTE distribution
axes[0, 0].hist(df['CTE (1/K)'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('CTE (1/K)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('CTE Distribution')

# K distribution
axes[0, 1].hist(df['K (W/(mK))'], bins=50, edgecolor='black')
axes[0, 1].set_xlabel('K (W/(mK))')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('K Distribution')

# CTE vs K scatter
axes[0, 2].scatter(df['CTE (1/K)'], df['K (W/(mK))'], alpha=0.5, s=1)
axes[0, 2].set_xlabel('CTE (1/K)')
axes[0, 2].set_ylabel('K (W/(mK))')
axes[0, 2].set_title('CTE vs K')

# Box plots
axes[1, 0].boxplot([df['CTE (1/K)']], labels=['CTE'])
axes[1, 0].set_ylabel('CTE (1/K)')
axes[1, 0].set_title('CTE Box Plot')

axes[1, 1].boxplot([df['K (W/(mK))']], labels=['K'])
axes[1, 1].set_ylabel('K (W/(mK))')
axes[1, 1].set_title('K Box Plot')

# Composition distribution (stacked bar for first 100 samples)
sample = df.head(100)
axes[1, 2].bar(range(len(sample)), sample['Fe'], label='Fe')
axes[1, 2].bar(range(len(sample)), sample['Co'], bottom=sample['Fe'], label='Co')
axes[1, 2].bar(range(len(sample)), sample['Cr'], bottom=sample['Fe']+sample['Co'], label='Cr')
axes[1, 2].bar(range(len(sample)), sample['Ni'], bottom=sample['Fe']+sample['Co']+sample['Cr'], label='Ni')
axes[1, 2].bar(range(len(sample)), sample['V'], bottom=sample['Fe']+sample['Co']+sample['Cr']+sample['Ni'], label='V')
axes[1, 2].set_xlabel('Sample')
axes[1, 2].set_ylabel('Composition (mole fraction)')
axes[1, 2].set_title('Composition Distribution (first 100)')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300)
print("Saved visualization to data_analysis.png")

# Save cleaned dataset
df.to_csv('training_data_cleaned.csv', index=False)
print(f"\nSaved cleaned dataset to training_data_cleaned.csv")
print(f"Final dataset: {len(df)} compositions")

# Summary statistics
print("\n" + "="*60)
print("FINAL STATISTICS")
print("="*60)
print(f"\nCTE (1/K):")
print(f"  Mean: {df['CTE (1/K)'].mean():.6e}")
print(f"  Std:  {df['CTE (1/K)'].std():.6e}")
print(f"  Min:  {df['CTE (1/K)'].min():.6e}")
print(f"  Max:  {df['CTE (1/K)'].max():.6e}")

print(f"\nK (W/(mK)):")
print(f"  Mean: {df['K (W/(mK))'].mean():.4f}")
print(f"  Std:  {df['K (W/(mK))'].std():.4f}")
print(f"  Min:  {df['K (W/(mK))'].min():.4f}")
print(f"  Max:  {df['K (W/(mK))'].max():.4f}")

print("\n" + "="*60)