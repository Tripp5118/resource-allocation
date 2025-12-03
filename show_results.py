"""
Generate publication-quality bar chart comparing Weighted and LLM strategy results.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
weighted_explore = [0.0, 0.2, 0.5, 0.8, 1.0]
weighted_exploit = [1.0, 0.8, 0.5, 0.2, 0.0]
weighted_ratios = [5.98e7, 5.74e7, 2.78e7, 6.40e7, 4.15e7]

llm_temps = [0.0, 0.3, 0.7, 1.0]
llm_ratios = [4.87e7, 4.87e7, 6.02e7, 4.39e7]

# Design space maximum (true optimum from Thermo-Calc)
design_space_max = 7248303.754531006

# Create figure with more width
fig, ax = plt.subplots(figsize=(15, 7))

# X positions with wider spacing
n_weighted = len(weighted_ratios)
n_llm = len(llm_ratios)
bar_spacing = 1.7  # Increased spacing between bars
x_weighted = np.arange(n_weighted) * bar_spacing
x_llm = (np.arange(n_llm) * bar_spacing) + x_weighted[-1] + 2.5  # Larger gap between groups
x_max = x_llm[-1] + 2.5  # Position for design space maximum

# Colors
color_weighted = '#2E86AB'  # Professional blue
color_llm = '#A23B72'       # Professional magenta
color_max = '#E63946'       # Professional red for maximum

# Plot bars with narrower width for more spacing
bar_width = 1.0
bars_weighted = ax.bar(x_weighted, weighted_ratios, width=bar_width, 
                       color=color_weighted, edgecolor='black', linewidth=1.2,
                       label='Weighted Strategy')

bars_llm = ax.bar(x_llm, llm_ratios, width=bar_width,
                  color=color_llm, edgecolor='black', linewidth=1.2,
                  label='LLM Strategy')

bar_max = ax.bar([x_max], [design_space_max], width=bar_width,
                 color=color_max, edgecolor='black', linewidth=1.2,
                 label='Design Space Maximum')

# X-axis labels
weighted_labels = [f'Explore={e:.1f}\nExploit={x:.1f}' for e, x in zip(weighted_explore, weighted_exploit)]
llm_labels = [f'Temp={t:.1f}' for t in llm_temps]
max_label = ['Design Space\nMaximum']

# Set x-ticks at bar positions
ax.set_xticks(list(x_weighted) + list(x_llm) + [x_max])
ax.set_xticklabels(weighted_labels + llm_labels + max_label, fontsize=14)

# Y-axis
ax.set_ylabel('Best K/|CTE| Ratio', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(weighted_ratios + llm_ratios + [design_space_max]) * 1.15)

# Format y-axis in scientific notation
ax.ticklabel_format(style='scientific', axis='y', scilimits=(7,7))
ax.yaxis.get_offset_text().set_fontsize(14)
ax.tick_params(axis='y', labelsize=14)

# Remove grid
ax.grid(False)
ax.set_axisbelow(True)

# Add subtle horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=0.8)

# Legend
ax.legend(loc='upper right', fontsize=14, frameon=True, 
          edgecolor='black', fancybox=False, shadow=False)

# Title
ax.set_title('Comparison of Resource Allocation Strategies', 
             fontsize=18, fontweight='bold', pad=20)

# Add value labels on top of bars
for bar in bars_weighted:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2e}',
            ha='center', va='bottom', fontsize=11, rotation=0)

for bar in bars_llm:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2e}',
            ha='center', va='bottom', fontsize=11, rotation=0)

# Add value label for design space maximum
height = bar_max[0].get_height()
ax.text(bar_max[0].get_x() + bar_max[0].get_width()/2., height,
        f'{height:.2e}',
        ha='center', va='bottom', fontsize=11, rotation=0)

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Adjust x-axis limits to add padding
ax.set_xlim(-1, x_max + 1.5)

# Tight layout
plt.tight_layout()

# Save
plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('results_comparison.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✓ Charts saved as results_comparison.png and results_comparison.pdf")
print(f"✓ Design space maximum K/|CTE|: {design_space_max:.2e}")
plt.show()