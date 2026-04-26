# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load and basic clean (same as Part 1 to ensure consistency)
# should load a file: dataset.csv
df = pd.read_csv('path/to/data/dataset.csv')

df = df.dropna(subset=['track_name', 'artists'])
df = df[df['tempo'] > 0]
df = df.drop_duplicates(subset=['track_id'])

print("Data re-loaded and ready for statistical testing.")

# calculate Pearson correlation coefficient and p-value
r_stat, p_value = stats.pearsonr(df['loudness'], df['energy'])

print(f"Pearson Correlation Coefficient: {r_stat:.4f}")
print(f"P-value: {p_value}")

# visualizing the relationship with a regression line
plt.figure(figsize=(10, 6))
# .sample(2000) is used to prevent the plot from being overcrowded
# and to ensure the red line remains clearly visible
sns.regplot(x='loudness', y='energy', data=df.sample(2000),
            scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title("Loudness vs Energy (with Regression Line)")
plt.xlabel("Loudness (dB)")
plt.ylabel("Energy")
plt.show()

# define genres for ANOVA comparison
test_genres = ['acoustic', 'afrobeat', 'alt-rock', 'cantopop', 'tango']

# perform One-Way ANOVA across selected genres
groups = [df[df['track_genre'] == genre]['energy'].values for genre in test_genres]
# unpack the F-statistic and p-value returned by the One-Way ANOVA test
# groups is a list of arrays (one per genre)
# *groups unpacks it, passing each array as a separate argument to f_oneway
f_stat, p_value_anova = stats.f_oneway(*groups)

print(f"F-statistic : {f_stat:.4f}")
print(f"P-value     : {p_value_anova:.2e}")


# create a bar plot to compare average energy across selected genres
plt.figure(figsize=(10, 6))

# filter the dataset to include only the selected genres
filtered_df = df[df['track_genre'].isin(test_genres)]

# plot mean energy for each genre with confidence intervals
sns.barplot(
    x='track_genre',
    y='energy',
    data=filtered_df,
    hue='track_genre',     # fix for FutureWarning - assigns color by genre
    legend=False,          # hides the legend (not needed here)
    capsize=0.1,           # adds small caps to error bars
    palette='viridis'      # color palette for better visualization
)

# add title and labels
plt.title("Average Energy by Genre with 95% Confidence Intervals")
plt.xlabel("Genre")
plt.ylabel("Mean Energy")

# display the plot
plt.show()

# print ANOVA test results in a readable format
print("=" * 45)
print("        ANOVA Test Results")
print("=" * 45)
print(f"  F-statistic : {f_stat:.4f}")
print(f"  P-value     : {p_value_anova:.2e}")
print("=" * 45)

# Interpret the result automatically
if p_value_anova < 0.05:
    print("  Result: Reject H0")
    print("  The differences between genres ARE significant.")
else:
    print("  Result: Fail to reject H0")
    print("  The differences between genres are NOT significant.")
print("=" * 45)

# visual inspection with non-linear trend
plt.figure(figsize=(10, 6))

sample_df = df.sample(2000, random_state=42)

sns.scatterplot(
    data=sample_df,
    x='tempo',
    y='danceability',
    alpha=0.25
)

# LOWESS curve (non-linear trend)
sns.regplot(
    data=sample_df,
    x='tempo',
    y='danceability',
    scatter=False,
    lowess=True,
    color='red'
)

plt.title("Tempo vs Danceability with Non-Linear Trend")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Danceability")
plt.show()

from scipy import stats

pearson_r, pearson_p = stats.pearsonr(df['tempo'], df['danceability'])
spearman_rho, spearman_p = stats.spearmanr(df['tempo'], df['danceability'])

print("=" * 45)
print("   Correlation Comparison")
print("=" * 45)
print(f"Pearson r        : {pearson_r:.4f}")
print(f"Pearson p-value  : {pearson_p:.4e}")
print(f"Spearman rho     : {spearman_rho:.4f}")
print(f"Spearman p-value : {spearman_p:.4e}")
print("=" * 45)

# Final Spearman test
rho, p_value_spearman = stats.spearmanr(df['tempo'], df['danceability'])

print(f"Spearman Correlation (rho): {rho:.4f}")
print(f"P-value: {p_value_spearman:.4e}")
