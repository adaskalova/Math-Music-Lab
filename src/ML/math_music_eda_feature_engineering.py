# -*- coding: utf-8 -*-
"""
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Dimensionality Reduction (PCA)
- Clustering (K-Means)
- Time Series Analysis
"""

# ============================================================
# Topic: File Check
# Goal: Verify that files exist inside the 'data' folder
# ============================================================

import os

required_files = [
    "master_dataset_clean.csv",
    "corgis_historical_clean.csv"
]

all_ok = True
for f in required_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"file: {f} — {size_mb:.2f} MB")
    else:
        print(f" {f} — NOT FOUND in 'data' folder.")
        all_ok = False

if all_ok:
    print("\n Validation successful. Continuing ...")

# ============================================================
# Topic: Data Loading
# Goal: Load the cleaned datasets produced in math_music_data_cleaning.ipynb
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# Load cleaned datasets
# ------------------------------------------------------------

df = pd.read_csv("master_dataset_clean.csv")
df_hist = pd.read_csv("corgis_historical_clean.csv")

# ------------------------------------------------------------
# Audio features used throughout the project
# ------------------------------------------------------------

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

print("Datasets loaded successfully")
print(f"Master dataset shape: {df.shape}")
print(f"Historical dataset shape: {df_hist.shape}")

print(df.head())

# ============================================================
# Topic: Exploratory Data Analysis (EDA)
# Goal: Inspect dataset structure and available features
# ============================================================

print("=" * 60)
print("MASTER DATASET OVERVIEW")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nData types:")
print(df.dtypes)

print("\nFirst statistics for numerical columns:")
print(df.describe())

# ============================================================
# Topic: Exploratory Data Analysis (EDA) - Correlation Analysis
# Goal: Examine relationships between numerical audio features,
#       popularity, and hit status
# ============================================================

# ------------------------------------------------------------
# Select numerical columns for correlation analysis
# NOTE:
# Exclude 'peak_pos' and 'wks_on_chart' from the correlation
# matrix used for core interpretation because they come from
# Billboard and may introduce leakage in later modeling.
# ------------------------------------------------------------

corr_features = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "is_hit"
]

corr_matrix = df[corr_features].corr()

print("Correlation matrix:")
print(corr_matrix)

# ------------------------------------------------------------
# Visualize correlation matrix
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix of Audio Features, Popularity, and Hit Status")
plt.show()

# ============================================================
# Topic: Feature Engineering & Preprocessing
# Goal: Create new domain-specific features and scale numerical data
# ============================================================

# creating an "Energy-to-Loudness Ratio"
# the strong mutual correlation suggests that analyzing deviations
# from the expected baseline is necessary for a deeper understanding of the data.
df['energy_loudness_ratio'] = df['energy'] / (np.abs(df['loudness']) + 1)

# "Mood Index" is created
# danceability and valence are combined due to their inherent relationship
df['mood_index'] = (df['danceability'] + df['valence']) / 2

# defining columns for scaling
# new signs are added to the analysis list
extended_features = audio_features + ['energy_loudness_ratio', 'mood_index']

# StandardScaler
# scaling is mandatory for the next step: PCA and Clustering
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[extended_features] = scaler.fit_transform(df[extended_features])

print(f"Feature Engineering completed.")
print(f"New signs: energy_loudness_ratio, mood_index")
print(f"Data is scaled by StandardScaler.")
print(df_scaled[extended_features].head())

# ============================================================
# Topic: Dimensionality Reduction (PCA)
# Goal: Reduce 11 features to 2D and 3D for visualization
#       and to understand the variance structure of the data
# ============================================================

# ------------------------------------------------------------
# PCA with all components to analyze explained variance
# ------------------------------------------------------------
pca_full = PCA()
pca_full.fit(df_scaled[extended_features])

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# ------------------------------------------------------------
# Scree Plot (Elbow Plot for PCA)
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.bar(range(1, len(explained_variance) + 1), explained_variance,
        alpha=0.7, label='Individual Explained Variance')
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance,
         marker='o', color='red', label='Cumulative Explained Variance')

plt.axhline(y=0.80, color='gray', linestyle='--', label='80% threshold')

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance per Component')
plt.legend()
plt.tight_layout()
plt.show()

print("\nExplained variance per component:")
for i, v in enumerate(explained_variance):
    print(f"  PC{i+1}: {v:.4f} ({cumulative_variance[i]*100:.1f}% cumulative)")

# ------------------------------------------------------------
# PCA with 2 components for visualization
# ------------------------------------------------------------
pca_2d = PCA(n_components=2)
pca_result = pca_2d.fit_transform(df_scaled[extended_features])

df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# ------------------------------------------------------------
# Visualize PCA colored by Hit Status
# ------------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df.sample(3000, random_state=42),
    x='PC1', y='PC2',
    hue='is_hit',
    palette={0: 'steelblue', 1: 'crimson'},
    alpha=0.5,
    s=20
)
plt.title('PCA: 2D Projection of Songs colored by Hit Status')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
plt.legend(title='Is Hit', labels=['Not a Hit', 'Hit'])
plt.tight_layout()
plt.show()

# ============================================================
# Topic: Clustering
# Goal: Identify natural groupings of songs using K-Means
#       and the Elbow Method
# ============================================================

# Elbow Method to find optimal K
inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled[extended_features])
    inertia.append(kmeans.inertia_)

# plot Elbow
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='darkgreen')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# applying K-Means with k=5 (often optimal for music: e.g., Relaxed, Energetic, Dance, Acoustic, etc.)
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(df_scaled[extended_features])

# visualize clusters in PCA Space
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df.sample(3000, random_state=42),
    x='PC1', y='PC2',
    hue='cluster',
    palette='viridis',
    alpha=0.6,
    s=30
)
plt.title(f'K-Means Clustering (k={optimal_k}) projected on PCA axes')
plt.legend(title='Cluster')
plt.show()

# ============================================================
# Topic: Time Series - Diagnostics
# Goal: Check actual column names in df_hist
# ============================================================

print("Columns in df_hist:")
print(df_hist.columns.tolist())

print("\nFirst rows:")
print(df_hist.head(2))

# ============================================================
# Topic: Time Series Analysis
# Goal: Analyze how musical features (Loudness, Tempo)
#       evolved over the years (Loudness War phenomenon)
# ============================================================

# ------------------------------------------------------------
# Filter out songs with unknown year (year = 0)
# ------------------------------------------------------------
df_hist_filtered = df_hist[df_hist['song.year'] > 1950].copy()

print(f"Songs with known year: {len(df_hist_filtered)}")
print(f"Year range: {df_hist_filtered['song.year'].min()} - {df_hist_filtered['song.year'].max()}")

# ------------------------------------------------------------
# Group by year and calculate mean features
# ------------------------------------------------------------
yearly_data = df_hist_filtered.groupby('song.year')[['song.loudness', 'song.tempo']].mean().reset_index()

# ------------------------------------------------------------
# Visualization of Musical Evolution
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Loudness over time
axes[0].plot(yearly_data['song.year'], yearly_data['song.loudness'],
             marker='o', color='blue', linewidth=2, markersize=4)
axes[0].set_title('Mean Loudness Over Time (Loudness War)')
axes[0].set_ylabel('Mean Loudness (dB)')
axes[0].grid(True, alpha=0.3)

# Tempo over time
axes[1].plot(yearly_data['song.year'], yearly_data['song.tempo'],
             marker='s', color='orange', linewidth=2, markersize=4)
axes[1].set_title('Mean Tempo Over Time')
axes[1].set_ylabel('Mean Tempo (BPM)')
axes[1].set_xlabel('Year')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Evolution of Musical Features Over Time (CORGIS Dataset)', fontsize=14)
plt.tight_layout()
plt.show()

print("Time series analysis complete.")

# ============================================================
# Topic: Exporting Data & Automated Download
# Goal: Save the processed dataset and trigger an
#       automatic download to the local machine
# ============================================================

import os

output_path = "data/master_dataset_final.csv"
df.to_csv(output_path, index=False)

print(f"File successfully saved to: {output_path}")
print(f"Final dimensions: {df.shape}")