# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# setting the visual style for charts
sns.set_theme(style="whitegrid")
# %matplotlib inline

print("Libraries imported successfully.")

"""Loading the files:"""

# loading the datasets
df_spotify = pd.read_csv('path/to/file')
df_corgis = pd.read_csv('path/to/file')

# basic check of the data size
print(f"Spotify Dataset: {df_spotify.shape[0]} rows, {df_spotify.shape[1]} columns")
print(f"CORGIS Dataset: {df_corgis.shape[0]} rows, {df_corgis.shape[1]} columns")

# preview the first few entries to check column names and data types consistency
df_spotify.head()

# preview the first few entries to check column names and data types consistency
df_corgis.head()

# checking for missing values in Spotify dataset
print("Missing values in Spotify dataset:")
print(df_spotify.isnull().sum().sort_values(ascending=False).head(5))

print("\nMissing values in CORGIS dataset:")
# focusing on core audio and metadata columns for CORGIS
print(df_corgis.isnull().sum().sort_values(ascending=False).head(5))

# cleaning Spotify Data
# remove rows with missing track names or artists
df_spotify_clean = df_spotify.dropna(subset=['track_name', 'artists'])

# filter out tracks with 0 tempo (likely errors or non-musical entries)
df_spotify_clean = df_spotify_clean[df_spotify_clean['tempo'] > 0]

# cleaning CORGIS Data
# In CORGIS, filter out songs with year 0 as it's a common placeholder
df_corgis_clean = df_corgis[df_corgis['song.year'] > 0].copy()

print(f"Cleaned Spotify size: {df_spotify_clean.shape}")
print(f"Cleaned CORGIS size: {df_corgis_clean.shape}")

# check for duplicates based on track ID
duplicates_count = df_spotify_clean.duplicated(subset=['track_id']).sum()
print(f"Number of duplicate Track IDs in Spotify: {duplicates_count}")

# keep the first occurrence
df_spotify_clean = df_spotify_clean.drop_duplicates(subset=['track_id'])
print(f"Final dataset size after removing duplicates: {df_spotify_clean.shape}")


# selecting essential columns for analysis from Spotify
# these represent the 'mathematical' audio features
spotify_features = ['popularity', 'duration_ms', 'danceability', 'energy',
                   'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'track_genre']

df_analysis = df_spotify_clean[spotify_features].copy()

df_analysis.head()

features = ['energy', 'danceability', 'valence', 'tempo', 'loudness']

df_analysis[features].hist(bins=30, figsize=(12, 8))
plt.suptitle("Distribution of Key Musical Features")
plt.show()

corr_features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

plt.figure(figsize=(12, 8))
sns.heatmap(df_analysis[corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Key Musical Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='loudness', y='energy', data=df_analysis, alpha=0.3)
plt.title("Energy vs Loudness")
plt.show()

top_genres = df_analysis['track_genre'].value_counts().head(5).index

df_top = df_analysis[df_analysis['track_genre'].isin(top_genres)]

plt.figure(figsize=(10, 6))
sns.boxplot(x='track_genre', y='energy', data=df_top)
plt.title("Energy Distribution Across Top Genres")
plt.xticks(rotation=30)
plt.show()
