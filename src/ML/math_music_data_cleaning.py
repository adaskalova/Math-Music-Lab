# -*- coding: utf-8 -*-
"""
# Data Acquisition and Cleaning

## Project
**Does the Hit Formula Change?
A Mathematical Analysis of Musical Evolution and Success Prediction
(The Mathematical Fingerprint of Musical Success)**

## Goals
1. Download datasets programmatically (no manual file uploads needed).
2. Store raw files locally for reproducibility.
3. Inspect and validate the data.
4. Clean and standardize both datasets.
5. Save cleaned versions for downstream analysis.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import requests

# Defining paths using pathlib
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Automatically create folders if they don't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"The folders are ready:")
print(f"- Raw: {RAW_DIR}")
print(f"- Processed: {PROCESSED_DIR}")

# Spotify Tracks
SPOTIFY_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"

# CORGIS Music
CORGIS_URL = "https://corgis-edu.github.io/corgis/datasets/csv/music/music.csv"

# Billboard Hot 100
BILLBOARD_URL = "https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/data-out/hot-100-current.csv"

# Local files
SPOTIFY_FILE = RAW_DIR / "spotify_raw.csv"
CORGIS_FILE = RAW_DIR / "corgis_raw.csv"
BILLBOARD_FILE = RAW_DIR / "billboard_raw.csv"

def download_file(url, destination):
    if destination.exists():
        print(f"The file {destination.name} already exists locally.")
        return

    print(f"Downloading '{destination.name}' from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        # set timeout=30 (30 seconds wait)
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            with open(destination, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded: {destination}")
        else:
            print(f"Download error: Status {response.status_code}")

    except requests.exceptions.Timeout:
        print(f"Error: The server did not respond within 30 seconds (Timeout).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

download_file(SPOTIFY_URL, SPOTIFY_FILE)
download_file(CORGIS_URL, CORGIS_FILE)
download_file(BILLBOARD_URL, BILLBOARD_FILE)

# list of files expected
required_files = [SPOTIFY_FILE, CORGIS_FILE, BILLBOARD_FILE]
missing = [f.name for f in required_files if not f.exists()]

if missing:
    print(f"WARNING: The following files are missing from {RAW_DIR}: {missing}")
    print("Please check your internet connection or upload the files manually to the folder.")
else:
    # only loads if everything is available
    df_spotify = pd.read_csv(SPOTIFY_FILE)
    df_corgis = pd.read_csv(CORGIS_FILE)
    df_billboard = pd.read_csv(BILLBOARD_FILE)

    print(f"All data has been loaded successfully!")
    print(f"Spotify: {df_spotify.shape}")
    print(f"CORGIS:  {df_corgis.shape}")
    print(f"Billboard: {df_billboard.shape}")

    # a quick look at the structure (head)
    print(df_spotify.head(2))
    print(df_corgis.head(2))
    print(df_billboard.head(2))

# ============================================================
# Cleanup of columns and NaN values
# ============================================================

# remove 'Unnamed: 0' from Spotify if it exists
if 'Unnamed: 0' in df_spotify.columns:
    df_spotify = df_spotify.drop(columns=['Unnamed: 0'])
    print("Removed column 'Unnamed: 0' from Spotify")

# check for missing values
print("\n--- Missing values in datasets ---")
print(f"Spotify:   {df_spotify.isnull().sum().sum()} missing")
print(f"CORGIS:    {df_corgis.isnull().sum().sum()} missing")
print(f"Billboard: {df_billboard.isnull().sum().sum()} missing")

# remove rows with missing critical values
df_spotify   = df_spotify.dropna(subset=['track_name', 'artists'])
df_corgis    = df_corgis.dropna(subset=['release.name', 'artist.name'])
df_billboard = df_billboard.dropna(subset=['title', 'performer'])

print(f"\n--- Dimensions after removing NaN rows ---")
print(f"Spotify:   {df_spotify.shape}")
print(f"CORGIS:    {df_corgis.shape}")
print(f"Billboard: {df_billboard.shape}")

# ============================================================
# Remove duplicates
# ============================================================

dup_spotify = df_spotify.duplicated(subset=['track_name', 'artists']).sum()
dup_corgis = df_corgis.duplicated(subset=['artist.name', 'release.name', 'song.year']).sum()
dup_billboard = df_billboard.duplicated(subset=['title', 'performer']).sum()

print("--- Duplicates before cleaning ---")
print(f"Spotify:   {dup_spotify}")
print(f"CORGIS:    {dup_corgis}")
print(f"Billboard: {dup_billboard}")

df_spotify = df_spotify.drop_duplicates(subset=['track_name', 'artists'], keep='first')
df_corgis = df_corgis.drop_duplicates(subset=['artist.name', 'release.name', 'song.year'], keep='first')
df_billboard = df_billboard.drop_duplicates(subset=['title', 'performer'], keep='first')

print("\n--- Dimensions after removing duplicates ---")
print(f"Spotify:   {df_spotify.shape}")
print(f"CORGIS:    {df_corgis.shape}")
print(f"Billboard: {df_billboard.shape}")

"""## Note on the CORGIS dataset

An initial review of the data showed that the `song.title` column in the
CORGIS dataset contains only null values and cannot be used as a song
identifier.

For this reason, **CORGIS will not be linked directly to Spotify and Billboard
at the song level**.

Instead, CORGIS will be used as:
- a **historical dataset** for analyzing the evolution of music by year (`song.year`);
- an **additional source** for unsupervised analysis (clustering, dimensionality reduction);
- a **time series analysis** of audio features such as `tempo`, `loudness`, and `key` across decades.

The main merge for modeling will be between **Spotify** and **Billboard**.
"""

# ============================================================
# Standardization of names
# ============================================================

# Spotify
df_spotify['track_name_clean'] = df_spotify['track_name'].astype(str).str.lower().str.strip()
df_spotify['artist_clean'] = df_spotify['artists'].astype(str).str.lower().str.strip()

# CORGIS
df_corgis['artist_clean'] = df_corgis['artist.name'].astype(str).str.lower().str.strip()
df_corgis['release_name_clean'] = df_corgis['release.name'].astype(str).str.lower().str.strip()

# Billboard
df_billboard['track_name_clean'] = df_billboard['title'].astype(str).str.lower().str.strip()
df_billboard['artist_clean'] = df_billboard['performer'].astype(str).str.lower().str.strip()

print("Standardization is complete.")

print("\nSpotify example:")
print(df_spotify[['track_name', 'track_name_clean', 'artists', 'artist_clean']].head(3))

print("\nExample from CORGIS:")
print(df_corgis[['release.name', 'release_name_clean', 'artist.name', 'artist_clean', 'song.year']].head(3))

print("\nExample from Billboard:")
print(df_billboard[['title', 'track_name_clean', 'performer', 'artist_clean']].head(3))

# ============================================================
# Check for matches between Spotify and Billboard
# ============================================================

# sets of (song, artist) are created for each dataset
spotify_songs   = set(zip(df_spotify['track_name_clean'], df_spotify['artist_clean']))
billboard_songs = set(zip(df_billboard['track_name_clean'], df_billboard['artist_clean']))

# matches are found
matches = spotify_songs & billboard_songs

print(f"Songs on Spotify:             {len(spotify_songs)}")
print(f"Songs on Billboard:           {len(billboard_songs)}")
print(f"Matches (Spotify ∩ Billboard): {len(matches)}")
print(f"Percentage of Billboard on Spotify:   {len(matches)/len(billboard_songs)*100:.1f}%")

# several examples are shown
print("\nExamples of matches:")
for song, artist in list(matches)[:5]:
    print(f"  Song-Artist: '{song}' — {artist}")

# ============================================================
# Merge of Spotify and Billboard → master dataset
# ============================================================

# merge is done using standardized names
df_master = pd.merge(
    df_spotify,
    df_billboard[['track_name_clean', 'artist_clean', 'peak_pos', 'wks_on_chart']],
    on=['track_name_clean', 'artist_clean'],
    how='left'  # all Spotify songs are saved
)

# adding a column: whether the song was on Billboard (1/0)
df_master['is_hit'] = df_master['wks_on_chart'].notna().astype(int)

print(f"Master dataset size: {df_master.shape}")
print(f"\nDistribution of is_hit:")
print(df_master['is_hit'].value_counts())
print(f"\nHit percentage: {df_master['is_hit'].mean()*100:.1f}%")

print("\nExample from master dataset:")
print(df_master[df_master['is_hit'] == 1][
    ['track_name', 'artists', 'popularity', 'peak_pos', 'wks_on_chart', 'is_hit']
].head(5))

# ============================================================
# Saving processed data
# ============================================================

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# saving master dataset (Spotify + Billboard)
master_path = PROCESSED_DIR / "master_dataset.csv"
df_master.to_csv(master_path, index=False)
print(f"master_dataset.csv → {master_path} ({df_master.shape})")

# CORGIS is saved separately (for time series analysis)
corgis_path = PROCESSED_DIR / "corgis_historical.csv"
df_corgis_historical = df_corgis[df_corgis['song.year'] > 0].copy()
df_corgis_historical.to_csv(corgis_path, index=False)
print(f"corgis_historical.csv → {corgis_path} ({df_corgis_historical.shape})")

print("\n--- Final review ---")
print(f"master_dataset:     {df_master.shape}")
print(f"corgis_historical:  {df_corgis_historical.shape}")
print(f"\nFiles in data/processed/:")
for f in PROCESSED_DIR.iterdir():
    size_kb = f.stat().st_size / 1024
    print(f"  File: {f.name} — {size_kb:.1f} KB")

# files with the CORRECT variable names are saved
# df_master and df_corgis_historical are used
df_master.to_csv('master_dataset_clean.csv', index=False)
df_corgis_historical.to_csv('corgis_historical_clean.csv', index=False)

print("The files are prepared for download with correct data.")

print("The files are saved locally with correct data.")

"""### Conclusion

In this first stage of the project, the following steps were completed successfully:
1. **Automated data acquisition:** Three independent sources (Spotify, CORGIS, Billboard) were downloaded via stable URL addresses.
2. **Data cleaning:** Missing values were handled, technical columns were removed, and thousands of duplicate records were cleaned out.
3. **Standardization:** `lower-case` and `trim` processing was applied to song and artist names to ensure reliable joining.
4. **Integration (Feature Enrichment):** A `master_dataset` was created, combining Spotify's audio features with Billboard's market-success information. The target variable `is_hit` was also created.
5. **Historical data preparation:** A `corgis_historical` dataset was set aside for the upcoming Time Series analysis.

**Next steps:**
- **Exploratory Data Analysis (EDA):** Visualization of the audio features.
- **Dimensionality Reduction (PCA):** Reducing the dimensionality of the musical vectors.
- **Clustering:** Searching for natural groups of songs.
- **Feature Engineering:** Generating new features to improve future models.
"""