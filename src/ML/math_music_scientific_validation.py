# -*- coding: utf-8 -*-
"""
This file does not replace the original project pipeline. Instead, it acts as
a scientific validation layer on top of the existing data-cleaning,
feature-engineering, and modeling scripts.

The original scripts are responsible for:

1. math_music_data_cleaning.py
   - data acquisition
   - raw data storage
   - cleaning
   - Spotify-Billboard merge
   - target creation: is_hit

2. math_music_eda_feature_engineering.py
   - exploratory data analysis
   - feature engineering
   - PCA visualization
   - K-Means clustering
   - CORGIS time-series analysis

3. math_music_modeling_and_mlflow.py
   - initial regression and classification models
   - MLflow tracking
   - confusion matrix
   - ROC-AUC evaluation

This extension focuses on the evaluator's main feedback:

1. Reproducibility
   - verify required output files
   - record package versions, random seed, shapes, columns, and checksums

2. Scientific discipline
   - define research questions and hypotheses
   - compare models against a baseline
   - use stratified cross-validation
   - avoid preprocessing and feature leakage

3. Data-source justification
   - document how Spotify-style audio features and Billboard outcomes are linked
   - clarify the limitations of this linkage

4. Model diagnostics
   - inspect what the model learns
   - analyze false positives and false negatives
   - explain why audio-only prediction is inherently limited

The goal is not to prove that audio features fully explain commercial success.
The goal is to test whether they contain a weak but measurable predictive signal.
"""

# ============================================================
# Project role
# ============================================================

"""
Main research question:

Can the mathematical fingerprint of a song, represented by audio features,
predict whether the song becomes a mainstream Billboard hit?

This question connects the Math-Music-Lab ecosystem with machine learning:
songs are represented as numerical vectors, and success is modeled as a
classification problem.

The project therefore follows the core machine learning formulation:

input vector x = audio features of a song
target y = binary Billboard hit indicator

The model learns an approximation of the relationship:

audio structure -> probability of Billboard hit status

This formulation is intentionally limited. It does not include artist-level,
marketing, playlist, social media, or cultural variables.

Primary hypothesis:

H0:
Audio features do not contain predictive information about Billboard hit
status beyond random or majority-class guessing.

H1:
Audio features contain a weak but measurable predictive signal for Billboard
hit status.

Secondary hypothesis:

Even if audio features provide some signal, many hits will remain difficult
to predict because commercial success also depends on external factors:
artist popularity, marketing, cultural timing, platform exposure, and social
trends.

Role of this file:

This script tests the original project more rigorously. It does not simply ask
whether a model can produce a high accuracy score. Instead, it checks whether
the model performs better than a baseline, whether the validation is leakage-safe,
what the model learns, and where it fails.
"""

print("=" * 80)
print("Math-Music-Lab: Scientific Validation Extension")
print("=" * 80)
print("This file extends the existing pipeline with reproducibility, validation,")
print("interpretability, and error analysis.")
print("=" * 80)

# ============================================================
# 1. Imports, configuration, and reproducibility checks
# ============================================================

"""
Reproducibility goal:

This section verifies that the validation script is running on the expected
processed datasets and records enough information to detect accidental changes
between runs.

It checks:
- required file paths;
- dataset shapes;
- column names;
- target distribution;
- file checksums;
- Python and library versions;
- random seed.

This does not replace a full environment lock file. For full reproducibility,
the project should also include a requirements.txt or environment.yml file.
"""

import sys
import platform
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(".").resolve()

MASTER_DATASET_PROCESSED = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
MASTER_DATASET_FINAL = PROJECT_ROOT / "data" / "master_dataset_final.csv"
CORGIS_HISTORICAL_PROCESSED = PROJECT_ROOT / "data" / "processed" / "corgis_historical.csv"

REQUIRED_FILES = {
    "master_dataset_processed": MASTER_DATASET_PROCESSED,
    "master_dataset_final": MASTER_DATASET_FINAL,
    "corgis_historical_processed": CORGIS_HISTORICAL_PROCESSED,
}


def compute_file_checksum(path, algorithm="sha256"):
    """
    Compute a checksum for a file.

    The checksum is used as a simple data-versioning mechanism.
    If the source URLs change or the data is regenerated differently,
    the checksum will change as well.
    """
    hash_object = hashlib.new(algorithm)

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_object.update(chunk)

    return hash_object.hexdigest()


print("\n" + "=" * 80)
print("1. Reproducibility checks")
print("=" * 80)

print(f"Project root: {PROJECT_ROOT}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Random seed: {RANDOM_SEED}")

print(
    "\nEnvironment note: these versions are recorded for reproducibility. "
    "For a submitted repository, the same dependencies should also be listed "
    "in requirements.txt."
)

print("\nChecking required result files:")

missing_files = []

for name, path in REQUIRED_FILES.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        checksum = compute_file_checksum(path)
        print(f"  OK: {name}")
        print(f"      Path: {path}")
        print(f"      Size: {size_mb:.2f} MB")
        print(f"      SHA256: {checksum[:16]}...")
    else:
        print(f"  MISSING: {name}")
        print(f"      Expected path: {path}")
        missing_files.append(path)

if missing_files:
    raise FileNotFoundError(
        "Some required files are missing. "
        "Please run the original three scripts before running this validation file."
    )

print("\nAll required result files are available.")

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

df_master = pd.read_csv(MASTER_DATASET_PROCESSED)
df_final = pd.read_csv(MASTER_DATASET_FINAL)
df_corgis = pd.read_csv(CORGIS_HISTORICAL_PROCESSED)

print("\nLoaded datasets:")
print(f"  df_master: {df_master.shape}")
print(f"  df_final:  {df_final.shape}")
print(f"  df_corgis: {df_corgis.shape}")

# ------------------------------------------------------------
# Basic target validation
# ------------------------------------------------------------

if "is_hit" not in df_master.columns:
    raise ValueError("Column 'is_hit' is missing from df_master.")

if "is_hit" not in df_final.columns:
    raise ValueError("Column 'is_hit' is missing from df_final.")

print("\nTarget distribution in df_master:")
print(df_master["is_hit"].value_counts(dropna=False))
print(f"Hit rate: {df_master['is_hit'].mean() * 100:.2f}%")

print("\nTarget distribution in df_final:")
print(df_final["is_hit"].value_counts(dropna=False))
print(f"Hit rate: {df_final['is_hit'].mean() * 100:.2f}%")

# ------------------------------------------------------------
# Expected modeling columns
# ------------------------------------------------------------

BASE_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

missing_audio_features = [
    col for col in BASE_AUDIO_FEATURES if col not in df_master.columns
]

if missing_audio_features:
    raise ValueError(
        f"The following expected audio features are missing: {missing_audio_features}"
    )

print("\nBase audio features available:")
for col in BASE_AUDIO_FEATURES:
    print(f"  - {col}")

print("\nReproducibility check complete.")

# ------------------------------------------------------------
# Important note on reproducibility comparison
# ------------------------------------------------------------
#
# This manifest-based comparison requires at least two executions:
#
# 1. First execution:
#    - No previous manifest exists yet.
#    - The script creates data/processed/data_version_manifest.json.
#    - This first manifest becomes the baseline data version.
#
# 2. Second and later executions:
#    - The script loads the existing manifest.
#    - It compares the current datasets against the baseline.
#    - If checksums, shapes, columns, file sizes, or target distribution differ,
#      the script reports that the data has changed.
#
# Therefore, the script must be run at least twice to verify whether the data
# is unchanged compared to a previous recorded version.
#
# If all datasets are unchanged and the code, random seed, and package versions
# are also unchanged, the model results should be reproducible or numerically
# very close to the previous run.

# ------------------------------------------------------------
# Data version manifest
# ------------------------------------------------------------

import json
from datetime import datetime, timezone

DATA_VERSION_MANIFEST = PROJECT_ROOT / "data" / "processed" / "data_version_manifest.json"


def summarize_dataset_for_manifest(name, path, dataframe):
    """
    Create a compact reproducibility summary for a dataset.

    This summary is stored in a JSON manifest and can be compared
    across different runs of the project.
    """
    summary = {
        "name": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(dataframe.shape[0]),
        "columns_count": int(dataframe.shape[1]),
        "columns": list(dataframe.columns),
        "sha256": compute_file_checksum(path),
        "file_size_bytes": int(path.stat().st_size),
    }

    if "is_hit" in dataframe.columns:
        value_counts = dataframe["is_hit"].value_counts(dropna=False).to_dict()

        summary["target_is_hit"] = {
            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
            "hit_rate": float(dataframe["is_hit"].mean()),
        }

    return summary


def build_current_manifest():
    """
    Build the current data-version manifest.

    The manifest records:
    - dataset checksums
    - dataset shapes
    - column names
    - target distribution
    - package versions
    - random seed
    """
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "random_seed": RANDOM_SEED,
        },
        "datasets": {
            "master_dataset_processed": summarize_dataset_for_manifest(
                "master_dataset_processed",
                MASTER_DATASET_PROCESSED,
                df_master,
            ),
            "master_dataset_final": summarize_dataset_for_manifest(
                "master_dataset_final",
                MASTER_DATASET_FINAL,
                df_final,
            ),
            "corgis_historical_processed": summarize_dataset_for_manifest(
                "corgis_historical_processed",
                CORGIS_HISTORICAL_PROCESSED,
                df_corgis,
            ),
        },
    }

    return manifest


def compare_dataset_summaries(previous_summary, current_summary):
    """
    Compare one dataset summary from the previous manifest
    with the current dataset summary.
    """
    differences = []

    fields_to_compare = [
        "sha256",
        "rows",
        "columns_count",
        "columns",
        "file_size_bytes",
    ]

    for field in fields_to_compare:
        if previous_summary.get(field) != current_summary.get(field):
            differences.append(field)

    previous_target = previous_summary.get("target_is_hit")
    current_target = current_summary.get("target_is_hit")

    if previous_target != current_target:
        differences.append("target_is_hit")

    return differences


def compare_manifests(previous_manifest, current_manifest):
    """
    Compare previous and current data manifests.

    Returns a dictionary describing which datasets changed.
    """
    comparison = {}

    previous_datasets = previous_manifest.get("datasets", {})
    current_datasets = current_manifest.get("datasets", {})

    for dataset_name, current_summary in current_datasets.items():
        if dataset_name not in previous_datasets:
            comparison[dataset_name] = {
                "status": "NEW",
                "differences": ["dataset_missing_in_previous_manifest"],
            }
            continue

        previous_summary = previous_datasets[dataset_name]
        differences = compare_dataset_summaries(previous_summary, current_summary)

        if differences:
            comparison[dataset_name] = {
                "status": "CHANGED",
                "differences": differences,
            }
        else:
            comparison[dataset_name] = {
                "status": "UNCHANGED",
                "differences": [],
            }

    return comparison


def print_manifest_comparison(comparison):
    """
    Print a readable comparison report.
    """
    print("\n" + "=" * 80)
    print("Data version manifest comparison")
    print("=" * 80)

    changed_anything = False

    for dataset_name, result in comparison.items():
        status = result["status"]
        differences = result["differences"]

        print(f"\n{dataset_name}: {status}")

        if differences:
            changed_anything = True
            print("  Differences:")
            for diff in differences:
                print(f"    - {diff}")

    print("\nConclusion:")

    if changed_anything:
        print("At least one dataset changed compared to the previous manifest.")
        print("Model results may differ from the previous run.")
    else:
        print("All datasets match the previous recorded manifest.")
        print(
            "Results should be reproducible or very close, "
            "provided that code, seed, and package versions are unchanged."
        )


current_manifest = build_current_manifest()

if DATA_VERSION_MANIFEST.exists():
    print("\nExisting data version manifest found.")
    print(f"Manifest path: {DATA_VERSION_MANIFEST}")

    with open(DATA_VERSION_MANIFEST, "r", encoding="utf-8") as file:
        previous_manifest = json.load(file)

    manifest_comparison = compare_manifests(previous_manifest, current_manifest)
    print_manifest_comparison(manifest_comparison)
    print(
    "\nManifest note: the existing manifest is treated as the baseline and is "
    "not overwritten automatically."
)

else:
    print("\nNo previous data version manifest found.")
    print("Creating baseline data version manifest.")

    DATA_VERSION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_VERSION_MANIFEST, "w", encoding="utf-8") as file:
        json.dump(current_manifest, file, indent=2, ensure_ascii=False)

    print(f"Created: {DATA_VERSION_MANIFEST}")
    print("This file will be used as the baseline for future reproducibility checks.")

# ============================================================
# 2. Data source documentation and data dictionary
# ============================================================

"""
This section documents the role of each dataset in the project.

The goal is not to reload or recreate the data pipeline. The original scripts
already created the processed datasets. Here we document what each dataset
represents and how it supports the machine learning problem.

Dataset roles:

1. master_dataset_processed
   - Main supervised learning dataset.
   - Combines Spotify-style audio features with Billboard hit information.
   - Contains the binary target variable: is_hit.

2. master_dataset_final
   - Extended analytical dataset created after EDA and feature engineering.
   - Includes additional descriptive features, PCA coordinates, and clusters.
   - Useful for visualization and interpretation.
   - Some columns from this file must be treated carefully during modeling
     because they may have been computed before train/test splitting.

3. corgis_historical_processed
   - Historical music dataset used mainly for time-series and contextual analysis.
   - It supports the broader Math-Music-Lab narrative but is not the primary
     supervised classification dataset.
"""

print("\n" + "=" * 80)
print("2. Data source documentation and data dictionary")
print("=" * 80)

DATASET_DOCUMENTATION = pd.DataFrame(
    [
        {
            "dataset": "master_dataset_processed",
            "source_role": "Spotify audio features + Billboard hit target",
            "main_use": "Supervised classification and regression",
            "target_available": "yes",
        },
        {
            "dataset": "master_dataset_final",
            "source_role": "Feature-engineered analytical dataset",
            "main_use": "EDA, visualization, diagnostics, cautious modeling",
            "target_available": "yes",
        },
        {
            "dataset": "corgis_historical_processed",
            "source_role": "Historical/contextual music data",
            "main_use": "Time-series and background analysis",
            "target_available": "no / not primary",
        },
    ]
)

print("\nDataset documentation:")
print(DATASET_DOCUMENTATION.to_string(index=False))

# ----
# Pipeline documentation
# ----

PIPELINE_DOCUMENTATION = pd.DataFrame(
    [
        {
            "step": "1_data_cleaning",
            "script": "math_music_data_cleaning.py",
            "main_inputs": "raw Spotify/CORGIS/Billboard data sources",
            "main_outputs": "master_dataset.csv, corgis_historical.csv, cleaned CSV files",
            "role": "downloads, cleans, standardizes, merges sources, creates is_hit",
        },
        {
            "step": "2_eda_feature_engineering",
            "script": "math_music_eda_feature_engineering.py",
            "main_inputs": "cleaned master and CORGIS datasets",
            "main_outputs": "master_dataset_final.csv",
            "role": "EDA, feature engineering, scaling, PCA, clustering, time-series analysis",
        },
        {
            "step": "3_modeling_mlflow",
            "script": "math_music_modeling_and_mlflow.py",
            "main_inputs": "master_dataset_final.csv",
            "main_outputs": "MLflow runs, metrics, model diagnostics",
            "role": "initial regression/classification modeling and experiment tracking",
        },
        {
            "step": "4_scientific_validation",
            "script": "math_music_scientific_validation.py",
            "main_inputs": "processed/final datasets produced by earlier scripts",
            "main_outputs": "reproducibility, leakage, CV, interpretation, and error-analysis reports",
            "role": "scientific validation layer that addresses evaluator feedback",
        },
    ]
)

print("\nPipeline documentation:")
print(PIPELINE_DOCUMENTATION.to_string(index=False))

# ------------------------------------------------------------
# Compact feature dictionary for the main audio variables
# ------------------------------------------------------------

FEATURE_DICTIONARY = pd.DataFrame(
    [
        {
            "feature": "danceability",
            "meaning": "How suitable a track is for dancing",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "energy",
            "meaning": "Perceptual intensity and activity",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "loudness",
            "meaning": "Overall loudness in decibels",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "speechiness",
            "meaning": "Presence of spoken words",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "acousticness",
            "meaning": "Confidence that the track is acoustic",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "instrumentalness",
            "meaning": "Likelihood that the track has no vocals",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "liveness",
            "meaning": "Presence of live performance characteristics",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "valence",
            "meaning": "Musical positiveness or mood",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "tempo",
            "meaning": "Estimated beats per minute",
            "ml_role": "safe audio feature",
        },
        {
            "feature": "is_hit",
            "meaning": "Binary Billboard hit indicator",
            "ml_role": "target variable",
        },
    ]
)

print("\nMain feature dictionary:")
print(FEATURE_DICTIONARY.to_string(index=False))

# ------------------------------------------------------------
# Columns that require caution
# ------------------------------------------------------------

CAUTION_COLUMNS = [
    col
    for col in ["PC1", "PC2", "cluster", "peak_pos", "wks_on_chart"]
    if col in df_final.columns
]

print("\nColumns requiring caution:")
if CAUTION_COLUMNS:
    for col in CAUTION_COLUMNS:
        print(f"  - {col}")
else:
    print("  No predefined caution columns found in df_final.")

"""
Interpretation of caution columns:

- PC1 and PC2 are useful for visualization, but if they were computed on the
  full dataset before cross-validation, they should not be used directly as
  supervised learning features.

- cluster is useful for descriptive segmentation, but if it was assigned before
  train/test splitting, it can also introduce leakage-like information.

- peak_pos and wks_on_chart are Billboard outcome variables. They should not be
  used to predict whether a song is a hit, because they are known only after the
  song has already appeared on the chart.
"""

# ----
# Source linkage rationale
# ----

"""
Source linkage rationale:

The main supervised dataset links two different types of information:

1. Spotify-style audio features
   These variables describe the internal acoustic and mathematical structure
   of a song. Examples include tempo, loudness, energy, valence, danceability,
   acousticness, instrumentalness, and speechiness.

2. Billboard chart information
   This represents an external market outcome: whether a song achieved
   mainstream chart visibility.

The scientific purpose of linking these sources is to test the following
machine learning question:

Can a numerical audio-feature vector contain predictive signal about the
probability that a song becomes a Billboard hit?

This linkage is meaningful because it converts a cultural/music question into
a supervised learning problem:

audio features -> Billboard hit indicator

However, the linkage has important limitations. Billboard success is affected
by many factors that are not present in the audio features, including artist
popularity, label promotion, playlist placement, release timing, social media
exposure, genre trends, and broader cultural context.

Therefore, this project does not claim that audio features fully explain
commercial success. It only tests whether audio features contain partial
predictive signal.
"""

SOURCE_LINKAGE_DOCUMENTATION = pd.DataFrame(
    [
        {
            "source": "Spotify-style audio features",
            "represents": "Internal acoustic/mathematical song structure",
            "examples": "tempo, loudness, energy, valence, danceability",
            "ml_role": "input features X",
        },
        {
            "source": "Billboard chart data",
            "represents": "External market/chart outcome",
            "examples": "chart presence, hit indicator",
            "ml_role": "target variable y",
        },
        {
            "source": "CORGIS historical music data",
            "represents": "Historical/contextual music trends",
            "examples": "year-level or historical descriptors",
            "ml_role": "contextual analysis, not primary target",
        },
    ]
)

print("\nSource linkage documentation:")
print(SOURCE_LINKAGE_DOCUMENTATION.to_string(index=False))

print("\nSource linkage interpretation:")
print(
    "Spotify-style audio variables provide the numerical input representation, "
    "while Billboard data provides the external success label."
)
print(
    "The merge is scientifically useful because it allows the project to test "
    "whether audio structure has measurable predictive signal for chart success."
)
print(
    "The merge is also limited because many non-audio causes of success are "
    "not represented in the feature matrix."
)

print("\nData source documentation complete.")

# ============================================================
# 3. Merge quality and target audit
# ============================================================

"""
The target variable is_hit is created by linking Spotify-style audio data
with Billboard chart information.

This creates a useful supervised learning problem, but it also introduces
possible matching limitations:

- song titles may differ across sources;
- featured artists may be written differently;
- punctuation, remaster tags, remix labels, and casing may affect matching;
- Billboard success is an outcome, not an audio property.

This script performs a post-merge audit. It verifies the resulting target
distribution, class imbalance, missing targets, duplicates, and leakage-prone
columns in the processed dataset.

A full merge-quality audit would require intermediate files from the original
cleaning pipeline, such as:
- successfully matched Spotify-Billboard records;
- unmatched Spotify records;
- unmatched Billboard records;
- normalized title/artist keys used during matching.

Because this script receives already processed datasets, it documents merge
limitations and checks the consequences visible in the final data. A future
pipeline improvement would save matched and unmatched records for direct
inspection.
"""

print("\n" + "=" * 80)
print("3. Merge quality and target audit")
print("=" * 80)

# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

target_counts = df_master["is_hit"].value_counts().sort_index()
target_percentages = df_master["is_hit"].value_counts(normalize=True).sort_index() * 100

target_audit = pd.DataFrame(
    {
        "count": target_counts,
        "percentage": target_percentages.round(2),
    }
)

target_audit.index = target_audit.index.map({0: "non_hit", 1: "hit"})

print("\nTarget distribution:")
print(target_audit)

minority_rate = df_master["is_hit"].mean()

print(f"\nHit rate: {minority_rate * 100:.2f}%")

if minority_rate < 0.10:
    print("Class imbalance warning: hits are a small minority of the dataset.")
    print("Accuracy alone is not a sufficient evaluation metric.")
else:
    print("Class imbalance is present but not extremely severe.")

TARGET_INTERPRETATION = pd.DataFrame(
    [
        {
            "label": 1,
            "meaning": "Matched to Billboard hit evidence",
            "interpretation": "positive class for supervised learning",
        },
        {
            "label": 0,
            "meaning": "No Billboard hit evidence in the linked dataset",
            "interpretation": "negative class, but not proof that the song was culturally unsuccessful",
        },
    ]
)

print("\nTarget label interpretation:")
print(TARGET_INTERPRETATION.to_string(index=False))

# ------------------------------------------------------------
# Billboard-related columns
# ------------------------------------------------------------

possible_billboard_columns = [
    col
    for col in df_master.columns
    if any(token in col.lower() for token in ["billboard", "chart", "peak", "wk", "week"])
]

print("\nPossible Billboard/chart-related columns:")
if possible_billboard_columns:
    for col in possible_billboard_columns:
        print(f"  - {col}")
else:
    print("  No obvious Billboard/chart-related columns found.")

LEAKAGE_OUTCOME_COLUMNS = [
    col
    for col in ["peak_pos", "wks_on_chart", "weeks_on_chart", "chart_position"]
    if col in df_master.columns or col in df_final.columns
]

print("\nOutcome/leakage-prone columns detected:")
if LEAKAGE_OUTCOME_COLUMNS:
    for col in LEAKAGE_OUTCOME_COLUMNS:
        print(f"  - {col}")
else:
    print("  No predefined outcome columns detected.")

# ------------------------------------------------------------
# Duplicate and missing target checks
# ------------------------------------------------------------

missing_target = df_master["is_hit"].isna().sum()
duplicate_rows = df_master.duplicated().sum()

print(f"\nMissing values in is_hit: {missing_target}")
print(f"Fully duplicated rows in df_master: {duplicate_rows}")

if missing_target > 0:
    print("Warning: Missing target values should be removed before supervised modeling.")

if duplicate_rows > 0:
    print("Warning: Duplicate rows may inflate model evaluation if not handled carefully.")

# ------------------------------------------------------------
# Short interpretation
# ------------------------------------------------------------

print("\nMerge and target interpretation:")
print(
    "The binary target is useful for supervised learning, but it should be "
    "interpreted as a proxy for Billboard success rather than as a pure musical "
    "quality label."
)
print(
    "Non-hit does not necessarily mean unsuccessful; it may also mean that the "
    "song was not matched to Billboard data or did not appear in the selected "
    "chart source."
)
print(
    "For this reason, the project evaluates models with imbalance-aware metrics "
    "and avoids using post-outcome Billboard variables as predictors."
)

print("\nMerge quality and target audit complete.")

# ============================================================
# 4. Leakage audit and safe feature set
# ============================================================

"""
A central risk in machine learning projects is data leakage.

Data leakage occurs when information from the target, the future, or the full
dataset enters the training process in a way that would not be available in a
real prediction setting.

In this project, three groups of columns require caution:

1. Billboard outcome columns
   These variables describe chart performance after a song has already become
   a charting song. They must not be used to predict is_hit.

2. Precomputed PCA and clustering columns
   PC1, PC2, and cluster are useful for visualization. However, if they were
   computed on the full dataset before cross-validation, they should not be used
   directly as supervised learning features.

3. Identifier or text columns
   Song names, artist names, and IDs are not used in this compact validation
   model because they require separate text/entity handling.
"""

"""
Modeling assumption:

The validation model simulates a pre-outcome prediction setting.

This means that the model may use audio features that are available before
knowing chart performance, but it must not use variables that are created from
Billboard outcomes, full-dataset transformations, or post-chart information.
"""

print("\n" + "=" * 80)
print("4. Leakage audit and safe feature set")
print("=" * 80)

# ------------------------------------------------------------
# Define safe numeric audio features
# ------------------------------------------------------------

SAFE_AUDIO_FEATURES = [
    col for col in BASE_AUDIO_FEATURES if col in df_master.columns
]

print("\nSafe audio features selected for supervised validation:")
for col in SAFE_AUDIO_FEATURES:
    print(f"  - {col}")

# ------------------------------------------------------------
# Define excluded columns
# ------------------------------------------------------------

PRECOMPUTED_FEATURES = [
    col for col in ["PC1", "PC2", "cluster"] if col in df_final.columns
]

OUTCOME_COLUMNS = [
    col
    for col in [
        "peak_pos",
        "wks_on_chart",
        "weeks_on_chart",
        "chart_position",
        "last_week",
        "rank",
    ]
    if col in df_master.columns or col in df_final.columns
]

IDENTIFIER_OR_TEXT_COLUMNS = [
    col
    for col in df_master.columns
    if any(
        token in col.lower()
        for token in ["id", "uri", "track", "song", "title", "artist", "name"]
    )
]

EXCLUDED_FROM_SUPERVISED_MODELING = sorted(
    set(
        PRECOMPUTED_FEATURES
        + OUTCOME_COLUMNS
        + IDENTIFIER_OR_TEXT_COLUMNS
        + ["is_hit"]
    )
)

print("\nColumns excluded from compact supervised validation:")
if EXCLUDED_FROM_SUPERVISED_MODELING:
    for col in EXCLUDED_FROM_SUPERVISED_MODELING:
        print(f"  - {col}")
else:
    print("  No excluded columns detected.")

SAFE_FEATURE_AUDIT = pd.DataFrame(
    [
        {
            "feature_group": "raw audio features",
            "examples": ", ".join(SAFE_AUDIO_FEATURES),
            "decision": "included",
            "reason": "available before Billboard outcome and directly describes audio structure",
        },
        {
            "feature_group": "precomputed PCA / cluster features",
            "examples": ", ".join(PRECOMPUTED_FEATURES) if PRECOMPUTED_FEATURES else "not present",
            "decision": "excluded",
            "reason": "may have been computed on the full dataset before cross-validation",
        },
        {
            "feature_group": "Billboard outcome columns",
            "examples": ", ".join(OUTCOME_COLUMNS) if OUTCOME_COLUMNS else "not detected",
            "decision": "excluded",
            "reason": "known only after chart performance occurs",
        },
        {
            "feature_group": "identifier or text columns",
            "examples": ", ".join(IDENTIFIER_OR_TEXT_COLUMNS[:5])
            if IDENTIFIER_OR_TEXT_COLUMNS
            else "not detected",
            "decision": "excluded",
            "reason": "require separate entity/text modeling and may encode popularity proxies",
        },
    ]
)

print("\nSafe feature audit:")
print(SAFE_FEATURE_AUDIT.to_string(index=False))

# ------------------------------------------------------------
# Build modeling matrix
# ------------------------------------------------------------

modeling_df = df_master[SAFE_AUDIO_FEATURES + ["is_hit"]].copy()

before_dropna = modeling_df.shape[0]
modeling_df = modeling_df.dropna(subset=SAFE_AUDIO_FEATURES + ["is_hit"])
after_dropna = modeling_df.shape[0]

dropped_rows = before_dropna - after_dropna

X = modeling_df[SAFE_AUDIO_FEATURES]
y = modeling_df["is_hit"].astype(int)

print(f"\nRows before dropping missing modeling values: {before_dropna}")
print(f"Rows after dropping missing modeling values:  {after_dropna}")
print(f"Dropped rows: {dropped_rows}")

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nClass distribution used for supervised validation:")
print(y.value_counts().sort_index())

# ------------------------------------------------------------
# Final leakage statement
# ------------------------------------------------------------

print("\nLeakage audit conclusion:")
print(
    "The compact validation model uses only raw audio features that would be "
    "available before knowing Billboard chart outcomes."
)
print(
    "Precomputed PCA coordinates, cluster labels, and chart outcome variables "
    "are excluded from this supervised validation step."
)
print(
    "This makes the next cross-validation step more conservative and "
    "leakage-aware."
)

print("\nLeakage audit and safe feature definition complete.")

# ============================================================
# 5. Baseline and leakage-safe cross-validation
# ============================================================

"""
This section evaluates whether audio features contain predictive signal beyond
a simple baseline.

The validation uses Stratified K-Fold cross-validation so that every fold keeps
approximately the same hit/non-hit ratio.

All preprocessing is placed inside sklearn Pipelines. This is important because
the scaler is fitted only on each training fold and then applied to that fold's
validation split. This avoids preprocessing leakage.
"""

print("\n" + "=" * 80)
print("5. Baseline and leakage-safe cross-validation")
print("=" * 80)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Cross-validation setup
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED,
)

scoring = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

models = {
    "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
}

# ------------------------------------------------------------
# Run cross-validation
# ------------------------------------------------------------

cv_results_rows = []

for model_name, model in models.items():
    print(f"\nEvaluating model: {model_name}")

    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    row = {"model": model_name}

    for metric_name in scoring:
        scores = results[f"test_{metric_name}"]
        row[f"{metric_name}_mean"] = scores.mean()
        row[f"{metric_name}_std"] = scores.std()

    cv_results_rows.append(row)

cv_results = pd.DataFrame(cv_results_rows)

# ------------------------------------------------------------
# Display compact results
# ------------------------------------------------------------

metric_columns = [
    "roc_auc_mean",
    "roc_auc_std",
    "average_precision_mean",
    "average_precision_std",
    "f1_mean",
    "f1_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
]

cv_results_display = cv_results[["model"] + metric_columns].copy()

for col in metric_columns:
    cv_results_display[col] = cv_results_display[col].round(4)

print("\nCross-validation results:")
print(cv_results_display.to_string(index=False))

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

best_model_by_pr_auc = cv_results.sort_values(
    by="average_precision_mean",
    ascending=False,
).iloc[0]

print("\nBest model by Average Precision / PR-AUC:")
print(f"  Model: {best_model_by_pr_auc['model']}")
print(f"  Average Precision: {best_model_by_pr_auc['average_precision_mean']:.4f}")
print(f"  ROC-AUC: {best_model_by_pr_auc['roc_auc_mean']:.4f}")
print(f"  F1: {best_model_by_pr_auc['f1_mean']:.4f}")

# ----
# Hypothesis-oriented decision rule
# ----

dummy_row = cv_results[
    cv_results["model"] == "dummy_most_frequent"
].iloc[0]

best_ap_improvement = (
    best_model_by_pr_auc["average_precision_mean"]
    - dummy_row["average_precision_mean"]
)

best_roc_improvement = (
    best_model_by_pr_auc["roc_auc_mean"]
    - dummy_row["roc_auc_mean"]
)

print("\nHypothesis-oriented validation summary:")
print(f"  Dummy Average Precision: {dummy_row['average_precision_mean']:.4f}")
print(f"  Best Average Precision:  {best_model_by_pr_auc['average_precision_mean']:.4f}")
print(f"  AP improvement:          {best_ap_improvement:.4f}")
print(f"  Dummy ROC-AUC:           {dummy_row['roc_auc_mean']:.4f}")
print(f"  Best ROC-AUC:            {best_model_by_pr_auc['roc_auc_mean']:.4f}")
print(f"  ROC-AUC improvement:     {best_roc_improvement:.4f}")

if best_ap_improvement > 0 and best_roc_improvement > 0:
    print(
        "Interpretation: the best model improves over the dummy baseline in both "
        "Average Precision and ROC-AUC. This supports H1: audio features contain "
        "some measurable predictive signal."
    )
else:
    print(
        "Interpretation: the best model does not clearly improve over the dummy "
        "baseline in both key metrics. This does not provide strong evidence "
        "against H0."
    )

print(
    "This is not interpreted as causal proof. It only evaluates predictive "
    "signal under the selected data representation and validation design."
)

print("\nValidation interpretation:")
print(
    "The dummy baseline represents a model that does not learn musical structure. "
    "A useful ML model should improve over this baseline, especially in ROC-AUC "
    "and Average Precision."
)
print(
    "Average Precision is emphasized because the positive class is rare and "
    "precision-recall behavior is more informative than accuracy alone."
)

print("\nBaseline and leakage-safe cross-validation complete.")

# ============================================================
# 6. Model interpretation with permutation importance
# ============================================================

"""
This section asks what the model appears to learn from the audio features.

Permutation importance measures how much model performance decreases when one
feature is randomly shuffled. If shuffling a feature strongly hurts performance,
the model relied on that feature.

This method is model-agnostic and easier to reproduce than SHAP because it does
not require additional dependencies.
"""

print("\n" + "=" * 80)
print("6. Model interpretation with permutation importance")
print("=" * 80)

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

# ----
# Descriptive feature profile by target class
# ----

feature_profile_by_class = (
    modeling_df
    .groupby("is_hit")[SAFE_AUDIO_FEATURES]
    .mean()
    .round(3)
)

print("\nMean audio feature profile by target class:")
print(feature_profile_by_class)

if {0, 1}.issubset(set(feature_profile_by_class.index)):
    feature_difference_hit_minus_nonhit = (
        feature_profile_by_class.loc[1] - feature_profile_by_class.loc[0]
    ).sort_values(key=lambda values: values.abs(), ascending=False)

    feature_difference_table = pd.DataFrame(
        {
            "feature": feature_difference_hit_minus_nonhit.index,
            "hit_minus_non_hit_mean_difference": feature_difference_hit_minus_nonhit.values,
        }
    )

    feature_difference_table["hit_minus_non_hit_mean_difference"] = (
        feature_difference_table["hit_minus_non_hit_mean_difference"].round(3)
    )

    print("\nLargest mean feature differences: hit minus non-hit:")
    print(feature_difference_table.to_string(index=False))
else:
    print(
        "\nFeature difference table skipped because both target classes "
        "are not present in the modeling data."
    )

print("\nFeature profile interpretation:")
print(
    "This descriptive comparison shows how average audio characteristics differ "
    "between hit and non-hit labels in the linked dataset."
)
print(
    "These differences are not causal effects. They only provide context for "
    "what the supervised model may learn from the available audio features."
)

# ------------------------------------------------------------
# Train/test split for interpretation only
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)

interpretation_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

interpretation_model.fit(X_train, y_train)

y_test_proba = interpretation_model.predict_proba(X_test)[:, 1]

interpretation_pr_auc = average_precision_score(y_test, y_test_proba)
interpretation_roc_auc = roc_auc_score(y_test, y_test_proba)

print("\nInterpretation holdout performance:")
print(f"  Average Precision: {interpretation_pr_auc:.4f}")
print(f"  ROC-AUC: {interpretation_roc_auc:.4f}")

# ------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------

perm_importance = permutation_importance(
    interpretation_model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

importance_table = pd.DataFrame(
    {
        "feature": SAFE_AUDIO_FEATURES,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std,
    }
).sort_values(by="importance_mean", ascending=False)

importance_table["importance_mean"] = importance_table["importance_mean"].round(5)
importance_table["importance_std"] = importance_table["importance_std"].round(5)

print("\nPermutation importance ranked by Average Precision decrease:")
print(importance_table.to_string(index=False))

top_features = importance_table.head(3)["feature"].tolist()

print("\nTop interpreted features:")
for feature in top_features:
    print(f"  - {feature}")

print("\nInterpretation note:")
print(
    "Permutation importance should be interpreted as model behavior, not as "
    "a causal explanation of musical success."
)
print(
    "A feature can be important for prediction without being the true reason "
    "why a song becomes popular."
)

print("\nModel interpretation complete.")

# ============================================================
# 7. Error analysis and scientific conclusion
# ============================================================

"""
This section studies where the model fails.

Instead of reporting only aggregate metrics, it compares false positives and
false negatives. This is important because Billboard success is influenced by
many non-audio factors such as artist popularity, marketing, timing, playlists,
social media, and cultural context.
"""

print("\n" + "=" * 80)
print("7. Error analysis and scientific conclusion")
print("=" * 80)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

# ------------------------------------------------------------
# Predictions on the same interpretation holdout split
# ------------------------------------------------------------

# ----
# Threshold analysis
# ----

DEFAULT_THRESHOLD = 0.50

precision_values, recall_values, threshold_values = precision_recall_curve(
    y_test,
    y_test_proba,
)

threshold_summary = pd.DataFrame(
    {
        "threshold": threshold_values,
        "precision": precision_values[:-1],
        "recall": recall_values[:-1],
    }
)

precision_recall_sum = (
    threshold_summary["precision"] + threshold_summary["recall"]
)

threshold_summary["f1"] = np.where(
    precision_recall_sum > 0,
    2
    * threshold_summary["precision"]
    * threshold_summary["recall"]
    / precision_recall_sum,
    0,
)

threshold_summary = threshold_summary.replace([np.inf, -np.inf], np.nan).dropna()

if not threshold_summary.empty:
    selected_threshold_indices = np.linspace(
        0,
        len(threshold_summary) - 1,
        min(6, len(threshold_summary)),
        dtype=int,
    )

    selected_thresholds = threshold_summary.iloc[selected_threshold_indices].round(3)

    print("\nPrecision-recall trade-off at selected thresholds:")
    print(selected_thresholds.to_string(index=False))

    best_f1_threshold_row = threshold_summary.sort_values(
        by="f1",
        ascending=False,
    ).iloc[0]

    print("\nBest threshold by F1 on interpretation holdout:")
    print(f"  Threshold: {best_f1_threshold_row['threshold']:.3f}")
    print(f"  Precision: {best_f1_threshold_row['precision']:.3f}")
    print(f"  Recall:    {best_f1_threshold_row['recall']:.3f}")
    print(f"  F1:        {best_f1_threshold_row['f1']:.3f}")
else:
    print("\nThreshold summary is empty. Using default threshold only.")

print(
    "\nThreshold note: threshold tuning is shown for diagnostic interpretation. "
    "A production system should tune thresholds inside cross-validation or on a "
    "separate validation set."
)

# Default threshold used for the main error analysis
y_test_pred = (y_test_proba >= DEFAULT_THRESHOLD).astype(int)

conf_matrix = confusion_matrix(y_test, y_test_pred)

print("\nConfusion matrix at threshold 0.50:")
print(conf_matrix)

print("\nClassification report at threshold 0.50:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["non_hit", "hit"],
        zero_division=0,
    )
)

# ------------------------------------------------------------
# Build error analysis table
# ------------------------------------------------------------

error_analysis_df = X_test.copy()

metadata_columns = [
    col
    for col in df_master.columns
    if any(
        token in col.lower()
        for token in ["track", "song", "title", "artist", "name"]
    )
]

# Keep only metadata columns that can be safely aligned by index.
metadata_columns = [
    col for col in metadata_columns
    if col not in SAFE_AUDIO_FEATURES and col != "is_hit"
]

if metadata_columns:
    metadata_for_errors = df_master.loc[X_test.index, metadata_columns].copy()
    error_analysis_df = pd.concat(
    [
        metadata_for_errors.reset_index(drop=True),
        error_analysis_df.reset_index(drop=True),
    ],
    axis=1,
)
else:
    error_analysis_df = error_analysis_df.reset_index(drop=True)

error_analysis_df["actual_is_hit"] = y_test.reset_index(drop=True).values
error_analysis_df["predicted_is_hit"] = y_test_pred
error_analysis_df["predicted_probability"] = y_test_proba

display_columns_for_errors = (
    metadata_columns[:4]
    + SAFE_AUDIO_FEATURES
    + ["predicted_probability"]
)

def classify_prediction_error(row):
    if row["actual_is_hit"] == 1 and row["predicted_is_hit"] == 1:
        return "true_positive"
    if row["actual_is_hit"] == 0 and row["predicted_is_hit"] == 0:
        return "true_negative"
    if row["actual_is_hit"] == 0 and row["predicted_is_hit"] == 1:
        return "false_positive"
    if row["actual_is_hit"] == 1 and row["predicted_is_hit"] == 0:
        return "false_negative"
    return "unknown"

error_analysis_df["error_type"] = error_analysis_df.apply(
    classify_prediction_error,
    axis=1,
)

error_counts = error_analysis_df["error_type"].value_counts()

print("\nPrediction outcome counts:")
print(error_counts)

# ------------------------------------------------------------
# Feature averages by error type
# ------------------------------------------------------------

feature_means_by_error = (
    error_analysis_df
    .groupby("error_type")[SAFE_AUDIO_FEATURES]
    .mean()
    .round(3)
)

print("\nAverage audio features by prediction outcome:")
print(feature_means_by_error)

print("\nError analysis interpretation:")
print(
    "False positives are songs whose audio profile looks hit-like to the model, "
    "but which are not labeled as Billboard hits in the linked dataset."
)
print(
    "False negatives are labeled hits whose audio profile was not recognized as "
    "hit-like by the model."
)
print(
    "Both error types are expected because Billboard success depends on many "
    "non-audio variables that are not included in this compact feature set."
)

# ------------------------------------------------------------
# Most confident mistakes
# ------------------------------------------------------------

false_positives = error_analysis_df[
    error_analysis_df["error_type"] == "false_positive"
].sort_values(by="predicted_probability", ascending=False)

false_negatives = error_analysis_df[
    error_analysis_df["error_type"] == "false_negative"
].sort_values(by="predicted_probability", ascending=True)


print("\nMost confident false positives:")
if not false_positives.empty:
    print(false_positives.head(5)[display_columns_for_errors])
else:
    print("No false positives at threshold 0.50.")

print("\nMost confident false negatives:")
if not false_negatives.empty:
    print(false_negatives.head(5)[display_columns_for_errors])
else:
    print("No false negatives at threshold 0.50.")

# ------------------------------------------------------------
# Conclusion
# ------------------------------------------------------------

print("\nConclusion:")

print("\n1. Evidence relative to the hypotheses:")
print(
    "The validation framework tests whether audio features provide predictive "
    "signal beyond a trivial baseline. Evidence for H1 is based on whether "
    "non-trivial models improve over the dummy classifier in cross-validated "
    "Average Precision and ROC-AUC."
)
print(
    "If the best model improves over the dummy baseline in these metrics, the "
    "results support the alternative hypothesis that audio features contain "
    "some measurable predictive signal for Billboard hit status."
)
print(
    "This does not prove that audio features cause commercial success. It only "
    "shows predictive association under the selected dataset, features, and "
    "validation design."
)

print("\n2. Reproducibility and validation discipline:")
print(
    "The script records dataset shapes, column names, target distribution, file "
    "checksums, package versions, and the random seed. This makes accidental "
    "changes in the processed data easier to detect across runs."
)
print(
    "The supervised validation uses a leakage-aware feature set and stratified "
    "cross-validation. Precomputed PCA coordinates, cluster labels, identifiers, "
    "and post-outcome Billboard variables are excluded from the compact "
    "supervised model."
)

print("\n3. Source-linkage limitations:")
print(
    "The project links Spotify-style audio features, which represent internal "
    "musical/acoustic structure, with Billboard chart data, which represents an "
    "external market outcome."
)
print(
    "This linkage is scientifically useful for testing partial predictive "
    "signal, but it is limited because chart success also depends on artist "
    "reputation, promotion, playlist exposure, social media, release timing, "
    "genre trends, and cultural context."
)
print(
    "Therefore, a non-hit label should be interpreted as absence of Billboard "
    "hit evidence in the linked dataset, not as proof that a song was musically "
    "or culturally unsuccessful."
)

print("\n4. Model learning and errors:")
print(
    "Permutation importance and feature-profile comparisons show which audio "
    "features the model relies on most strongly. Error analysis then separates "
    "true positives, true negatives, false positives, and false negatives."
)
print(
    "False positives may represent songs with hit-like audio profiles that did "
    "not appear as Billboard hits in the linked data. False negatives may "
    "represent Billboard hits whose success was driven by factors outside the "
    "available audio features."
)
print(
    "This confirms that aggregate metrics alone are insufficient. The model "
    "must be evaluated through baseline comparison, threshold behavior, feature "
    "interpretation, and concrete error patterns."
)






