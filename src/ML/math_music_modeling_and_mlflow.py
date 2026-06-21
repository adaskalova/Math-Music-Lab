# -*- coding: utf-8 -*-
"""
## Objectives
- Train a **Linear Regression** model to predict song popularity.
- Train a **Classification** model (Random Forest) to predict hit status.
- Use **MLflow** for experiment tracking and model versioning.
- Perform **Feature Importance** analysis to decode the "formula for success".
- Evaluate models using advanced metrics (Confusion Matrix, F1-score, RMSE).
"""

# ============================================================
# Topic: Environment Setup
# Goal: Install MLflow and import necessary ML libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)

# experiment Tracking
import mlflow

print("Libraries and MLflow installed successfully.")

# ==============================================================================
# Topic: Data Loading
# Goal: Load the processed dataset from math_music_eda_feature_engineering.ipynb
# ==============================================================================
import os

processed_file = "data/master_dataset_final.csv"

if os.path.exists(processed_file):
    df_model = pd.read_csv(processed_file)
    print(f"Data loaded successfully: {df_model.shape[0]} rows, {df_model.shape[1]} columns.")

    # quick sanity check for the newly added features from math_music_eda_feature_engineering.ipynb
    required_new_cols = ['energy_loudness_ratio', 'mood_index', 'cluster', 'PC1', 'PC2']
    for col in required_new_cols:
        if col in df_model.columns:
            print(f"   - Feature '{col}' found")
        else:
            print(f"   - Feature '{col}' MISSING! Check math_music_eda_feature_engineering.ipynb output.")

    print(df_model.head(3))
else:
    print(f"ERROR: {processed_file} not found in 'data/' folder. Please upload it.")

# ============================================================
# Topic: Data Preparation
# Goal: Define feature columns and target variables for
#       classification and regression
# ============================================================

# ------------------------------------------------------------
# Feature selection for modeling
# Keep only numerical / engineered music-related features
# Exclude text/id columns and leakage-prone columns
# ------------------------------------------------------------

feature_cols = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "energy_loudness_ratio",
    "mood_index",
    "PC1",
    "PC2",
    "cluster"
]

# ------------------------------------------------------------
# Convert boolean column to integer if needed
# ------------------------------------------------------------
if df_model["explicit"].dtype == "bool":
    df_model["explicit"] = df_model["explicit"].astype(int)

# ------------------------------------------------------------
# define X and targets
# classification target: is_hit
# regression target: popularity
# ------------------------------------------------------------
X = df_model[feature_cols]

y_class = df_model["is_hit"]
y_reg = df_model["popularity"]

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
print("Feature matrix and targets created successfully.")
print(f"Number of features: {len(feature_cols)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Classification target shape: {y_class.shape}")
print(f"Regression target shape: {y_reg.shape}")

print("\nSelected features:")
print(feature_cols)

print("\nTarget distribution (is_hit):")
print(y_class.value_counts())

print("\nPopularity summary:")
print(y_reg.describe())

#============================================================
# Topic: Data Splitting
# Goal:
#   1. Divide the dataset into Training (80%) and Testing (20%) sets
#   2. Use 'stratify' for classification to maintain the rare hit ratio (2.6%)
#   3. Ensure reproducibility using a fixed random_state
# ============================================================

from sklearn.model_selection import train_test_split

# splitting for classification
# goal: train the model on 80% of the songs, and on the remaining 20% ​check whether it recognizes the hits
# use 'stratify' to avoid having no hits in the test due to their rarity
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# splitting for regression
# objective: Preparing data to predict the numerical value of 'popularity'
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

print("Data splitting complete.")
print(f"Classification Train/Test: {len(X_train_cls)} / {len(X_test_cls)}")
print(f"Regression Train/Test:     {len(X_train_reg)} / {len(X_test_reg)}")

# check hit ratio consistency
hit_ratio_test = (y_test_cls.sum() / len(y_test_cls)) * 100
print(f"Hit ratio in Test set: {hit_ratio_test:.2f}% (Consistent with master data)")

# ============================================================
# Topic: Regression Model & MLflow Tracking
# Goal:
#   1. Train a 'Linear Regression' baseline model to predict popularity
#   2. Track the experiment using MLflow (versioning and logging)
#   3. Evaluate the model using MAE, RMSE, and R2 metrics
# ============================================================

# MLflow setup
# goal: create a "log" of the experiment
# If change the model later, MLflow will allow us to compare the results
experiment_name = "Musical_Success_Regression"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="Baseline_Linear_Regression"):
    # target: popularity (0-100)
    # Objective: to test whether musical characteristics explain the state of popularity
    lr_model = LinearRegression()
    lr_model.fit(X_train_reg, y_train_reg)

    # Predictions
    y_pred_reg = lr_model.predict(X_test_reg)

    # Metrics calculation
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)

    # logging results (MLflow integration)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    print(" Linear Regression Results (Popularity Prediction):")
    print("-" * 50)
    print(f"   MAE (Mean Absolute Error): {mae:.2f} points")
    print(f"   RMSE (Root Mean Sq Error): {rmse:.2f} points")
    print(f"   R2 Score (Variance explained): {r2:.4f}")

    print("\n Results successfully logged to MLflow UI.")

"""### Observations: Linear Regression (Popularity Prediction)

The baseline Linear Regression model achieved an R2 score of only **0.0628**, meaning
that audio features alone explain less than **7% of the variance** in song popularity.

This is a **non-trivial and scientifically meaningful result**:

- It confirms that musical popularity is **not linearly predictable** from audio features.
- A song's success depends on factors **beyond its sound** — marketing, timing, artist
  fame, social media trends, and cultural context all play a role.
- The MAE of ~15 points on a 0-100 scale means the model's predictions are
  consistently "in the right ballpark" but far from precise.

**Conclusion:** Linear Regression provides a starting point for comparison. A more advanced model (Random Forest) is now tested to see if it can predict song popularity more accurately by capturing more complex patterns in the data.
"""

# ============================================================
# Topic: Classification Model & MLflow Tracking
# Goal:
#   1. Train a Random Forest Classifier to predict hit status (is_hit)
#   2. Handle class imbalance using class_weight='balanced'
#      (Only 2.6% of songs are hits — without balancing, the model
#       would simply predict "not a hit" for everything.)
#   3. Log the experiment to MLflow for comparison with future models
#   4. Evaluate using F1-score (more meaningful than accuracy for imbalanced data)
# ============================================================

experiment_name = "Musical_Success_Classification"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="RandomForest_Balanced_Classification"):

    # train Random Forest
    # class_weight='balanced' automatically gives more weight to rare hits
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_cls, y_train_cls)

    # predictions
    y_pred_cls = rf_model.predict(X_test_cls)

    # metrics
    acc = accuracy_score(y_test_cls, y_pred_cls)
    report = classification_report(y_test_cls, y_pred_cls, output_dict=True)
    f1_hit = report['1']['f1-score']
    precision_hit = report['1']['precision']
    recall_hit = report['1']['recall']

    # log to MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_hit", f1_hit)
    mlflow.log_metric("precision_hit", precision_hit)
    mlflow.log_metric("recall_hit", recall_hit)

    print("Random Forest Classification Results (Hit Prediction):")
    print("-" * 55)
    print(f"   Accuracy:           {acc:.4f}")
    print(f"   F1-Score (Hits):    {f1_hit:.4f}")
    print(f"   Precision (Hits):   {precision_hit:.4f}")
    print(f"   Recall (Hits):      {recall_hit:.4f}")
    print("\n" + classification_report(y_test_cls, y_pred_cls))
    print("Results successfully logged to MLflow.")

"""### Observations: The "Lazy Model" Trap (Classification)

The initial Random Forest model achieved a high Accuracy of **97.35%**, but a
near-zero F1-Score for hits (**0.0046**).

- The model fell into the **"Majority Class Trap"**: since hits are only 2.6% of
  the data, the model simply predicted "Not a Hit" for almost every song.
- It is 97% accurate but **100% useless** for finding hits.
- **Recall (0.0023):** Out of 429 actual hits, the model identified only **1**.

**Conclusion:** This result indicates that high accuracy can be misleading. While the model appears successful overall, it fails to identify the rare cases in practice. To address this, techniques like SMOTE will be used to help the model focus on the minority class by providing it with more examples for training.

### The "Lazy Model" Trap (Classification)

The **"Lazy Model" Trap** occurs when a dataset is highly imbalanced (e.g., 90% non-hits and only 10% hits).

*   **The Problem:** Instead of learning musical patterns, the model simply predicts "Non-Hit" for every single song.
*   **The Misleading Result:** The model achieves a high **Accuracy (90%)**, making it look successful on paper. However, it is "lazy" because it fails to identify a single real hit song.
*   **The Solution:** This is why accuracy is not enough. Should use metrics like **Precision...**, and techniques like **SMOTE** to force the model to actually learn the characteristics of the minority class (the hits).
"""

# ============================================================
# Topic: Handling Class Imbalance with SMOTE
# Goal:
#   1. Apply SMOTE to create a 50/50 balance between hits and non-hits
#   2. Re-train the model to see if it can finally "see" the hits
# ============================================================

# Import SMOTE (Synthetic Minority Over-sampling Technique)
# Used to balance the dataset by generating synthetic examples of the minority class
from imblearn.over_sampling import SMOTE

# data balancing
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_cls, y_train_cls)

print(f"Data balanced: {y_train_balanced.value_counts().to_dict()}")

# training a new model
with mlflow.start_run(run_name="RandomForest_SMOTE_Balanced"):
    rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_balanced, y_train_balanced)

    y_pred_smote = rf_smote.predict(X_test_cls)

    # results
    print("\n Random Forest + SMOTE Classification Results:")
    print("-" * 40)
    print(classification_report(y_test_cls, y_pred_smote))

    # logging new metrics into MLflow for comparison
    report = classification_report(y_test_cls, y_pred_smote, output_dict=True)
    mlflow.log_metric("f1_hit_smote", report['1']['f1-score'])
    mlflow.log_metric("recall_hit_smote", report['1']['recall'])

print("SMOTE experiment logged to MLflow UI.")

"""### Observations: SMOTE and Hit Detection

Applying SMOTE significantly improved the model's ability to detect hits.

- Recall increased from almost zero to approximately 12%.
- The model is now capable of identifying some hit songs.
- Accuracy decreased slightly because the model is no longer predicting
the majority class only.

Even after balancing the data, hit prediction
remains difficult.

This suggests that audio features alone are insufficient to fully explain
musical success. External factors such as artist popularity, marketing,
social influence, and cultural trends likely play a major role.

Therefore, musical success appears to be only partially encoded in the
mathematical structure of the music itself.
"""

# ============================================================
# Topic: Feature Importance Analysis
# Goal:
#   1. Identify which musical/mathematical features have the most
#      influence on the model's decisions
#   2. Compare the importance of engineered features (PC1, Mood Index, etc.)
#      against raw audio features (Tempo, Energy, etc.)
#   3. Visualize the "Secret Formula" for success according to the model
# ============================================================

# get importance from the Random Forest (SMOTE version) model
importances = rf_smote.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# visualization
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, hue='feature', palette='viridis', legend=False)
plt.title('The "Secret Code" of Music: Feature Importance for Hit Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Musical Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------------
# LOG TO MLFLOW
# ------------------------------------------------------------
# logging the top features to MLflow effectively "documents"
top_3_features = feature_importance_df['feature'].iloc[:3].tolist()
mlflow.log_param("top_features", ", ".join(top_3_features))

print("Top 5 Most Influential Musical Features:")
print("-" * 40)
for i, row in feature_importance_df.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")

print("\n Feature importance visualization complete and logged.")

"""###  Interpretation of the "Musical Secret Code" (Feature Importance)

**Instrumentalness** is the single most important feature (0.14).
- **Finding:** A high score here typically correlates with a **decreased** chance of being a mainstream hit.
- **Highlight:** Modern hits are almost exclusively vocal-centric. The model uses this feature as a primary filter to distinguish between "background/niche" music and "mainstream" potential.

**Acousticness** ranks 2nd (0.10).
- **Finding:** The mathematical dominance of this feature suggests a strong divide between traditional and modern successful tracks.
- **Highlight:** Most successful contemporary hits lean towards synthesized and highly produced sounds rather than raw acoustic instruments.

**'Cluster'** feature ranks 3rd (0.07), outperforming raw metrics like Tempo or Danceability.
- **Highlight:** It proves that the "higher-level" mathematical groupings created in Notebook 2 capture a more significant "essence" of what makes a song a hit than simple individual audio metrics.

The Constraints of Success (**Duration** & **Energy**)
- **Duration_ms:** Songs that are too long or too short rarely break into the "hit" category.
- **Energy:** While important, it is not the top driver. This suggests that while a hit needs energy, it is the instrumentalness/acousticness that defines its success more than its raw power.

Low Importance: **Danceability** & **Tempo**
Contrary to popular belief, **Danceability** and **Tempo** are relatively low on the list.
- **Conclusion:** You cannot make a hit just by making it "fast" or "danceable." These are baseline requirements for many genres, but they are not the *defining* characteristics that separate a hit from a regular song.

Moving from basic physics (like Tempo) to more advanced mathematical groups (Clusters) was a key step in this analysis. The model shows that musical success depends on the overall texture and feel of the sound—such as whether it is vocal-focused or electronic—rather than just having a "catchy" beat.

### What is a "Mainstream Hit"?

In this analysis, a **Mainstream Hit** is defined not just by musical quality, but by its widespread commercial success and cultural visibility.

-  **A track** that ranks high on global charts (like the Billboard Hot 100) and maintains a high popularity score on platforms like Spotify.
- **Characteristics:** These songs are played across diverse radio stations, playlists, and public venues, appealing to a broad, general audience rather than a specific niche.
- **The "Hit Formula":** From a data perspective, most hits follow similar patterns. They often have a specific level of loudness, very clean studio production, and rhythms that are mathematically designed to be easy for a wide audience to enjoy.
"""

# ============================================================
# Topic: Model Error Analysis (Confusion Matrix)
# ============================================================

# Import tools for evaluating and visualizing model performance
# confusion_matrix: calculates how many predictions were correct/incorrect per class
# ConfusionMatrixDisplay: creates a visual representation of the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# matrix calculation
cm = confusion_matrix(y_test_cls, y_pred_smote)

# clean drawing without hidden shapes
# create the confusion matrix display object with readable class labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Hit', 'Hit'])
# set up the figure size for a clean and readable visualization
fig, ax = plt.subplots(figsize=(7, 5))
# plot the confusion matrix using a blue color scale
# values_format='d' displays whole numbers instead of scientific notation
disp.plot(cmap='Blues', values_format='d', ax=ax)
# add a descriptive title to the plot
ax.set_title('Confusion Matrix: Hit Prediction (SMOTE Model)')
# remove the grid lines for a cleaner visual appearance
ax.grid(False)

# save BEFORE plt.show()
plt.savefig("confusion_matrix.png", bbox_inches='tight')

# show and clear
plt.show()
plt.close()

# log to MLflow
mlflow.log_artifact("confusion_matrix.png")
mlflow.end_run()

print("\nConfusion Matrix generated")
print(f" - Correctly identified non-hits: {cm[0,0]}")
print(f" - False Alarms (Predicted hit, but isn't): {cm[0,1]}")
print(f" - Missed Hits (Actual hit, but missed): {cm[1,0]}")
print(f" - Correctly identified hits: {cm[1,1]}")

"""### Final Evaluation: Confusion Matrix Analysis

The Confusion Matrix provides a detailed picture of the model's performance. It reveals the balance between correctly identified hits and "false alarms," providing a final evaluation of the model's predictive accuracy.

#### Summary of Results:
"""

# create a visual summary table of the confusion matrix results
import matplotlib.pyplot as plt

# set up a wide, short figure since the table does not need much vertical space
fig, ax = plt.subplots(figsize=(8, 2))
# hide the axes — only the table should be visible
ax.axis('off')

# define the table content with labels and prediction counts
# each cell shows the count and a short description of what it represents
table_data = [
    ['', 'Predicted: Not Hit', 'Predicted: Hit'],
    ['Actual: Not Hit', '15,132  [OK]', '708  [FALSE ALARM]'],
    ['Actual: Hit',     '376  [MISSED]', '53  [DETECTED]']
]

# render the table in the center of the figure
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
# adjust cell width and height for better readability
table.scale(1.5, 2)

# apply color coding to each cell for quick visual interpretation
table[1, 1].set_facecolor('#d4edda')  # green - correct
table[1, 2].set_facecolor('#fff3cd')  # yellow - false alarm
table[2, 1].set_facecolor('#f8d7da')  # red - missed
table[2, 2].set_facecolor('#d4edda')  # green - detected

plt.tight_layout()
plt.show()

"""#### Key Insights:

**1. The SMOTE :**
By applying the SMOTE balancing technique, the model successfully identified **53 real hits**.
In baseline model (Cell 6), this count was practically zero.
This shows that the mathematical model successfully captured some fundamental characteristics of musical success.

**2. False Alarms:**
The model flagged **708 songs** as potential hits that did not ultimately reach the charts.
Тhis works like a professional talent scout. Instead of listening to thousands of songs, the model acts as a filter that selects a small group of "promising" tracks for further review.

**3. The Unexpected Hits (Missed Opportunities):**
376 hits remained undetected, which is a significant finding. This demonstrates that success does not always follow mathematical logic. These numbers explain why the initial analysis showed low results—because many songs become hits for reasons that data alone cannot capture.

#### Scientific Conclusion:

The results of the Confusion Matrix, combined with the Feature Importance analysis,
lead to a **conclusion**:

> The mathematical fingerprint of a song is only one piece of the puzzle.
> While success is influenced by audio parameters, it is largely determined
> by factors external to the sound signal: artist branding, marketing strategy,
> cultural timing, and social trends.

This model serves as a **mathematical filter** — it identifies potential, but cannot
guarantee a hit, as music is as much a **social phenomenon** as it is a mathematical one.
"""

# ============================================================
# Topic: Classification Metrics - ROC Curve and AUC
# Goal:
#   1. Evaluate the model's ability to distinguish classes at
#      different thresholds
#   2. Log the AUC (Area Under Curve) metric to MLflow
# ============================================================

# Import metrics to evaluate the model's ability to distinguish between classes
# roc_curve: calculates the true positive and false positive rates for various thresholds
# roc_auc_score: calculates the Area Under the Curve, providing a single score for model quality
from sklearn.metrics import roc_curve, roc_auc_score

# get probabilities for the positive class (Hit)
y_probs = rf_smote.predict_proba(X_test_cls)[:, 1]

# calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test_cls, y_probs)
auc_score = roc_auc_score(y_test_cls, y_probs)

# visualization
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) - Hit Prediction')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# ------------------------------------------------------------
# lOG TO MLFLOW
# ------------------------------------------------------------
# log the AUC score as a key performance metric
mlflow.log_metric("auc_score", auc_score)

print(f"ROC-AUC score: {auc_score:.4f}")
print("ROC Curve plotted and AUC logged to MLflow.")

"""### Classification Metric: ROC-AUC Analysis

The **ROC-AUC** score of 0.7048 provides a deeper insight into the model's ability to distinguish between hits and non-hits. This metric helps us understand the model's quality much better than simple accuracy, as it measures how successfully it separates the two groups from one another.

#### Key Interpretation:

- **Better than random:** An AUC of ~0.70 is considered "fair to good" in predictive modeling. It shows that the model is **70% more likely** to rank a real hit higher than a non-hit song based purely on its mathematical features.
- **The "Ranking" Capability:** While the Confusion Matrix showed many "False Alarms," the ROC curve tells us that those alarms aren't completely random. The model correctly identifies the *direction* of success, even if it struggles with the exact *threshold* of becoming a hit.
- **Validation:** This result reinforces hypothesis: there is a **mathematical structure to popularity**. If success were purely random or purely marketing-based, the AUC would be 0.50 (the diagonal line). Being at 0.70 confirms that about **20% of the predictive "signal"** is indeed found in the audio signal itself.

#### Conclusion:
The model demonstrates a robust ability to distinguish "hit potential." The distance between orange curve and the dashed diagonal line represents the **"Value Added" by Machine Learning** to the musical analysis.

It can be confidently stated that while hits cannot be predicted perfectly, they can be effectively filtered with 70% reliability using the mathematical fingerprint of the sound.
"""

# convert the column names to a list and join them with newlines for a vertical display
# this allows for a quick and clear inspection of the final feature set
print('\n'.join(df_model.columns.tolist()))

"""## Modeling & MLflow — Summary

### Performed Analysis

This notebook applied three machine learning methods to answer the
research question: **"Can the mathematical structure of music predict success?"**

### Methods Applied

**1. Linear Regression**
Predicted the popularity score of a song from its audio features.
Result: R² = 0.0628 — audio features explain only 6% of the variance in popularity.

**2. Random Forest Classification (Baseline)**
Classified songs as "hits" or "non-hits" using raw class distribution.
Result: The model predicted almost no hits due to severe class imbalance (97:3 ratio).

**3. Random Forest Classification (SMOTE)**
Applied SMOTE to balance the training data before classification.
Result: Hit detection improved significantly — 53 real hits identified.

### Key Findings

- Instrumentalness and Acousticness are the most important predictors of hit potential.
- Our engineered feature "cluster" ranked 3rd in importance, validating the
feature engineering work done in ***math_music_eda_feature_engineering.ipynb***.
- Danceability and Tempo — contrary to popular belief — are among the least
important features for predicting success.

### Conclusion

> The mathematical fingerprint of music is a weak but real predictor of success.
> Audio features alone explain only a small fraction of what makes a song a hit.
> The rest belongs to the human world: marketing, culture, timing, and social trends.

### MLflow Tracking

All three experiments were logged to MLflow with full parameter and metric tracking.
"""