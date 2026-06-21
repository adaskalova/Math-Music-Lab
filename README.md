# Mathematics in Music

A project exploring the deep mathematical foundations of music — from the physics of sound to musical scales, intervals, and harmony.

---

## What is this project about?

Music is not just art — it is mathematics. Every note, chord, and scale follows precise mathematical rules. This project investigates those rules using Python, visualizations, and audio examples.

---

## Contents by Part

### Part 1 – Physics of Sound
- What is a sound wave?
- The sine wave formula: `A(t) = A · sin(2πft + φ)`
- How **frequency** determines pitch and **amplitude** determines loudness
- The **beating phenomenon**: what happens when two close frequencies play together
- Generating and saving real `.wav` audio files

### Part 2 – Musical Scales & Tuning Systems
- **Pythagorean tuning** — built from perfect fifths (ratio 3:2)
- **Equal temperament** — divides the octave into 12 equal semitones: `f_n = f₀ × 2^(n/12)`
- Why Western music uses exactly **12 notes** (mathematical proof)
- **Cents deviation** — measuring the difference between tuning systems
- **Modular arithmetic** for transposing notes between keys

### Part 3 – Intervals & Consonance
- What is **consonance** and why some note combinations sound pleasant
- What is **beating** and how it causes dissonance
- **Helmholtz's theory** — consonance = absence of beating between harmonics
- Visualizing waveforms of 6 key intervals (Unison, Minor Second, Major Third, Perfect Fifth, Tritone, Octave)
- **Consonance spectrum** — ranking all 13 intervals from most to least consonant

### Part 4 – Empirical Analysis Using Real-World Data
- **Data Integration**: Merging and cleaning a dataset of 114,000+ tracks.
- **Hypothesis Testing**: Applying statistical methods (Pearson, ANOVA, Spearman) to verify observed patterns in the data.
- **Physical vs. Perceptual**: Proving the link between loudness (dB) and energy.
- **Genre Profiling**: Demonstrating that different music genres have statistically distinct characteristics.
- **Visual Evidence**: Supporting results with clear visualizations such as regression plots, confidence intervals, and non-linear trend curves (LOWESS).
---

## Statistical Methodology

The project applies the following tests to validate musical theories:

1. **Pearson Correlation**: To measure the linear relationship between loudness and energy ($r=0.76$).
	- **Why:** Both variables are continuous and expected to have a linear relationship, making Pearson correlation the most appropriate metric.

2. **One-Way ANOVA**: To verify if the variance in energy levels between genres is statistically significant.
	- **Why:** This method compares the mean values of a continuous variable (energy) across multiple categorical groups (genres), which is exactly the purpose of ANOVA.

3. **Spearman Rank Correlation**: Used with **LOWESS curves** to investigate non-linear relationships between tempo and danceability.
	- **Why:** The relationship is not strictly linear, and Spearman captures monotonic trends. LOWESS helps visualize potential non-linear patterns in the data.

4. **95% Confidence Intervals**: Visualized to ensure comparisons between genres are accurate and not due to random noise.
	- **Why:** Confidence intervals provide a measure of uncertainty and reliability, helping interpret whether observed differences are statistically meaningful.
	

### Machine Learning: The Mathematical Fingerprint of Musical Success

This part answers two questions: *Does what makes a song a "hit" have a stable
mathematical signature?* and *Can a model trained on past data recognize today's hits?*

**1. Data Acquisition & Cleaning** (`math_music_data_cleaning`)
- Programmatic download of three sources (Spotify, CORGIS, Billboard Hot 100).
- Handling missing values, removing duplicates, standardizing song/artist names.
- Merging Spotify + Billboard into a `master_dataset` with the target `is_hit`.
- CORGIS set aside as a historical dataset for time series analysis.

**2. EDA & Feature Engineering** (`math_music_eda_feature_engineering`)
- Exploratory analysis: structure, missing values, correlation matrix.
- Engineered features: `energy_loudness_ratio` and `mood_index`.
- **Dimensionality Reduction (PCA)** — scree plot and 2D projection.
- **Clustering (K-Means)** with the Elbow method.
- **Time Series** analysis of loudness and tempo across decades ("Loudness War").

**3. Modeling & MLflow** (`math_music_modeling_and_mlflow`)
- **Linear Regression** to predict popularity.
- **Random Forest** classification for hit prediction (baseline + **SMOTE** for imbalance).
- **Feature Importance** — Instrumentalness, Acousticness and the engineered `cluster`
  emerged as the strongest predictors.
- Evaluation: Confusion Matrix, F1-score, **ROC-AUC**.
- Full experiment tracking with **MLflow** (params, metrics, artifacts).
---

## Data Sources

This project utilizes two independent and publicly accessible data sources:
1. **Spotify Tracks Dataset (Kaggle)**:
	- **Content:** A large-scale collection of over 114,000 tracks.
	- **Focus:** Detailed audio features such as energy, tempo, danceability, and loudness.
	- **Status:** Publicly available and free to use for research purposes.
2. **CORGIS Music Dataset (Educational Repository)**:
	- **Content:** A separate collection of 10,000 tracks.
	- **Focus:** Мetadata including artist popularity, release year, and album details.
	- **Status:** Free educational resource provided by the CORGIS Educational Repository.

**Different Origin:** The datasets were curated by different organizations for different purposes (one for general research, the other specifically for educational analysis).

**Diverse Scope:** The Kaggle dataset provides deep "technical/physical" audio metrics, while the CORGIS dataset offers "historical/popularity" metadata.

**Accessibility:**
Both datasets are publicly available and free to access. The [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) is available on Kaggle, and the music metadata dataset is provided by the [CORGIS Educational Repository](https://corgis-edu.github.io/corgis/csv/music/). No proprietary or paid data sources were required.

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Main programming language |
| NumPy | Mathematical calculations and signal generation |
| Matplotlib | Visualizations and plots |
| SciPy | Fourier analysis (FFT) |
| wave (stdlib) | Saving audio as .wav files |
| Jupyter Notebook | Interactive code + explanations |
|----|----|
| Pandas | Data manipulation and cleaning |
| Seaborn | Advanced statistical visualizations |
| SciPy (stats) | Hypothesis testing (Pearson, ANOVA, Spearman) |
| Statsmodels | LOWESS non-linear trend analysis |
| Scikit-learn | ML models (Linear Regression, Random Forest, PCA, K-Means) |
| imbalanced-learn (SMOTE) | Handling class imbalance for hit prediction |
| MLflow | Experiment tracking, metrics and model versioning |
| Google Colab | Cloud environment where the ML notebooks were developed |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Math-Music-Lab.git
   cd Math-Music-Lab
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy jupyter
   ```

3. Run the Python modules in `/src`:
   ```bash
   cd src
   python math_music_part1.py
   python math_music_part2.py
   python math_music_part3.py
   ```
   > Each script can be run independently. Output audio files will be saved in the `audio/` folder.

4. Open the Jupyter notebooks in `/notebooks`:
   ```bash
   cd ../notebooks
   jupyter notebook
   ```

5. Run the notebooks in order:
   - `math_music_part1.ipynb` — Part 1: Physics of Sound
   - `math_music_part2.ipynb` — Part 2: Musical Scales
   - `math_music_part3.ipynb` — Part 3: Intervals & Consonance

---

## Key Mathematical Concepts

| Concept | Formula |
|---------|---------|
| Sine wave | `A(t) = A · sin(2πft + φ)` |
| Octave | `f_octave = 2 · f` |
| Equal temperament | `f_n = f₀ · 2^(n/12)` |
| Beating frequency | f_beat = &#124;f₁ - f₂&#124; |
| Cents deviation | `cents = 1200 · log₂(f₁/f₂)` |
| Pythagorean fifth | `f = f₀ · (3/2)^k · (1/2)^m` |

---

## Project Goals

- Demonstrate the mathematical structure behind music
- Connect abstract math (ratios, logarithms, modular arithmetic) to real sound
- Generate audio examples that can be heard, not just seen
- Present findings clearly through code, plots, and explanations

---
