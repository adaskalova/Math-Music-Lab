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
- **Hypothesis Testing**: Using Pearson, ANOVA, and Spearman tests to prove musical patterns.
- **Physical vs. Perceptual**: Proving the link between loudness (dB) and energy.
- **Genre Profiling**: Statistical proof that genres have distinct mathematical "fingerprints".
- **Visual Evidence**: Regplots, Confidence Intervals, and LOWESS curves.
---

## Statistical Methodology

The project applies the following tests to validate musical theories:
1. **Pearson Correlation**: To measure the linear relationship between loudness and energy ($r=0.76$).
2. **One-Way ANOVA**: To verify if the variance in energy levels between genres is statistically significant.
3. **Spearman Rank Correlation**: Used with **LOWESS curves** to investigate non-linear relationships between tempo and danceability.
4. **95% Confidence Intervals**: Visualized to ensure comparisons between genres are accurate and not due to random noise.

## Data Sources

This project utilizes data from two independent sources to ensure a comprehensive analysis:
1. **Spotify Tracks Dataset (Kaggle)**: Comprehensive audio features for 114k+ songs across 125 genres.
2. **Music Physics Data (Theoretical)**: Generated waveforms and intervals based on acoustic physics principles.

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
