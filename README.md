# 🎵 Mathematics in Music

A project exploring the deep mathematical foundations of music — from the physics of sound to musical scales, intervals, and harmony.

---

## 📖 What is this project about?

Music is not just art — it is mathematics. Every note, chord, and scale follows precise mathematical rules. This project investigates those rules using Python, visualizations, and audio examples.

---

## 📂 Project Structure


---

## 📚 Contents by Part

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

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Main programming language |
| NumPy | Mathematical calculations and signal generation |
| Matplotlib | Visualizations and plots |
| SciPy | Fourier analysis (FFT) |
| wave (stdlib) | Saving audio as .wav files |
| Jupyter Notebook | Interactive code + explanations |

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Math-Music-Lab.git
   cd Math-Music-Lab
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy jupyter
   ```

3. Open the notebooks:
   ```bash
   jupyter notebook
   ```

4. Run the notebooks in order: Part 1 → Part 2 → Part 3

---

## 🔢 Key Mathematical Concepts

| Concept | Formula |
|---------|---------|
| Sine wave | `A(t) = A · sin(2πft + φ)` |
| Octave | `f_octave = 2 · f` |
| Equal temperament | `f_n = f₀ · 2^(n/12)` |
| Beating frequency | f_beat = &#124;f₁ - f₂&#124; |
| Cents deviation | `cents = 1200 · log₂(f₁/f₂)` |
| Pythagorean fifth | `f = f₀ · (3/2)^k · (1/2)^m` |

---

## 🎯 Project Goals

- Demonstrate the mathematical structure behind music
- Connect abstract math (ratios, logarithms, modular arithmetic) to real sound
- Generate audio examples that can be heard, not just seen
- Present findings clearly through code, plots, and explanations

---
