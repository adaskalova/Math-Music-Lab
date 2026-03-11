# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Base frequency
A4 = 440  # Hz

# Pythagorean tuning ratios
pythagorean_ratios = {
    'C': 1.0,
    'C#': 256/243,  # Limma
    'D': 9/8,
    'D#': 32/27,
    'E': 81/64,
    'F': 4/3,
    'F#': 729/512,  # Apotome
    'G': 3/2,
    'G#': 128/81,
    'A': 27/16,
    'A#': 16/9,
    'B': 243/128
}

# Equal temperament frequencies
equal_temp = {note: A4 * 2**(i/12) for i, note in enumerate(pythagorean_ratios)}

plt.figure(figsize=(14, 7))

# Calculate frequencies
notes = list(pythagorean_ratios.keys())
pyth_freqs = [A4 * ratio for ratio in pythagorean_ratios.values()]
equal_freqs = [equal_temp[note] for note in notes]

# Plot
x = np.arange(len(notes))
plt.bar(x - 0.2, pyth_freqs, width=0.4, label='Pythagorean', color='#3498db')
plt.bar(x + 0.2, equal_freqs, width=0.4, label='Equal Temperament', color='#e74c3c')

plt.xticks(x, notes)
plt.title('Frequency Comparison: Pythagorean vs Equal Temperament')
plt.xlabel('Musical Notes')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

def cents_diff(f1, f2):
    return 1200 * np.log2(f1 / f2) if f1 != f2 else 0

deviations = [cents_diff(p, e) for p, e in zip(pyth_freqs, equal_freqs)]

plt.figure(figsize=(14, 6))
plt.bar(notes, deviations, color=['#27ae60' if abs(d) < 10 else '#e74c3c' for d in deviations])
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Cents Deviation: Pythagorean from Equal Temperament')
plt.xlabel('Notes')
plt.ylabel('Deviation (cents)')
plt.grid(True, alpha=0.3)
plt.show()

def transpose(note, semitones):
    note_index = notes.index(note)
    new_index = (note_index + semitones) % 12
    return notes[new_index]

# Example: Transpose C by 7 semitones
print(f"Transposing C by 7 semitones: {transpose('C', 7)}")  # Should be G
print(f"Transposing E by 5 semitones: {transpose('E', 5)}")  # Should be A
