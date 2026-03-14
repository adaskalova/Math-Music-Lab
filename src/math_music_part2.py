# -*- coding: utf-8 -*-

# For numerical operations
import numpy as np
# For data visualization
import matplotlib.pyplot as plt

# Base frequency reference (A4 standard pitch)
A4 = 440  # Hz (International concert pitch standard)

# Pythagorean tuning ratios based on perfect fifths (3:2 ratios)
# Uses pure intervals derived from harmonic series
pythagorean_ratios = {
    'C': 1.0, # Unison (reference note)
    'C#': 256/243,  # Limma (Pythagorean minor semitone)
    'D': 9/8, # Major whole tone
    'D#': 32/27, # Minor third approximation
    'E': 81/64, # Major third approximation
    'F': 4/3, # Perfect fourth
    'F#': 729/512,  # Apotome (Pythagorean major semitone)
    'G': 3/2, # Perfect fifth (basis of the tuning)
    'G#': 128/81, # Minor sixth approximation
    'A': 27/16, # Major sixth approximation
    'A#': 16/9, # Minor seventh
    'B': 243/128 # Major seventh
}

# Calculate Equal Temperament frequencies (modern standard)
# Divides octave into 12 equal logarithmic intervals
equal_temp = {note: A4 * 2**(i/12) for i, note in enumerate(pythagorean_ratios)}

# Create figure for frequency comparison
plt.figure(figsize=(14, 7))

# Prepare data for plotting
notes = list(pythagorean_ratios.keys()) # Musical note names
pyth_freqs = [A4 * ratio for ratio in pythagorean_ratios.values()] # Pythagorean frequencies
equal_freqs = [equal_temp[note] for note in notes] # Equal temperament frequencies

# Create bar positions
x = np.arange(len(notes))
# Plot Pythagorean frequencies in blue
plt.bar(x - 0.2, pyth_freqs, width=0.4, label='Pythagorean', color='#3498db')
# Plot Equal Temperament frequencies in red
plt.bar(x + 0.2, equal_freqs, width=0.4, label='Equal Temperament', color='#e74c3c')

# Configure plot appearance
plt.xticks(x, notes) # Set x-axis labels as note names
plt.title('Frequency Comparison: Pythagorean vs Equal Temperament')
plt.xlabel('Musical Notes')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.grid(True, alpha=0.3) # Add grid with low opacity
plt.tight_layout() # Adjust layout
plt.show()

# Function to calculate cents difference between frequencies
# Cents = 1200 × log₂(f₁/f₂) - standard unit for musical intervals
def cents_diff(f1, f2):
    """Calculate interval between two frequencies in cents"""
    return 1200 * np.log2(f1 / f2) if f1 != f2 else 0 # Avoid division by zero

# Calculate deviations between Pythagorean and Equal Temperament
deviations = [cents_diff(p, e) for p, e in zip(pyth_freqs, equal_freqs)]

# Create plot for deviations
plt.figure(figsize=(14, 6))
# Color coding: green for small deviations (<10 cents), red for larger ones
plt.bar(notes, deviations, color=['#27ae60' if abs(d) < 10 else '#e74c3c' for d in deviations])
plt.axhline(0, color='black', linewidth=0.8) # Reference line at zero
plt.title('Cents Deviation: Pythagorean from Equal Temperament')
plt.xlabel('Notes')
plt.ylabel('Deviation (cents)')
plt.grid(True, alpha=0.3)
plt.show()

# Transposition function using chromatic circle
def transpose(note, semitones):
    """
        Transpose a note by specified number of semitones

        Parameters:
        note (str): Starting note (e.g., 'C')
        semitones (int): Number of semitones to transpose

        Returns:
        str: Transposed note name
    """
    note_index = notes.index(note) # Get index of starting note
    new_index = (note_index + semitones) % 12 # Calculate new position modulo 12
    return notes[new_index] # Return transposed note

# Example: Transpose C by 7 semitones
print(f"Transposing C by 7 semitones: {transpose('C', 7)}")  # Perfect fifth (C → G)
print(f"Transposing E by 5 semitones: {transpose('E', 5)}")  # Perfect fourth (E → A)
