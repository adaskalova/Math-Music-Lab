# -*- coding: utf-8 -*-

# Import required libraries for math, plotting, and audio
import numpy as np
import matplotlib.pyplot as plt
import wave

# Define musical intervals using Just Intonation frequency ratios (simple integer fractions)
intervals = {
    'Unison':          1/1,
    'Minor Second':    16/15,
    'Major Second':    9/8,
    'Minor Third':     6/5,
    'Major Third':     5/4,
    'Perfect Fourth':  4/3,
    'Tritone':         45/32,
    'Perfect Fifth':   3/2,
    'Minor Sixth':     8/5,
    'Major Sixth':     5/3,
    'Minor Seventh':   16/9,
    'Major Seventh':   15/8,
    'Octave':          2/1
}

# Set the base frequency (A4 = 440 Hz) and audio sample rate
BASE_FREQ = 440    # A4 in Hz
SAMPLE_RATE = 44100

# Function to generate two sine wave tones forming a musical interval
def generate_interval(base_freq=440, ratio=1.0, duration=2.0):
    """Generate two tones forming a musical interval."""
    # Create a time array from 0 to duration with enough samples
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Generate the first tone as a sine wave at the base frequency
    tone1 = 0.5 * np.sin(2 * np.pi * base_freq * t)
    # Generate the second tone at base_freq multiplied by the interval ratio
    tone2 = 0.5 * np.sin(2 * np.pi * base_freq * ratio * t)
    # Return both tones and their sum (combined sound)
    return t, tone1, tone2, tone1 + tone2

# Print a table showing each interval's ratio, frequencies, and beating frequency
print(f"{'Interval':<20} {'Ratio':>10} {'Freq 1 (Hz)':>12} {'Freq 2 (Hz)':>12} {'Beat (Hz)':>10}")
print('-' * 68)
for name, ratio in intervals.items():
    # Calculate the second frequency by multiplying base by ratio
    f2 = BASE_FREQ * ratio
    # Beating frequency = absolute difference between the two frequencies
    beat = abs(BASE_FREQ - f2)
    print(f"{name:<20} {ratio:>10.4f} {BASE_FREQ:>12.2f} {f2:>12.2f} {beat:>10.2f}")

"""## 6. Visualizing Intervals"""

# Select 6 representative intervals to visualize
demo = ['Unison', 'Minor Second', 'Major Third', 'Perfect Fifth', 'Tritone', 'Octave']
# Create a 3x2 grid of subplots, one for each interval
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Part 3: Musical Intervals - Waveform Visualization', fontsize=16, fontweight='bold')
# Loop through each selected interval and plot its waveforms
for i, name in enumerate(demo):
    ratio = intervals[name]
    # Generate the two tones and their combination for a short 0.05s window
    t, tone1, tone2, combined = generate_interval(BASE_FREQ, ratio, duration=0.05)
    # Select the correct subplot position
    ax = axes[i // 2][i % 2]
    # Plot tone 1 in blue
    ax.plot(t[:500], tone1[:500], color='#3498db', alpha=0.7, label=f'Tone 1: {BASE_FREQ} Hz')
    # Plot tone 2 in red
    ax.plot(t[:500], tone2[:500], color='#e74c3c', alpha=0.7, label=f'Tone 2: {BASE_FREQ*ratio:.1f} Hz')
    # Plot the combined waveform in dark color
    ax.plot(t[:500], combined[:500], color='#2c3e50', lw=1.5, label='Combined')
    ax.set_title(f'{name}  (ratio: {ratio:.4f})', fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""## 7. Consonance Spectrum"""

# Define consonance levels for each interval (higher = more consonant)
# These values are based on psychoacoustic research and music theory
consonance_levels = {
    'Unison':         0.99,
    'Octave':         0.95,
    'Perfect Fifth':  0.90,
    'Perfect Fourth': 0.85,
    'Major Sixth':    0.80,
    'Major Third':    0.75,
    'Minor Sixth':    0.70,
    'Minor Third':    0.65,
    'Major Second':   0.50,
    'Tritone':        0.35,
    'Minor Second':   0.30,
    'Major Seventh':  0.25,
    'Minor Seventh':  0.20
}
# Sort intervals from most to least consonant for the bar chart
sorted_cons = sorted(consonance_levels.items(), key=lambda x: x[1], reverse=True)
names = [k for k, v in sorted_cons]
values = [v for k, v in sorted_cons]
# Assign colors: green = consonant, yellow = neutral, red = dissonant
colors = ['#2ecc71' if v > 0.7 else '#f1c40f' if v > 0.4 else '#e74c3c' for v in values]

# Plot the consonance spectrum as a bar chart
plt.figure(figsize=(14, 7))
plt.bar(names, values, color=colors, alpha=0.85)
plt.title('Consonance Spectrum of Musical Intervals', fontsize=15, fontweight='bold')
plt.ylabel('Consonance Level (0-1)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
# Add horizontal threshold lines for consonance and dissonance
plt.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Consonance threshold')
plt.axhline(0.4, color='orange', linestyle='--', alpha=0.5, label='Dissonance threshold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

"""## 8. Audio Examples"""

# Function to save a numpy audio signal as a .wav file
def save_wav(filename, signal, sample_rate=44100):
    """Save a numpy signal as a .wav file."""
    # Normalize signal to 16-bit integer range (-32767 to 32767)
    signal_int = np.int16(signal / np.max(np.abs(signal)) * 32767)
    # Open a new .wav file for writing
    with wave.open(filename, 'w') as wf:
        # Set mono channel (1 channel)
        wf.setnchannels(1)
        # Set sample width to 2 bytes (16-bit audio)
        wf.setsampwidth(2)
        # Set the sample rate (44100 Hz = CD quality)
        wf.setframerate(sample_rate)
        # Write the audio data as bytes
        wf.writeframes(signal_int.tobytes())

# List of intervals to generate audio for
audio_intervals = ['Unison', 'Minor Second', 'Major Third', 'Perfect Fifth', 'Tritone', 'Octave']

print('Generating audio files...')
for name in audio_intervals:
    ratio = intervals[name]
    # Generate 3 seconds of the combined interval sound
    _, _, _, combined = generate_interval(BASE_FREQ, ratio, duration=3.0)
    # Build a filename from the interval name (replace spaces with underscores)
    filename = f"interval_{name.replace(' ', '_')}.wav"
    # Save the audio to disk
    save_wav(filename, combined)
    print(f'  Saved: {filename}')
print('Done!')

"""## 9. Mathematical Analysis"""

# Analyze the beating frequency for each interval and classify the perception
print('Beating Frequency Analysis')
print('=' * 60)
print(f"{'Interval':<20} {'Ratio':>8} {'Beat (Hz)':>10} {'Perception':>15}")
print('-' * 60)

for name, ratio in intervals.items():
    # Calculate the second frequency
    f2 = BASE_FREQ * ratio
    # Beating frequency = absolute difference between the two tones
    beat = abs(BASE_FREQ - f2)
    # Classify the beating perception based on frequency range
    if beat < 5:
        perception = 'Smooth'        # Very slow beating, sounds pleasant
    elif beat < 30:
        perception = 'Rough'         # Noticeable roughness, mildly dissonant
    elif beat < 100:
        perception = 'Very rough'    # Fast beating, strongly dissonant
    else:
        perception = 'Fused/Smooth'  # Beating too fast to perceive, sounds fused
    print(f"{name:<20} {ratio:>8.4f} {beat:>10.2f} {perception:>15}")
