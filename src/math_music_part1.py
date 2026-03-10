# -*- coding: utf-8 -*-
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("✅ Libraries imported successfully")

# Parameters for A4 note
frequency = 440  # Hz
amplitude = 1.0
duration = 0.01  # 10 milliseconds for visualization
sample_rate = 44100  # Standard audio sample rate

# Generate time axis
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate sine wave: y(t) = A * sin(2πft)
y = amplitude * np.sin(2 * np.pi * frequency * t)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(t * 1000, y, color='#2196F3', linewidth=2)
plt.title('Pure Tone: A4 = 440 Hz', fontsize=14, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate period
period = 1 / frequency
print(f"Period of A4: T = 1/{frequency} = {period*1000:.4f} ms")

# Compare different frequencies
frequencies = [261.63, 440, 880]
colors = ['#4CAF50', '#2196F3', '#F44336']
labels = ['C4 (261.63 Hz)', 'A4 (440 Hz)', 'A5 (880 Hz)']

plt.figure(figsize=(14, 5))

for freq, color, label in zip(frequencies, colors, labels):
    y_temp = np.sin(2 * np.pi * freq * t)
    plt.plot(t * 1000, y_temp, color=color, linewidth=1.5, label=label, alpha=0.8)

plt.title('Comparison of Different Frequencies', fontsize=14, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend(fontsize=11, loc='upper right')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Observation: Higher frequency → more cycles in the same time period → higher pitch")

# Compare different amplitudes
amplitudes = [0.3, 0.7, 1.0]
colors_amp = ['#FF9800', '#9C27B0', '#2196F3']

plt.figure(figsize=(14, 5))

for amp, color in zip(amplitudes, colors_amp):
    y_temp = amp * np.sin(2 * np.pi * 440 * t)
    plt.plot(t * 1000, y_temp, color=color, linewidth=1.5, label=f'Amplitude = {amp}', alpha=0.8)

plt.title('Effect of Amplitude on Sound (440 Hz)', fontsize=14, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend(fontsize=11)
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Observation: Higher amplitude → louder sound (same pitch)")

# Demonstrate beating
f1, f2 = 440, 446  # Close frequencies
t_long = np.linspace(0, 1.0, int(sample_rate * 1.0), endpoint=False)

y1 = np.sin(2 * np.pi * f1 * t_long)
y2 = np.sin(2 * np.pi * f2 * t_long)
y_sum = y1 + y2

# Plot only first 500ms for clarity
plot_duration = 0.5
plot_samples = int(sample_rate * plot_duration)

plt.figure(figsize=(14, 5))
plt.plot(t_long[:plot_samples] * 1000, y_sum[:plot_samples], color='#E91E63', linewidth=0.8)
plt.title(f'Beating: {f1} Hz + {f2} Hz (Beat frequency = {abs(f1-f2)} Hz)',
         fontsize=14, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Beat frequency: |{f1} - {f2}| = {abs(f1-f2)} Hz")
print(f"You would hear {abs(f1-f2)} pulses per second")


def generate_tone(freq, duration=1.0, sample_rate=44100, amplitude=0.5):
    """
    Generate a pure sine wave tone.

    Parameters:
    -----------
    freq : float
        Frequency in Hz
    duration : float
        Duration in seconds
    sample_rate : int
        Samples per second (44100 is CD quality)
    amplitude : float
        Amplitude (0.0 to 1.0)

    Returns:
    --------
    signal : numpy array
        The generated audio signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return signal


def save_wav(filename, signal, sample_rate=44100):
    """
    Save audio signal as a .wav file.

    Parameters:
    -----------
    filename : str
        Output filename (should end with .wav)
    signal : numpy array
        Audio signal (values between -1 and 1)
    sample_rate : int
        Samples per second
    """
    # Convert to 16-bit integers
    signal_int = np.int16(signal * 32767)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes = 16 bits
        wav_file.setframerate(sample_rate)

        for sample in signal_int:
            wav_file.writeframes(struct.pack('h', sample))

print("✅ Audio utility functions defined")

# Generate three tones
tones = {
    'tone_C4_261hz.wav': 261.63,
    'tone_A4_440hz.wav': 440.0,
    'tone_A5_880hz.wav': 880.0
}

for filename, freq in tones.items():
    signal = generate_tone(freq, duration=1.0)
    save_wav(filename, signal)
    print(f"✅ Generated: {filename} ({freq} Hz)")

print("\n🎵 Audio files created! You can download and listen to them.")
