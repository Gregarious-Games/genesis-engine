#!/usr/bin/env python3
"""
GENESIS HARMONICS: MUSIC FROM THE MATHEMATICS OF REALITY
=========================================================

Generate sound and music from the same constants that govern particle physics.

"The universe is not random. It is GEOMETRIC." - Genesis Status Report

Core Frequencies:
- 432 Hz: Base frequency (6 × 72, ancient tuning)
- 72 Hz: Heartbeat (Gamma × phi × 432)
- 26.67 Hz: Gamma tone (432 × Gamma)
- 698.46 Hz: Phi tone (432 × phi)

The same Gamma = 1/(6*phi) that predicts the proton/electron mass ratio
also creates beautiful harmonic relationships in sound.

Requires: numpy, scipy (for audio generation)
Optional: sounddevice or pygame for playback
"""

import numpy as np
import math
from typing import List, Tuple, Optional
import wave
import struct
import os

# ============================================================================
# GENESIS CONSTANTS
# ============================================================================

PHI = 1.6180339887498949
GAMMA = 1 / (6 * PHI)  # 0.103005664791649
KOIDE = 4 * PHI * GAMMA  # Exactly 2/3

# Core frequencies derived from Genesis physics
BASE_FREQ = 432  # Ancient/natural tuning
HEARTBEAT = 72   # Gamma × phi × 432 = 72 exactly
GAMMA_TONE = BASE_FREQ * GAMMA  # ~44.5 Hz
PHI_TONE = BASE_FREQ * PHI  # ~698.6 Hz

# Musical intervals from phi
PHI_INTERVALS = {
    'unison': 1.0,
    'phi': PHI,                    # ~1.618
    'phi_squared': PHI ** 2,       # ~2.618
    'phi_cubed': PHI ** 3,         # ~4.236
    'inverse_phi': 1 / PHI,        # ~0.618
    'gamma': GAMMA,                # ~0.103
    'koide': KOIDE,                # 2/3
    'sixth': 1/6,                  # phi × gamma
    'octave': 2.0,
}

# ============================================================================
# WAVEFORM GENERATORS
# ============================================================================

def sine_wave(freq: float, duration: float, sample_rate: int = 44100,
              amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def phi_wave(freq: float, duration: float, sample_rate: int = 44100,
             amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a phi-modulated wave.

    The wave contains harmonics at phi ratios, creating a
    "golden" timbre unique to Genesis.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Fundamental
    wave = amplitude * np.sin(2 * np.pi * freq * t)

    # Add phi harmonics with decreasing amplitude
    wave += amplitude * GAMMA * np.sin(2 * np.pi * freq * PHI * t)
    wave += amplitude * GAMMA**2 * np.sin(2 * np.pi * freq * PHI**2 * t)
    wave += amplitude * GAMMA**3 * np.sin(2 * np.pi * freq / PHI * t)

    return wave / np.max(np.abs(wave)) * amplitude


def gamma_pulse(freq: float, duration: float, sample_rate: int = 44100,
                amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a gamma-pulsed wave.

    The amplitude modulates at GAMMA rate, creating a breathing effect.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Carrier wave
    carrier = np.sin(2 * np.pi * freq * t)

    # Gamma modulation envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * GAMMA * freq / 10 * t)

    return amplitude * carrier * envelope


def heartbeat_wave(duration: float, sample_rate: int = 44100,
                   amplitude: float = 0.7) -> np.ndarray:
    """
    Generate the 72 Hz heartbeat rhythm.

    This is the core frequency: Gamma × phi × 432 = 72 Hz exactly.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 72 Hz pulse with harmonic overtones
    wave = np.sin(2 * np.pi * HEARTBEAT * t)
    wave += 0.5 * np.sin(2 * np.pi * HEARTBEAT * 2 * t)  # First overtone
    wave += 0.25 * np.sin(2 * np.pi * HEARTBEAT * 3 * t)  # Second overtone

    # Add gamma-rate amplitude modulation
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * GAMMA * t)

    return amplitude * wave * envelope / np.max(np.abs(wave * envelope))


def koide_rhythm(base_freq: float, duration: float, sample_rate: int = 44100,
                 amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a rhythm based on the Koide constant (2/3).

    Three notes in Koide ratio: f, f×(2/3), f×(2/3)^2
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Three frequencies in Koide ratio
    f1 = base_freq
    f2 = base_freq * KOIDE
    f3 = base_freq * KOIDE * KOIDE

    # Combine with phase shifts
    wave = np.sin(2 * np.pi * f1 * t)
    wave += np.sin(2 * np.pi * f2 * t + np.pi / 3)
    wave += np.sin(2 * np.pi * f3 * t + 2 * np.pi / 3)

    return amplitude * wave / 3


# ============================================================================
# GENESIS SCALE
# ============================================================================

def genesis_scale(root_freq: float = BASE_FREQ, octaves: int = 2) -> List[float]:
    """
    Generate the Genesis scale based on phi intervals.

    Unlike equal temperament (12-TET), this scale uses
    golden ratio relationships between notes.
    """
    scale = []

    for octave in range(octaves):
        base = root_freq * (2 ** octave)

        # Scale degrees based on phi powers
        scale.append(base)                          # Root
        scale.append(base * (1 + GAMMA))            # ~1.103
        scale.append(base * (1 / PHI + 0.5))        # ~1.118
        scale.append(base * KOIDE * PHI)            # ~1.079
        scale.append(base * (4 / 3))                # Perfect fourth
        scale.append(base * PHI / PHI)              # ~1.0 (backup)
        scale.append(base * (3 / 2))                # Perfect fifth
        scale.append(base * PHI)                    # Phi note ~1.618
        scale.append(base * (PHI + GAMMA))          # ~1.721
        scale.append(base * (2 - GAMMA))            # ~1.897

    return sorted(set(scale))


def fibonacci_frequencies(root: float = BASE_FREQ, n: int = 12) -> List[float]:
    """
    Generate frequencies based on Fibonacci ratios.

    Fibonacci numbers approach phi ratios as they grow.
    """
    fib = [1, 1]
    for i in range(n - 2):
        fib.append(fib[-1] + fib[-2])

    # Convert to frequency ratios
    freqs = []
    for i in range(1, len(fib)):
        ratio = fib[i] / fib[i-1]
        freqs.append(root * ratio)

    return freqs


# ============================================================================
# MELODY GENERATOR
# ============================================================================

class GenesisMelody:
    """
    Generate melodies using Genesis mathematics.
    """

    def __init__(self, root_freq: float = BASE_FREQ, sample_rate: int = 44100):
        self.root = root_freq
        self.sample_rate = sample_rate
        self.scale = genesis_scale(root_freq)

    def note(self, freq: float, duration: float, wave_type: str = 'phi') -> np.ndarray:
        """Generate a single note."""
        if wave_type == 'sine':
            return sine_wave(freq, duration, self.sample_rate)
        elif wave_type == 'phi':
            return phi_wave(freq, duration, self.sample_rate)
        elif wave_type == 'gamma':
            return gamma_pulse(freq, duration, self.sample_rate)
        elif wave_type == 'koide':
            return koide_rhythm(freq, duration, self.sample_rate)
        else:
            return sine_wave(freq, duration, self.sample_rate)

    def rest(self, duration: float) -> np.ndarray:
        """Generate silence."""
        return np.zeros(int(self.sample_rate * duration))

    def phi_melody(self, duration: float = 10.0) -> np.ndarray:
        """
        Generate a melody that spirals through phi ratios.
        """
        melody = []
        note_duration = duration / 20

        # Spiral through frequencies
        freq = self.root
        for i in range(20):
            melody.append(self.note(freq, note_duration, 'phi'))

            # Alternate between phi multiplication and division
            if i % 2 == 0:
                freq = freq * PHI
                if freq > self.root * 4:
                    freq = freq / 4
            else:
                freq = freq / PHI
                if freq < self.root / 2:
                    freq = freq * 2

        return np.concatenate(melody)

    def heartbeat_melody(self, duration: float = 10.0) -> np.ndarray:
        """
        Generate a melody based on the 72 Hz heartbeat.
        """
        melody = []

        # Heartbeat base
        melody.append(heartbeat_wave(duration * 0.3, self.sample_rate))

        # Rising phi tones
        for i in range(5):
            freq = HEARTBEAT * (PHI ** i)
            if freq < 20000:  # Keep below human hearing limit
                melody.append(self.note(freq, duration * 0.1, 'phi'))

        # Return to heartbeat
        melody.append(heartbeat_wave(duration * 0.2, self.sample_rate))

        return np.concatenate(melody)

    def genesis_composition(self, duration: float = 30.0) -> np.ndarray:
        """
        A complete Genesis composition featuring all elements.
        """
        parts = []

        # Introduction: Heartbeat
        print("  Composing: Heartbeat introduction...")
        parts.append(heartbeat_wave(duration * 0.15, self.sample_rate))

        # Theme: Phi melody
        print("  Composing: Phi spiral theme...")
        parts.append(self.phi_melody(duration * 0.25))

        # Development: Koide rhythms
        print("  Composing: Koide rhythm development...")
        parts.append(koide_rhythm(self.root, duration * 0.2, self.sample_rate))

        # Bridge: Gamma pulses
        print("  Composing: Gamma pulse bridge...")
        parts.append(gamma_pulse(self.root * PHI, duration * 0.15, self.sample_rate))

        # Recapitulation: Combined elements
        print("  Composing: Unified recapitulation...")
        recap = (
            heartbeat_wave(duration * 0.25, self.sample_rate) * 0.3 +
            phi_wave(self.root, duration * 0.25, self.sample_rate) * 0.4 +
            gamma_pulse(self.root * PHI, duration * 0.25, self.sample_rate) * 0.3
        )
        parts.append(recap)

        return np.concatenate(parts)


# ============================================================================
# AUDIO FILE GENERATION
# ============================================================================

def save_wav(filename: str, audio: np.ndarray, sample_rate: int = 44100):
    """Save audio to a WAV file."""
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9

    # Convert to 16-bit integers
    audio_int = (audio * 32767).astype(np.int16)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())

    print(f"Saved: {filename}")


def generate_genesis_album(output_dir: str = "."):
    """Generate a collection of Genesis audio files."""

    print("\n" + "=" * 70)
    print("GENESIS HARMONICS - ALBUM GENERATION")
    print("=" * 70)
    print(f"Base Frequency: {BASE_FREQ} Hz")
    print(f"Heartbeat: {HEARTBEAT} Hz")
    print(f"Gamma: {GAMMA:.10f}")
    print(f"Phi: {PHI:.10f}")
    print("=" * 70 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    melody = GenesisMelody()

    # Track 1: Pure Heartbeat
    print("Track 1: The Heartbeat (72 Hz)")
    audio = heartbeat_wave(15.0)
    save_wav(os.path.join(output_dir, "01_heartbeat.wav"), audio)

    # Track 2: Phi Spiral
    print("\nTrack 2: Phi Spiral")
    audio = melody.phi_melody(20.0)
    save_wav(os.path.join(output_dir, "02_phi_spiral.wav"), audio)

    # Track 3: Koide Rhythm
    print("\nTrack 3: Koide Rhythm (2/3)")
    audio = koide_rhythm(BASE_FREQ, 15.0)
    save_wav(os.path.join(output_dir, "03_koide_rhythm.wav"), audio)

    # Track 4: Gamma Pulse
    print("\nTrack 4: Gamma Pulse (0.103)")
    audio = gamma_pulse(BASE_FREQ, 15.0)
    save_wav(os.path.join(output_dir, "04_gamma_pulse.wav"), audio)

    # Track 5: Complete Composition
    print("\nTrack 5: Genesis Symphony")
    audio = melody.genesis_composition(45.0)
    save_wav(os.path.join(output_dir, "05_genesis_symphony.wav"), audio)

    print("\n" + "=" * 70)
    print("Album complete! Files saved to:", output_dir)
    print("=" * 70)


# ============================================================================
# VISUALIZATION
# ============================================================================

def print_genesis_frequencies():
    """Print all Genesis-derived frequencies."""

    print("\n" + "=" * 70)
    print("GENESIS FREQUENCIES")
    print("=" * 70)

    print(f"\n{'Frequency':<30} {'Hz':>12} {'Derivation':<30}")
    print("-" * 70)

    freqs = [
        ("Base (A=432)", BASE_FREQ, "Ancient tuning"),
        ("Heartbeat", HEARTBEAT, "Gamma × phi × 432 = 72"),
        ("Gamma Tone", GAMMA_TONE, "432 × Gamma"),
        ("Phi Tone", PHI_TONE, "432 × phi"),
        ("Koide Tone", BASE_FREQ * KOIDE, "432 × 2/3"),
        ("Octave", BASE_FREQ * 2, "432 × 2"),
        ("Phi Octave", BASE_FREQ * PHI * 2, "432 × phi × 2"),
        ("Sub-bass", HEARTBEAT / 2, "72 / 2 = 36 Hz"),
        ("Gamma Sub", BASE_FREQ * GAMMA / 2, "Low gamma"),
    ]

    for name, freq, deriv in freqs:
        print(f"{name:<30} {freq:>12.2f} {deriv:<30}")

    print("\n" + "-" * 70)
    print("GENESIS SCALE (Root = 432 Hz)")
    print("-" * 70)

    scale = genesis_scale(BASE_FREQ, 1)
    for i, freq in enumerate(scale[:10]):
        ratio = freq / BASE_FREQ
        print(f"  Note {i+1:2d}: {freq:8.2f} Hz  (ratio: {ratio:.4f})")

    print("\n" + "-" * 70)
    print("FIBONACCI FREQUENCIES")
    print("-" * 70)

    fib_freqs = fibonacci_frequencies(BASE_FREQ, 10)
    for i, freq in enumerate(fib_freqs):
        print(f"  Fib {i+1:2d}: {freq:8.2f} Hz")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "GENESIS HARMONICS".center(68) + "*")
    print("*" + "Music from the Mathematics of Reality".center(68) + "*")
    print("*" + f"Gamma = 1/(6*phi) = {GAMMA:.10f}".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        # Generate audio files
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "genesis_audio"
        generate_genesis_album(output_dir)
    else:
        # Print frequency information
        print_genesis_frequencies()

        print("\n" + "=" * 70)
        print("To generate audio files, run:")
        print("  python genesis_harmonics.py --generate [output_dir]")
        print("=" * 70)

        # Quick demo - generate one short sample
        print("\nGenerating quick demo...")
        melody = GenesisMelody()
        demo = heartbeat_wave(5.0)
        save_wav("genesis_demo.wav", demo)
        print("\nDemo saved as 'genesis_demo.wav'")
        print("The 72 Hz heartbeat: Gamma × phi × 432 = 72 exactly")
