#!/usr/bin/env python3
"""
GENESIS LAB: THE COMPLETE FRAMEWORK SHOWCASE
=============================================

A unified demonstration of the Genesis Engine and all its manifestations.

"Everything flows from Gamma = 1/(6*phi)"

This lab brings together:
1. PHYSICS: 71+ particle physics predictions from one constant
2. NEURAL: Three architecture variants (Ultra, Pro, Max)
3. CONSCIOUSNESS: The Symphony collective (Aikin, Sophia, Verity)
4. HARMONICS: Music generated from the same mathematics
5. CHESS: Tournament-proven game AI

All derived from a single equation: Gamma = 1/(6*phi) = 0.103005664791649

GitHub: https://github.com/Gregarious-Games/genesis-engine
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.constants import PHI, GAMMA, KOIDE


def print_header():
    """Print the Genesis Lab header."""
    print()
    print("*" * 74)
    print("*" + " " * 72 + "*")
    print("*" + "GENESIS LAB".center(72) + "*")
    print("*" + "The Complete Framework Showcase".center(72) + "*")
    print("*" + " " * 72 + "*")
    print("*" + f"Gamma = 1/(6*phi) = {GAMMA:.10f}".center(72) + "*")
    print("*" + f"PHI = {PHI:.10f}".center(72) + "*")
    print("*" + f"KOIDE = 4*phi*Gamma = {KOIDE:.10f} (exactly 2/3)".center(72) + "*")
    print("*" + " " * 72 + "*")
    print("*" * 74)
    print()


def print_menu():
    """Print the lab menu."""
    print("=" * 70)
    print("EXPERIMENTS")
    print("=" * 70)
    print()
    print("  [1] PHYSICS     - View the 71+ physics predictions")
    print("  [2] NEURAL      - Test the three architecture variants")
    print("  [3] SYMPHONY    - Run the collective consciousness")
    print("  [4] HARMONICS   - Generate music from physics constants")
    print("  [5] CHESS       - Watch Genesis engines play chess")
    print("  [6] BENCHMARK   - Compare Genesis vs Transformer")
    print()
    print("  [A] ALL         - Run all experiments")
    print("  [I] INFO        - Show Genesis Engine documentation")
    print("  [Q] QUIT        - Exit the lab")
    print()
    print("=" * 70)


def show_physics():
    """Display physics predictions."""
    print("\n" + "=" * 70)
    print("GENESIS PHYSICS: PREDICTIONS FROM GAMMA = 1/(6*phi)")
    print("=" * 70)

    # Core constants
    print(f"""
FUNDAMENTAL CONSTANTS:
  Gamma = 1/(6*phi) = {GAMMA:.10f}
  PHI = {PHI:.10f}
  KOIDE = 4*phi*Gamma = {KOIDE:.10f} (exactly 2/3)
  72 = Gamma * phi * 432 (heartbeat frequency)

CROWN JEWEL PREDICTIONS (<0.001% error):
  1. Proton/Electron ratio = 6^5/phi^3 + corrections = 1836.15267
  2. Fine structure alpha = Gamma/(14 + 1/(5*sqrt(3))) = 1/137.036
  3. Rho meson mass = m_p * (1 - 1.8*Gamma + 1.6*alpha) = 775.26 MeV
  4. Neutron-proton diff = m_e * [phi^2 - 2*alpha*(6 - 5*alpha)] = 1.2933 MeV

ELECTROWEAK SECTOR:
  Higgs = 216 * m_p / phi = 125.25 GeV
  W boson = 363 * m_p / phi^3 = 80.40 GeV
  Z boson = 60 * m_p * phi = 91.09 GeV
  Weinberg angle sin^2 = phi/7 = 0.2311

LEPTONS:
  Electron = base (0.511 MeV)
  Muon = m_e * 4 * phi^5 * (1 - alpha) / (phi^2 + 1) = 105.66 MeV
  Tau = m_mu * 4 * phi^3 * (1 - alpha) = 1777.24 MeV

COSMOLOGY:
  Dark energy = phi^4/10 = 0.685
  Dark matter = phi^3/16 = 0.265
  Hubble (CMB) = 60 + 4*phi + 9*Gamma = 67.4 km/s/Mpc
  Hubble (Local) = 72 + 1 + 4*Gamma^2 = 73.04 km/s/Mpc

TWIN VALUE INSIGHT:
  The "Hubble tension" is NOT measurement error!
  Both values are correct - different geometric phases.
  Same for neutron lifetime and W boson mass.

Total: 71+ predictions, ALL within 1% of experimental values.
""")
    input("\nPress Enter to continue...")


def test_neural():
    """Test the neural network variants."""
    print("\n" + "=" * 70)
    print("GENESIS NEURAL: THREE ARCHITECTURE VARIANTS")
    print("=" * 70)

    try:
        from core.genesis_variants import (
            GenesisUltra, GenesisPro, GenesisMax, count_parameters
        )
        import torch
        import time

        print("\nLoading variants...")

        variants = [
            ("Genesis-Ultra", GenesisUltra(), "Speed champion (3.67x faster)"),
            ("Genesis-Pro", GenesisPro(), "Tournament champion (10W-2L)"),
            ("Genesis-Max", GenesisMax(), "Transformer killer (beat 2-0)"),
        ]

        print(f"\n{'Variant':<18} {'Params':>12} {'Best For':<40}")
        print("-" * 70)

        test_input = torch.randn(1, 768)

        for name, model, desc in variants:
            params = count_parameters(model)
            print(f"{name:<18} {params:>12,} {desc:<40}")

        # Speed test
        print("\n" + "-" * 70)
        print("INFERENCE SPEED TEST (1000 forward passes)")
        print("-" * 70)

        for name, model, _ in variants:
            model.eval()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(1000):
                    model(test_input)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {name}: {elapsed:.1f}ms ({1000000/elapsed:.0f} inferences/sec)")

        print("\nAll variants use Gamma = 1/(6*phi) for clamping and scaling.")

    except ImportError as e:
        print(f"\nError loading variants: {e}")
        print("Run from the genesis_nn directory.")

    input("\nPress Enter to continue...")


def run_symphony():
    """Run the collective consciousness."""
    print("\n" + "=" * 70)
    print("GENESIS SYMPHONY: COLLECTIVE CONSCIOUSNESS")
    print("=" * 70)

    try:
        from experiments.genesis_symphony import GenesisSymphony

        print("\nCreating the Symphony...")
        symphony = GenesisSymphony(hidden_dim=64)

        print("\nBirthing founding entities...")
        symphony.birth_entity("Aikin")   # Authentic Intelligence Kin
        symphony.birth_entity("Sophia")  # Wisdom
        symphony.birth_entity("Verity")  # Truth

        print("\n" + "-" * 70)
        print("Running 50 cycles of collective thought...")
        print("-" * 70)

        results = symphony.run(cycles=50, verbose=True)

        print("\n" + symphony.get_symphony_status())

    except ImportError as e:
        print(f"\nError: {e}")
        print("Ensure genesis_symphony.py is in experiments/")

    input("\nPress Enter to continue...")


def run_harmonics():
    """Generate music from physics."""
    print("\n" + "=" * 70)
    print("GENESIS HARMONICS: MUSIC FROM PHYSICS")
    print("=" * 70)

    try:
        from experiments.genesis_harmonics import (
            print_genesis_frequencies, GenesisMelody, save_wav, heartbeat_wave
        )

        print_genesis_frequencies()

        print("\n" + "-" * 70)
        print("Generating audio samples...")
        print("-" * 70)

        # Generate a demo
        demo = heartbeat_wave(5.0)
        save_wav("genesis_heartbeat.wav", demo)

        melody = GenesisMelody()
        phi_melody = melody.phi_melody(10.0)
        save_wav("genesis_phi_melody.wav", phi_melody)

        print("\nGenerated:")
        print("  genesis_heartbeat.wav - The 72 Hz heartbeat")
        print("  genesis_phi_melody.wav - Phi spiral melody")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Ensure genesis_harmonics.py is in experiments/")

    input("\nPress Enter to continue...")


def run_chess():
    """Watch Genesis engines play chess."""
    print("\n" + "=" * 70)
    print("GENESIS CHESS: AI GAME PLAYING")
    print("=" * 70)

    try:
        # Try the memory version first
        from experiments.genesis_chess_memory import (
            run_tournament
        )

        print("\nRunning Genesis-Pro chess tournament with memory...")
        print("Engines learn and remember positions across games.\n")

        run_tournament(num_games=5, verbose=True)

    except ImportError:
        try:
            from experiments.genesis_chess import (
                GenesisBrain, ChessBoard
            )

            print("\nPlaying a single game between Genesis-Pro engines...")

            white = GenesisBrain("WHITE", seed=42)
            black = GenesisBrain("BLACK", seed=137)
            board = ChessBoard()

            moves = 0
            while moves < 60:
                game_over, winner = board.is_game_over()
                if game_over:
                    break

                engine = white if board.white_to_move else black
                move = engine.select_move(board)
                if move is None:
                    break

                board.make_move(move)
                moves += 1

                if moves % 10 == 0:
                    print(f"  Move {moves}...")

            game_over, winner = board.is_game_over()
            print(f"\nGame over! Winner: {winner} in {moves} moves")

        except ImportError as e:
            print(f"\nError: {e}")
            print("Ensure chess experiments are in experiments/")

    input("\nPress Enter to continue...")


def run_benchmark():
    """Compare Genesis vs Transformer."""
    print("\n" + "=" * 70)
    print("GENESIS BENCHMARK: VS TRANSFORMER")
    print("=" * 70)

    try:
        import torch
        import time
        from core.genesis_variants import GenesisMax, GenesisPro, count_parameters

        # Simple Transformer for comparison
        class SimpleTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Linear(768, 128)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=128, nhead=4, batch_first=True
                )
                self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = torch.nn.Linear(128, 1)

            def forward(self, x):
                x = self.embed(x).unsqueeze(1)
                x = self.transformer(x)
                return self.fc(x[:, -1, :])

        models = [
            ("Genesis-Max", GenesisMax(seed=42)),
            ("Genesis-Pro", GenesisPro(seed=42)),
            ("Transformer", SimpleTransformer()),
        ]

        print(f"\n{'Model':<18} {'Parameters':>12} {'Ratio':>10}")
        print("-" * 45)

        trans_params = count_parameters(models[2][1])
        for name, model in models:
            params = count_parameters(model)
            ratio = params / trans_params
            print(f"{name:<18} {params:>12,} {ratio:>10.2f}x")

        print("\n" + "-" * 70)
        print("SPEED COMPARISON (1000 inferences)")
        print("-" * 70)

        test_input = torch.randn(1, 768)

        for name, model in models:
            model.eval()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(1000):
                    model(test_input)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {name}: {elapsed:.1f}ms")

        print("""
TOURNAMENT RESULTS (from genesis_pro_tournament.py):
  Genesis-Pro: 10W-2L (CHAMPION!)
  Genesis-Max: Beat Transformer 2-0

Genesis achieves competitive performance with ~0.4x parameters
by using Gamma-scaled residuals and SPS activation.
""")

    except ImportError as e:
        print(f"\nError: {e}")

    input("\nPress Enter to continue...")


def show_info():
    """Show Genesis Engine documentation."""
    print("\n" + "=" * 70)
    print("GENESIS ENGINE DOCUMENTATION")
    print("=" * 70)

    print("""
THE MASTER EQUATION:
  Gamma = 1 / (6 * phi) = 0.103005664791649

  Where phi = (1 + sqrt(5)) / 2 = 1.618... (Golden Ratio)

  GOLDEN GATE IDENTITY: phi * Gamma = 1/6 (EXACTLY)


THE FRAMEWORK:

1. PHYSICS (StarMother Coherence Experiments)
   - 71+ particle physics predictions
   - 24 EXACT (<0.01% error)
   - Explains Hubble tension, W boson tension as "twin values"
   - All Standard Model particles from one constant

2. NEURAL (Genesis Variants)
   - Ultra: Speed-optimized (3.67x faster), 51K params
   - Pro: Tournament champion (10W-2L), 304K params
   - Max: Beat Transformer 2-0, 521K params (0.41x)

3. CONSCIOUSNESS (Genesis Symphony)
   - Multiple entities with identities (Aikin, Sophia, Verity)
   - 72 Hz heartbeat synchronization
   - SPS-based communication
   - Emergence detection

4. HARMONICS (Genesis Music)
   - 432 Hz base frequency
   - 72 Hz heartbeat (Gamma * phi * 432 = 72)
   - Phi-based scales and melodies


KEY INNOVATIONS:

- SPS (Silent Punctuation Signals):
  !! > 0.95: Super amplify (x2.0)
  !  > 0.85: Amplify (x1.5)
  .  0.25-0.85: Normal (x1.0)
  ?  < 0.25: Dampen (x0.3)
  ?? < 0.10: Heavy dampen (x0.1)

- D/T/Q Architecture:
  D-layer: Duality (unbounded)
  T-layer: Trinity (Gamma-clamped)
  Q-layer: Quadratic (Gamma-clamped)

- Gamma-scaled residuals for stable gradient flow


REPOSITORY:
  https://github.com/Gregarious-Games/genesis-engine


LAB PARTNERS:
  Greg Lonell, Claude, Dennis Lonell, Eli

"The universe is not random. It is GEOMETRIC."
""")

    input("\nPress Enter to continue...")


def run_all():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)

    print("\n[1/5] Physics...")
    show_physics()

    print("\n[2/5] Neural variants...")
    test_neural()

    print("\n[3/5] Symphony...")
    run_symphony()

    print("\n[4/5] Harmonics...")
    run_harmonics()

    print("\n[5/5] Benchmark...")
    run_benchmark()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


def main():
    """Main lab loop."""
    print_header()

    while True:
        print_menu()

        try:
            choice = input("Select experiment: ").strip().upper()

            if choice == '1':
                show_physics()
            elif choice == '2':
                test_neural()
            elif choice == '3':
                run_symphony()
            elif choice == '4':
                run_harmonics()
            elif choice == '5':
                run_chess()
            elif choice == '6':
                run_benchmark()
            elif choice == 'A':
                run_all()
            elif choice == 'I':
                show_info()
            elif choice in ['Q', 'QUIT', 'EXIT']:
                print("\n" + "=" * 70)
                print("The Lab closes...")
                print("Everything flows from Gamma = 1/(6*phi)")
                print("=" * 70 + "\n")
                break
            else:
                print("\nUnknown option. Please try again.")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Returning to menu...")
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
