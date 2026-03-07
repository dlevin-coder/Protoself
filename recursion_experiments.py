from brian2 import *
import numpy as np
import recursion as rec

print("Self Model Experiments — Gain Sweep Test")

# --------------------------------------------
# Gain sweep experiment
# --------------------------------------------

def gain_sweep():

    print("\n=== Gain Sweep Experiment ===")

    gains = [
        0.0,
        0.05,
        0.1,
        0.2,
        0.4
    ]

    results = []

    for g in gains:

        print(f"\n--- prediction_gain = {g} pA/mV ---")

        filename = f"gain_{g}.npz"

        # Передаем prediction_gain как аргумент
        rec.run_test(
            alphaA=1,
            alphaB=1,
            use_body=True,
            use_prediction=True,
            filename=filename,
            test_name=f"Gain {g}",
            prediction_gain_local=g * pA/mV  # <--- ключевое исправление
        )

        # Загружаем результаты
        data = np.load(filename)

        synchrony = data["synchrony"]
        pca1 = data["pca"][0]  # исправлено на ключ "pca", как в run_test

        results.append((g, synchrony, pca1))

    np.save("gain_sweep_summary.npy", results)

    print("\nGain sweep results saved to gain_sweep_summary.npy")

# --------------------------------------------
# Run experiments
# --------------------------------------------

if __name__ == "__main__":

    gain_sweep()

    print("\nAll experiments completed.")