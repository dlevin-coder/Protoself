import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------
# Output directory
# ----------------------------------------

os.makedirs("figures", exist_ok=True)

# ----------------------------------------
# Helper function
# ----------------------------------------

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=300)
    plt.close()


# ========================================
# FIGURE 2 — Population dynamics
# ========================================

data = np.load("baseline.npz")

rate = data["rate"]

plt.figure(figsize=(8,4))
plt.plot(rate)

plt.title("Population activity under endogenous input")
plt.xlabel("Time step")
plt.ylabel("Population rate (Hz)")

save_fig("figure2_population_dynamics")


# ========================================
# FIGURE 3 — Body vs No Body
# ========================================

body = np.load("test_body_only.npz")
nobody = np.load("test_no_body.npz")
full = np.load("test_full.npz")

rate_body = body["population_rate"]
rate_nobody = nobody["population_rate"]
rate_full = full["population_rate"]

plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(rate_body)
plt.title("Body signals only")

plt.subplot(3,1,2)
plt.plot(rate_nobody)
plt.title("No endogenous input")

plt.subplot(3,1,3)
plt.plot(rate_full)
plt.title("Body signals + modulation channels")

plt.xlabel("Time step")

save_fig("figure3_body_vs_nobody")


# ========================================
# FIGURE 4 — Gain sweep
# ========================================

summary = np.load("gain_sweep_summary.npy", allow_pickle=True)

gains = [x[0] for x in summary]
synchrony = [x[1] for x in summary]
pca1 = [x[2] for x in summary]

plt.figure(figsize=(8,4))
plt.plot(gains, synchrony, marker="o")

plt.xlabel("Prediction gain")
plt.ylabel("Synchrony index")

plt.title("Effect of prediction gain")

save_fig("figure4_gain_sweep_synchrony")


# ========================================
# FIGURE 5 — PCA spectrum
# ========================================

data = np.load("baseline.npz")

pca = data["pca"]

plt.figure(figsize=(6,4))
plt.bar(range(1,len(pca)+1), pca)

plt.xlabel("Principal component")
plt.ylabel("Variance explained")

plt.title("Low-dimensional structure of network dynamics")

save_fig("figure5_pca_spectrum")


print("All figures saved to ./figures/")