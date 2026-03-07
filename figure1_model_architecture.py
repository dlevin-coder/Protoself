import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6))
ax.axis('off')

# helper function to draw boxes
def draw_box(x, y, text, width=2.5, height=1):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03",
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(
        x + width/2,
        y + height/2,
        text,
        ha='center',
        va='center',
        fontsize=11
    )
    return box

# helper function to draw arrows
def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle='->',
        linewidth=1.5,
        mutation_scale=15
    )
    ax.add_patch(arrow)

# boxes
body = draw_box(0.5, 3.5, "Endogenous signals\n(body noise)")
slow = draw_box(3.5, 3.5, "Σ_slow\nEmbodied layer")
fast = draw_box(3.5, 1.5, "Σ_fast\nObserver layer")
pred = draw_box(6.5, 1.5, "Prediction\nerror")

# arrows
draw_arrow(3.0, 4.0, 3.5, 4.0)      # body → slow
draw_arrow(4.75, 3.5, 4.75, 2.5)    # slow → fast
draw_arrow(6.0, 2.0, 6.5, 2.0)      # fast → prediction
draw_arrow(7.75, 2.0, 4.75, 3.5)    # prediction → slow

# limits
ax.set_xlim(0, 9)
ax.set_ylim(0.5, 5)

plt.tight_layout()
plt.savefig("figures/figure1_model_architecture.pdf")
plt.savefig("figures/figure1_model_architecture.png", dpi=300)
plt.close()

print("Figure 1 saved to figures/")