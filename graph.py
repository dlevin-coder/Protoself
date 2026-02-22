import matplotlib.pyplot as plt

# ===== Test 1: Contextual Modulation =====
conditions1 = ['Baseline', 'Modified']
STM = [0, 8307]
Summary = [0, 1815]

# ===== Test 2: Persistence Without Input =====
time_points = ['Before Pause', 'After Pause']
state_values = [0.008, 0.002]

# ===== Test 3: Valence-Like Dynamics =====
states = ['Baseline', 'Positive', 'Neutral', 'Negative']
values = [0.0189, 0.4039, 0.2758, -0.0762]

# ===== Создаем Figure с 3 подграфиками =====
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# --- Test 1: Contextual Modulation ---
width = 0.35
x = [0,1]
bars1 = axs[0].bar([i - width/2 for i in x], STM, width, color='steelblue', label='STM')
bars2 = axs[0].bar([i + width/2 for i in x], Summary, width, color='indianred', alpha=0.8, label='Summary')

axs[0].set_xticks(x)
axs[0].set_xticklabels(conditions1)
axs[0].set_title('Test 1: Contextual Modulation')
axs[0].set_ylabel('Activation Value')
axs[0].legend()
axs[0].grid(axis='y', linestyle='--', alpha=0.5)

# Цифровые подписи над столбцами
for bar in bars1 + bars2:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, height + max(STM)*0.02, f'{height:.0f}', ha='center', va='bottom', fontsize=10)

# --- Test 2: State Persistence ---
axs[1].plot(time_points, state_values, marker='o', linestyle='-', color='seagreen', linewidth=2)
axs[1].set_title('Test 2: State Persistence')
axs[1].set_ylabel('Self-object State')
axs[1].set_ylim(0, max(state_values)*1.5)
axs[1].grid(axis='y', linestyle='--', alpha=0.5)

# Цифровые подписи
for i, val in enumerate(state_values):
    axs[1].text(i, val + max(state_values)*0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# --- Test 3: Global Valence-Like State ---
colors3 = ['gray', 'limegreen', 'orange', 'crimson']
bars3 = axs[2].bar(states, values, color=colors3)
axs[2].set_title('Test 3: Global Valence-Like State')
axs[2].set_ylabel('Measured Value')
axs[2].grid(axis='y', linestyle='--', alpha=0.5)

# Цифровые подписи
for bar in bars3:
    height = bar.get_height()
    axs[2].text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# ===== Сохраняем в файл =====
plt.savefig('behavioral_tests_results_annotated.png', dpi=300)
print("Figure saved as 'behavioral_tests_results_annotated.png'")