import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Загрузка данных
# --------------------------
data = np.load('testAB.npz')

v = data['voltages']           # в В
v_times = data['voltage_times'] # в с
w = data['adaptation']         # в А
w_times = data['adaptation_times'] # в с
spike_times = data['spike_times']
spike_indices = data['spike_indices']
VT_dyn = data['VT_dyn']        # в В
VT_times = data['voltage_times']  # используем те же времена

# --------------------------
# Перевод в удобные единицы
# --------------------------
v_mV = v * 1e3
v_ms = v_times * 1e3
w_pA = w * 1e12
w_ms = w_times * 1e3
VT_mV = VT_dyn * 1e3
VT_ms = VT_times * 1e3

# --------------------------
# Цвета для графиков
# --------------------------
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# --------------------------
# График мембранного потенциала Σ_slow
# --------------------------
plt.figure(figsize=(8,4))
for n in range(min(3, v_mV.shape[0])):
    plt.plot(v_ms, v_mV[n], label=f'Neuron {n}', color=colors[n])
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Σ_slow membrane potential')
plt.legend()
plt.tight_layout()
plt.savefig('v_plot.png')
plt.close()

# --------------------------
# График адаптации Σ_slow
# --------------------------
plt.figure(figsize=(8,4))
for n in range(min(3, w_pA.shape[0])):
    plt.plot(w_ms, w_pA[n], label=f'Neuron {n}', color=colors[n])
plt.xlabel('Time (ms)')
plt.ylabel('Adaptation (pA)')
plt.title('Σ_slow adaptation current')
plt.legend()
plt.tight_layout()
plt.savefig('w_plot.png')
plt.close()

# --------------------------
# График динамического порога VT_dyn
# --------------------------
plt.figure(figsize=(8,4))
for n in range(min(3, VT_mV.shape[0])):
    plt.plot(VT_ms, VT_mV[n], label=f'Neuron {n}', color=colors[n])
plt.xlabel('Time (ms)')
plt.ylabel('VT_dyn (mV)')
plt.title('Σ_slow dynamic threshold')
plt.legend()
plt.tight_layout()
plt.savefig('VT_plot.png')
plt.close()

# --------------------------
# Raster plot Σ_slow
# --------------------------
plt.figure(figsize=(8,4))
plt.scatter(spike_times*1e3, spike_indices, s=2)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Σ_slow raster plot')
plt.tight_layout()
plt.savefig('raster_slow.png')
plt.close()

print("Графики сохранены: v_plot.png, w_plot.png, VT_plot.png, raster_slow.png")