# main_snn.py с надежной визуализацией
from brian2 import *
from audio_spiketrain import audio_to_spiketrain_loop
import matplotlib.pyplot as plt
import numpy as np
import os

prefs.codegen.target = 'cython'

# Создаем директорию для результатов
os.makedirs('results', exist_ok=True)

# =========================
# Параметры аудио
# =========================
audio_file = "drums_5sec.wav"
N_input = 3
threshold = 0.2
repeat = 1

# Устанавливаем единый dt
defaultclock.dt = 22.67573696 * us

try:
    # Генерация spike train
    inputs = audio_to_spiketrain_loop(audio_file, N_input=N_input, threshold=threshold, repeat=repeat)
    print(f"SpikeGeneratorGroup создан успешно")
except Exception as e:
    print(f"Ошибка при создании SpikeGeneratorGroup: {e}")
    exit(1)

# =========================
# SNN модель
# =========================
N_neurons = 2
tau = 10*ms

eqs = '''
dv/dt = -v / tau : 1
'''

G = NeuronGroup(N_neurons, model=eqs, threshold='v>1', reset='v=0', method='euler')
G.v = 'rand()*0.1'

# Подключение
syn = Synapses(inputs, G, model='w:1', on_pre='v_post += w')
syn.connect()
syn.w = '0.5 + 0.5*rand()'

# Мониторы
spike_mon = SpikeMonitor(G)
spike_mon_input = SpikeMonitor(inputs)
state_mon = StateMonitor(G, 'v', record=True)

# Запуск
duration = 100*ms
run(duration)

# =========================
# Сохранение результатов в файлы
# =========================

print("Сохранение результатов...")

# Получаем данные о спайках
input_spike_times = np.array(spike_mon_input.t)
input_spike_indices = np.array(spike_mon_input.i)
output_spike_times = np.array(spike_mon.t)
output_spike_indices = np.array(spike_mon.i)

# 1. Raster plots
plt.figure(figsize=(12, 10))

# Input spike raster
plt.subplot(3, 1, 1)
if len(input_spike_times) > 0:
    plt.plot(input_spike_times/ms, input_spike_indices, '.k', markersize=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title(f'Input Spike Raster (total: {len(input_spike_times)})')
else:
    plt.text(0.5, 0.5, 'No input spikes', transform=plt.gca().transAxes, 
             ha='center', va='center')
    plt.title('Input Spike Raster')
plt.xlim(0, duration/ms)

# Output spike raster
plt.subplot(3, 1, 2)
if len(output_spike_times) > 0:
    plt.plot(output_spike_times/ms, output_spike_indices, '.r', markersize=3)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title(f'Output Spike Raster (total: {len(output_spike_times)})')
else:
    plt.text(0.5, 0.5, 'No output spikes', transform=plt.gca().transAxes, 
             ha='center', va='center')
    plt.title('Output Spike Raster')
plt.xlim(0, duration/ms)

# Voltage traces
plt.subplot(3, 1, 3)
if len(state_mon.t) > 0 and len(state_mon.v) > 0:
    for idx in range(len(state_mon.v)):
        plt.plot(state_mon.t/ms, state_mon.v[idx], label=f'Neuron {idx}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    plt.title('Neuron Voltage Traces')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No voltage data', transform=plt.gca().transAxes, 
             ha='center', va='center')
    plt.title('Neuron Voltage Traces')

plt.tight_layout()
plt.savefig('results/snn_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Результаты SNN сохранены в: results/snn_results.png")

# 2. Текстовый отчет
with open('results/snn_report.txt', 'w') as f:
    f.write("=== SNN Analysis Report ===\n\n")
    
    f.write("Network Parameters:\n")
    f.write(f"  Input neurons: {N_input}\n")
    f.write(f"  Output neurons: {N_neurons}\n")
    f.write(f"  Simulation duration: {duration/ms} ms\n")
    f.write(f"  Threshold: {threshold}\n\n")
    
    f.write("Results:\n")
    f.write(f"  Input spikes: {len(input_spike_times)}\n")
    f.write(f"  Output spikes: {len(output_spike_times)}\n\n")
    
    if len(input_spike_times) > 0:
        f.write("Input spike times (first 20):\n")
        for i in range(min(20, len(input_spike_times))):
            f.write(f"  Time: {input_spike_times[i]/ms:.3f} ms, Neuron: {input_spike_indices[i]}\n")
    
    if len(output_spike_times) > 0:
        f.write(f"\nOutput spike times (first 20):\n")
        for i in range(min(20, len(output_spike_times))):
            f.write(f"  Time: {output_spike_times[i]/ms:.3f} ms, Neuron: {output_spike_indices[i]}\n")

print("Текстовый отчет сохранен в: results/snn_report.txt")

# 3. Сохранение данных в CSV для дальнейшего анализа
try:
    if len(input_spike_times) > 0:
        np.savetxt('results/input_spikes.csv', 
                   np.column_stack([input_spike_times/ms, input_spike_indices]), 
                   delimiter=',', header='time_ms,neuron_index', comments='')

    if len(output_spike_times) > 0:
        np.savetxt('results/output_spikes.csv', 
                   np.column_stack([output_spike_times/ms, output_spike_indices]), 
                   delimiter=',', header='time_ms,neuron_index', comments='')

    if len(state_mon.t) > 0 and len(state_mon.v) > 0:
        # Сохраняем voltage traces
        data_to_save = [state_mon.t/ms]
        headers = ['time_ms']
        for idx in range(len(state_mon.v)):
            data_to_save.append(state_mon.v[idx])
            headers.append(f'neuron_{idx}_voltage')
        
        np.savetxt('results/voltage_traces.csv', 
                   np.column_stack(data_to_save), 
                   delimiter=',', header=','.join(headers), comments='')

    print("CSV данные сохранены в: results/")
    
except Exception as e:
    print(f"Предупреждение: ошибка при сохранении CSV: {e}")

print("Анализ завершен!")
