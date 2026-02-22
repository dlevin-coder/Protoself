from brian2 import *
import numpy as np

prefs.codegen.target = 'cython'  # Используем Cython для ускорения
print("=== SNN с генеративной петлей и ошибкой предсказания ===")

########################################
# Параметры
########################################
N_input = 3
N_neurons = 2
N_pred = N_neurons

tau = 10*ms
tau_pre = 20*ms
tau_post = 20*ms

Apre_val = 0.02
Apost_val = -0.025

########################################
# Основная сеть: адаптированный LIF
########################################
eqs = '''
dv/dt = (-v - a + I)/tau : 1
da/dt = -a/(100*ms) : 1
dI/dt = -I/(30*ms) : 1
'''

G = NeuronGroup(N_neurons, model=eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='euler')
G.v = 'rand()*0.1'

########################################
# Входные спайки
########################################
input_indices = [0, 1, 2]*20
input_times = []
for i in range(60):
    input_times.append(i*5 + 2)
inputs = SpikeGeneratorGroup(N_input, indices=input_indices, times=input_times*ms)

########################################
# STDP для основной сети
########################################
stdp_eqs = '''
w : 1
dApre/dt = -Apre/tau_pre : 1 (event-driven)
dApost/dt = -Apost/tau_post : 1 (event-driven)
'''

on_pre = f'''
I_post += w
Apre += {Apre_val}
w = clip(w + Apost - 0.02*w, 0, 3)
'''

on_post = f'''
Apost += {Apost_val}
w = clip(w + Apre - 0.02*w, 0, 3)
'''

synapses = Synapses(inputs, G, model=stdp_eqs, on_pre=on_pre, on_post=on_post)
synapses.connect()
synapses.w = '0.5 + rand()*0.5'

########################################
# Генеративная сеть
########################################
G_pred = NeuronGroup(
    N_pred,
    eqs,
    threshold='v>1',
    reset='v=0',
    refractory=5*ms,
    method='euler'
)
G_pred.v = '0.5 + 0.1*rand()'

# Связь основной -> генеративная
pred_syn = Synapses(G, G_pred, model='w_pred : 1', on_pre='v_post += w_pred')
pred_syn.connect(j='i')
pred_syn.w_pred = 0.8
pred_syn.delay = 10*ms

# Ошибка предсказания: генеративная -> основная сеть
error_syn = Synapses(G_pred, G, model='w_err : 1', on_pre='v_post += 0.3*(v_post - v_pre)')
error_syn.connect(j='i')
error_syn.w_err = 0.3

########################################
# Мониторы
########################################
spike_mon_input = SpikeMonitor(inputs)
spike_mon_neurons = SpikeMonitor(G)
spike_mon_pred = SpikeMonitor(G_pred)
weight_mon = StateMonitor(synapses, 'w', record=range(min(5, len(synapses))))

########################################
# Запуск
########################################
duration = 0.3*second
print(f"Запуск симуляции на {duration/second} секунд...")
run(duration)

########################################
# Результаты
########################################
print("="*50)
print("ВХОДНЫЕ СПАЙКИ:", spike_mon_input.num_spikes)
print("СПАЙКИ ОСНОВНОЙ СЕТИ:", spike_mon_neurons.num_spikes)
for i in range(N_neurons):
    spikes_i = spike_mon_neurons.spike_trains()[i]
    print(f"  Нейрон {i}: {len(spikes_i)} спайков")

print("СПАЙКИ ГЕНЕРАТИВНОЙ СЕТИ:", spike_mon_pred.num_spikes)
for i in range(N_pred):
    spikes_i = spike_mon_pred.spike_trains()[i]
    print(f"  Генер. нейрон {i}: {len(spikes_i)} спайков")

print("ИЗМЕНЕНИЕ ВЕСОВ:")
for i in range(min(3, len(weight_mon.w))):
    if len(weight_mon.w[i]) > 0:
        initial = weight_mon.w[i][0]
        final = weight_mon.w[i][-1]
        print(f"  Синапс {i}: {initial:.3f} -> {final:.3f} (Δ {final-initial:+.3f})")

print("✓ Модель завершена успешно!")
