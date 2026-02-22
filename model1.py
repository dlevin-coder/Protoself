from brian2 import *
import numpy as np

prefs.codegen.target = 'cython'  # Используем Cython для ускорения
print("=== SNN с генеративной петлей + LTM (работающий вариант) ===")

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
# Основная сеть: адаптированный LIF с LTM
########################################
eqs = '''
dv/dt = (-v - a + I + 0.2*LTM)/tau : 1
da/dt = -a/(100*ms) : 1
dI/dt = -I/(30*ms) : 1
LTM : 1
'''

G = NeuronGroup(N_neurons, model=eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='euler')
G.v = 'rand()*0.1'
G.LTM = 0.0

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
G_pred.LTM = 0.0

# Основная -> генеративная сеть
pred_syn = Synapses(G, G_pred, model='w_pred : 1', on_pre='v_post += w_pred')
pred_syn.connect(j='i')
pred_syn.w_pred = 0.8
pred_syn.delay = 10*ms

# Генеративная -> основная сеть (ошибка предсказания)
error_syn = Synapses(G_pred, G, model='w_err : 1', on_pre='v_post += 0.3*(v_post - v_pre)')
error_syn.connect(j='i')
error_syn.w_err = 0.3

########################################
# LTM-поток: интегратор генеративной сети
########################################
# LTM обновляется в основной сети через генеративную сеть
ltm_syn = Synapses(G_pred, G, on_pre='LTM_post += 0.05')
ltm_syn.connect(j='i')

########################################
# Мониторы
########################################
spike_mon_input = SpikeMonitor(inputs)
spike_mon_neurons = SpikeMonitor(G)
spike_mon_pred = SpikeMonitor(G_pred)
ltm_mon = StateMonitor(G, 'LTM', record=True)
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

print("LTM состояние (последние значения):")
for i in range(N_pred):
    print(f"  LTM нейрон {i}: {ltm_mon.LTM[i][-1]:.3f}")

print("ИЗМЕНЕНИЕ ВЕСОВ:")
for i in range(min(3, len(weight_mon.w))):
    if len(weight_mon.w[i]) > 0:
        initial = weight_mon.w[i][0]
        final = weight_mon.w[i][-1]
        print(f"  Синапс {i}: {initial:.3f} -> {final:.3f} (Δ {final-initial:+.3f})")

print("✓ Модель завершена успешно!")
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Подготовка временных рядов для интерполяции
# ===========================
t_vector = np.arange(0, duration/ms, 0.1)  # шаг 0.1 ms
v_main = np.zeros((N_neurons, len(t_vector)))
v_pred = np.zeros((N_pred, len(t_vector)))

# Основная сеть: создаём бинарный сигнал спайков (0/1) на временной сетке
for i in range(N_neurons):
    spikes_i = spike_mon_neurons.spike_trains()[i]/ms
    v_main[i, :] = np.histogram(spikes_i, bins=np.append(t_vector, t_vector[-1]+0.1))[0]

# Генеративная сеть
for i in range(N_pred):
    spikes_i = spike_mon_pred.spike_trains()[i]/ms
    v_pred[i, :] = np.histogram(spikes_i, bins=np.append(t_vector, t_vector[-1]+0.1))[0]

# Ошибка предсказания (разница генеративной сети и основной)
error = v_main - v_pred[:N_neurons, :]

# ===========================
# Создаём фигуру
# ===========================
plt.figure(figsize=(12, 8))

# Спайки основной сети
plt.subplot(4, 1, 1)
for i in range(N_neurons):
    spikes_i = spike_mon_neurons.spike_trains()[i]
    plt.vlines(spikes_i/ms, i + 0.5, i + 1.5, color='b')
plt.title("Спайки основной сети (G)")
plt.ylabel("Нейроны")
plt.xlim(0, duration/ms)
plt.xticks([])

# Спайки генеративной сети
plt.subplot(4, 1, 2)
for i in range(N_pred):
    spikes_i = spike_mon_pred.spike_trains()[i]
    plt.vlines(spikes_i/ms, i + 0.5, i + 1.5, color='r')
plt.title("Спайки генеративной сети (G_pred)")
plt.ylabel("Генер. нейроны")
plt.xlim(0, duration/ms)
plt.xticks([])

# LTM состояние
plt.subplot(4, 1, 3)
for i in range(N_pred):
    plt.plot(ltm_mon.t/ms, ltm_mon.LTM[i], label=f'LTM нейрон {i}')
plt.title("Состояние LTM в основной сети")
plt.xlabel("Время (ms)")
plt.ylabel("LTM")
plt.xlim(0, duration/ms)
plt.legend()

# Ошибка предсказания
plt.subplot(4, 1, 4)
for i in range(N_neurons):
    plt.plot(t_vector, error[i], label=f'Ошибка нейрон {i}')
plt.title("Ошибка предсказания (G_pred vs G)")
plt.xlabel("Время (ms)")
plt.ylabel("Ошибка")
plt.xlim(0, duration/ms)
plt.legend()

plt.tight_layout()

# ===========================
# Сохраняем в файл PNG
# ===========================
plt.savefig("snn_proto_self_with_error.png", dpi=300)
print("График с ошибкой предсказания сохранён в snn_proto_self_with_error.png")
plt.close()
