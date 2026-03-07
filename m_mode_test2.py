from brian2 import *
import numpy as np

print("Шаг 3: Тест сохранения состояния Σ_slow при обрыве входа")

# Параметры
defaultclock.dt = 0.1*ms

# Размерности
N_channels = 3
N_per_channel = 30
N_slow = 25
N_inh = 1

# Временные масштабы
tau_slow = 300*ms
tau_syn_slow = 100*ms
tau_inh = 100*ms

v_rest = -70*mV
v_th_slow = -58*mV
v_reset = -65*mV

# Создаем отдельные TimedArray для каждого канала
# Сценарий: 0-3с активность, 3-5с обрыв, 5-8с восстановление

# Генерируем паттерны активности
t_total_ms = 8000  # 8 секунд в миллисекундах
time_points = np.arange(t_total_ms)

# Базовый паттерн для всех каналов
base_pattern = np.ones(t_total_ms) * 30  # 30 Hz базовая

# Модуляции для разных каналов
pattern_0 = base_pattern.copy()
pattern_1 = base_pattern.copy() 
pattern_2 = base_pattern.copy()

# Канал 0: повышенная активность 1-2с
pattern_0[1000:2000] = 50

# Канал 1: повышенная активность 2-3с
pattern_1[2000:3000] = 50

# Канал 2: высокая активность после восстановления 5-7с
pattern_2[5000:7000] = 60

# Обрыв 3-5с для всех
for pattern in [pattern_0, pattern_1, pattern_2]:
    pattern[3000:5000] = 0  # Обнуляем во время обрыва

# Создаем TimedArray напрямую (без списков)
rates_0 = TimedArray(pattern_0*Hz, dt=1*ms)
rates_1 = TimedArray(pattern_1*Hz, dt=1*ms) 
rates_2 = TimedArray(pattern_2*Hz, dt=1*ms)

# Создаем Poisson группы с фиксированными именами
channel_0 = PoissonGroup(N_per_channel, rates='rates_0(t)', name='channel_0')
channel_1 = PoissonGroup(N_per_channel, rates='rates_1(t)', name='channel_1')
channel_2 = PoissonGroup(N_per_channel, rates='rates_2(t)', name='channel_2')
channels = [channel_0, channel_1, channel_2]

# Σ_slow с медленной динамикой
slow_eqs = '''
dv/dt = (v_rest - v + I_syn - I_inh)/tau_slow : volt
dI_syn/dt = -I_syn/tau_syn_slow : volt
dI_inh/dt = -I_inh/tau_inh : volt
'''

sigma_slow = NeuronGroup(N_slow,
                        slow_eqs,
                        threshold='v > v_th_slow',
                        reset='v = v_reset',
                        method='euler')
sigma_slow.v = v_rest

# Ингибирование
inh = NeuronGroup(N_inh,
                 'dv/dt = (v_rest - v)/tau_inh : volt',
                 threshold='v > -55*mV',
                 reset='v = v_rest')
inh.v = v_rest

# Подключение с умеренными весами
S_0_slow = Synapses(channel_0, sigma_slow, on_pre='I_syn += 1.2*mV')
S_0_slow.connect(p=0.3)

S_1_slow = Synapses(channel_1, sigma_slow, on_pre='I_syn += 1.2*mV')
S_1_slow.connect(p=0.3)

S_2_slow = Synapses(channel_2, sigma_slow, on_pre='I_syn += 1.2*mV')
S_2_slow.connect(p=0.3)

synapses = [S_0_slow, S_1_slow, S_2_slow]

# Ингибирование
S_slow_inh = Synapses(sigma_slow, inh, on_pre='v += 0.5*mV')
S_slow_inh.connect()

S_inh_slow = Synapses(inh, sigma_slow, on_pre='I_inh += 0.4*mV')
S_inh_slow.connect()

# Мониторы
M_slow = StateMonitor(sigma_slow, 'v', record=True)
Sp_slow = SpikeMonitor(sigma_slow)
M_inh = StateMonitor(inh, 'v', record=0)

# Сеть
net = Network()
net.add(channel_0, channel_1, channel_2)
net.add(sigma_slow, inh)
net.add(*synapses)
net.add(S_slow_inh, S_inh_slow)
net.add(M_slow, Sp_slow, M_inh)

# Запуск
print("Запуск теста сохранения состояния...")
net.run(8*second)

print(f"\nРезультаты Step 3:")

# Анализ по периодам
t_seconds = np.array(M_slow.t/second)

print("Анализ активности Σ_slow по периодам:")
periods_analysis = [
    ("до обрыва (0-3с)", 0, 3),
    ("во время обрыва (3-5с)", 3, 5), 
    ("после восстановления (5-8с)", 5, 8)
]

for name, start, end in periods_analysis:
    mask = (t_seconds >= start) & (t_seconds < end)
    if np.any(mask):
        # Спайки в периоде
        spike_mask = (Sp_slow.t/second >= start) & (Sp_slow.t/second < end)
        spike_count = np.sum(spike_mask)
        
        # Потенциалы в периоде
        avg_potential = np.mean(M_slow.v[:, mask])
        max_potential = np.max(M_slow.v[:, mask])
        
        duration = end - start
        avg_spike_rate = spike_count / (N_slow * duration) if duration > 0 else 0
        
        print(f"  {name}:")
        print(f"    Спайки: {spike_count}, частота: {avg_spike_rate:.2f} Hz/нейрон")
        print(f"    Средний потенциал: {avg_potential/mV:.2f} mV")
        print(f"    Макс. потенциал: {max_potential/mV:.2f} mV")

# Ключевой тест: сохранение во время обрыва
break_mask = (t_seconds >= 3) & (t_seconds < 5)
if np.any(break_mask):
    break_potentials = M_slow.v[:, break_mask]
    avg_break_pot = np.mean(break_potentials)
    max_break_pot = np.max(break_potentials)
    
    print(f"\nКлючевой тест сохранения (3-5с):")
    print(f"  Средний потенциал во время обрыва: {avg_break_pot/mV:.2f} mV")
    
    # Сравнение с другими периодами
    before_mask = (t_seconds >= 2) & (t_seconds < 3)  # Перед обрывом
    after_mask = (t_seconds >= 5) & (t_seconds < 6)   # Сразу после
    
    if np.any(before_mask) and np.any(after_mask):
        before_pot = np.mean(M_slow.v[:, before_mask])
        after_pot = np.mean(M_slow.v[:, after_mask])
        
        print(f"  Сравнение: до={before_pot/mV:.2f} mV, обрыв={avg_break_pot/mV:.2f} mV, после={after_pot/mV:.2f} mV")
        
        # Если потенциал во время обрыва выше resting (-70mV), это признак сохранения
        if avg_break_pot > (v_rest + 2*mV):
            print("  ✓ ПОЛОЖИТЕЛЬНЫЙ РЕЗУЛЬТАТ: Σ_slow сохраняет активированное состояние!")
        else:
            print("  ⚠ РЕЗУЛЬТАТ: Низкая активность во время обрыва")

# Сохранение данных
np.savez('step3_state_preservation.npz',
         slow_v=np.array(M_slow.v),
         slow_t=np.array(M_slow.t),
         spike_times=np.array(Sp_slow.t),
         spike_indices=np.array(Sp_slow.i),
         inh_v=np.array(M_inh.v),
         inh_t=np.array(M_inh.t))

print("Step 3 завершен. Данные сохранены в 'step3_state_preservation.npz'")
