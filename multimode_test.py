from brian2 import *
import numpy as np

print("Шаг 2: Подключение Poisson нейронов к Σ_slow")

# Параметры - оптимизированные для активации
defaultclock.dt = 0.1*ms

# Размерности
N_channels = 3
N_per_channel = 30  # Меньше для тестирования
N_slow = 25
N_inh = 1

# Временные масштабы - оптимизированы для передачи
tau_slow = 100*ms   # Быстрее для реакции
tau_syn_slow = 50*ms  # Быстрее синапсы
tau_inh = 50*ms

v_rest = -70*mV
v_th_slow = -55*mV  # Более доступный порог
v_reset = -65*mV

# Создаем Poisson группы для каждого канала
channels = []
synapses_input_slow = []

# Модулированные частоты для тестирования корреляций
base_pattern = np.hstack([
    np.ones(500)*30,    # 0.5 сек: базовая активность
    np.ones(250)*60,    # 0.25 сек: высокая активность (паттерн 1)
    np.ones(500)*30,    # 0.5 сек: базовая
    np.ones(250)*20,    # 0.25 сек: низкая активность (паттерн 2)
    np.ones(500)*30     # 0.5 сек: базовая
])

for ch in range(N_channels):
    # Каждый канал имеет немного разные временные паттерны для корреляций
    phase_shift = ch * 100  # Смещение фазы для разных каналов
    rates_array = np.roll(np.tile(base_pattern, 3), phase_shift)[:2000]  # 2 секунды
    
    rates_timed = TimedArray(rates_array*Hz, dt=1*ms)
    channel_group = PoissonGroup(N_per_channel, rates='rates_timed(t)', name=f'channel_{ch}')
    channels.append(channel_group)

# Σ_slow нейроны
slow_eqs = '''
dv/dt = (v_rest - v + I_syn - I_inh)/tau_slow : volt
dI_syn/dt = -I_syn/tau_syn_slow : volt
dI_inh/dt = -I_inh/tau_inh : volt
'''

sigma_slow = NeuronGroup(N_slow,
                        slow_eqs,
                        threshold='v > v_th_slow',
                        reset='v = v_reset',
                        method='euler',
                        name='Sigma_slow')
sigma_slow.v = v_rest - 5*mV  # Начинаем ближе к порогу

# Ингибирующий нейрон (минимальный)
inh = NeuronGroup(N_inh,
                 'dv/dt = (v_rest - v)/tau_inh : volt',
                 threshold='v > -55*mV',
                 reset='v = v_rest',
                 method='euler')
inh.v = v_rest

# Подключение Poisson → Σ_slow (сильное)
for ch, channel_group in enumerate(channels):
    S_input = Synapses(channel_group, sigma_slow,
                      on_pre='I_syn += 2.0*mV')  # Сильные веса
    S_input.connect(p=0.4)  # Хорошее покрытие
    synapses_input_slow.append(S_input)

# Минимальное ингибирование (только для стабилизации)
S_slow_inh = Synapses(sigma_slow, inh, on_pre='v += 0.8*mV')
S_slow_inh.connect()

S_inh_slow = Synapses(inh, sigma_slow, on_pre='I_inh += 0.3*mV')  
S_inh_slow.connect()

# Мониторы
M_slow = StateMonitor(sigma_slow, 'v', record=True)
Sp_slow = SpikeMonitor(sigma_slow)
Sp_channels = [SpikeMonitor(ch) for ch in channels]

# Создаем сеть
net = Network()
for ch in channels:
    net.add(ch)
net.add(sigma_slow, inh, *synapses_input_slow, S_slow_inh, S_inh_slow)
net.add(M_slow, Sp_slow, *Sp_channels)

# Запуск
print("Запуск полной архитектуры...")
net.run(2*second)

print(f"\nРезультаты Step 2:")
print(f"- Число спайков Σ_slow: {len(Sp_slow.t)}")

if len(Sp_slow.t) > 0:
    print("✓ SUCCESS: Σ_slow начал спайковать!")
    print(f"  Первые спайки: {[f'{t/ms:.1f} ms' for t in Sp_slow.t[:5]]}")
    
    # Анализ частоты
    avg_rate = len(Sp_slow.t) / (N_slow * 2.0)
    print(f"  Средняя частота Σ_slow: {avg_rate:.2f} Hz/нейрон")
    
else:
    print("✗ ПРОБЛЕМА: Σ_slow не активировался")
    
    # Анализ потенциалов
    if len(M_slow.v) > 0:
        max_v = np.max(M_slow.v)
        avg_v = np.mean(M_slow.v)
        print(f"  Максимальный потенциал: {max_v/mV:.2f} mV")
        print(f"  Средний потенциал: {avg_v/mV:.2f} mV")
        
        # Проверим, был ли хоть какой-то отклик
        above_baseline = np.sum(M_slow.v > (v_rest + 2*mV))
        if above_baseline > 0:
            print(f"  Точки выше baseline: {above_baseline} ({above_baseline/len(M_slow.v.flatten())*100:.1f}%)")

# Сохраняем данные
save_data = {
    'slow_v': np.array(M_slow.v),
    'slow_t': np.array(M_slow.t),
    'slow_spikes': np.array(Sp_slow.t),
    'slow_spike_indices': np.array(Sp_slow.i),
}

# Данные по каналам
for i, sp_monitor in enumerate(Sp_channels):
    save_data[f'channel_{i}_spikes'] = np.array(sp_monitor.t)
    save_data[f'channel_{i}_indices'] = np.array(sp_monitor.i)

np.savez('step2_full_architecture.npz', **save_data)
print("Step 2 завершен. Данные сохранены в 'step2_full_architecture.npz'")
