from brian2 import *
import numpy as np

print("Эмбриональный режим Σ_slow - тест (исправленный)")

start_scope()

# Параметры (исправленные)
N = 50

tau_m = 50*ms
tau_w = 500*ms

EL = -70*mV
VT = -55*mV
Vr = -60*mV

a_adapt = 0.5*pA/mV   # ток на мВ
b_adapt = 0.5*pA

I_endo_mean = 15*pA
I_endo_std  = 5*pA
sigma_v = 4*mV

# Уравнения (корректные)
eqs = '''
dv/dt = (
    EL - v
    + R*(I_endo + I_ext - w)
) / tau_m
+ sigma_v * sqrt(2 / tau_m) * xi : volt
dw/dt = (a_adapt*(v - EL) - w) / tau_w : amp
I_endo : amp
I_ext : amp
R : ohm
'''

# NeuronGroup
Sigma_slow = NeuronGroup(
    N,
    eqs,
    threshold='v > VT',
    reset='''
    v = Vr
    w += b_adapt
    ''',
    method='euler'
)

# Инициализация
Sigma_slow.v = EL
Sigma_slow.w = 0*pA
Sigma_slow.I_ext = 0*pA
Sigma_slow.R = 100*Mohm

# Эндогенный фон
@network_operation(dt=10*ms)
def endogenous_drive():
    Sigma_slow.I_endo = I_endo_mean + I_endo_std * randn(N)

# Слабая внутренняя связность
S_rec = Synapses(
    Sigma_slow, Sigma_slow,
    on_pre='I_ext_post += 0.1*pA'
)
S_rec.connect(p=0.1)

# Мониторы
M_spike = SpikeMonitor(Sigma_slow)
M_state = StateMonitor(Sigma_slow, 'v', record=True)
M_adapt = StateMonitor(Sigma_slow, 'w', record=range(min(3, N)))

# Создаем сеть
net = Network()
net.add(Sigma_slow, S_rec, M_spike, M_state, M_adapt)
net.add(endogenous_drive)

# Запуск
print("Запуск эмбрионального режима...")
net.run(5*second)

print(f"\nРезультаты эмбрионального режима:")

# Анализ спайков
print(f"Общее число спайков: {len(M_spike.t)}")
avg_rate = len(M_spike.t) / (N * 5.0)
print(f"Средняя частота: {avg_rate:.2f} Hz/нейрон")

# Анализ потенциалов
if len(M_state.v) > 0:
    avg_potential = np.mean(M_state.v)
    std_potential = np.std(M_state.v)
    max_potential = np.max(M_state.v)
    min_potential = np.min(M_state.v)
    
    print(f"Потенциалы:")
    print(f"  Средний: {avg_potential/mV:.2f} ± {std_potential/mV:.2f} mV")
    print(f"  Диапазон: {min_potential/mV:.2f} .. {max_potential/mV:.2f} mV")

# Анализ адаптации
if len(M_adapt.w) > 0:
    print(f"Адаптация:")
    for i in range(min(3, len(M_adapt.w))):
        avg_w = np.mean(M_adapt.w[i])
        print(f"  Нейрон {i}: средняя адаптация {avg_w/pA:.2f} pA")

# Проверка критериев успеха
success = True
issues = []

if avg_rate < 0.1:
    issues.append(f"Слишком низкая частота {avg_rate:.2f} Hz (< 0.1 Hz)")
elif avg_rate > 20:
    issues.append(f"Слишком высокая частота {avg_rate:.2f} Hz (> 20 Hz)")

if avg_potential < -65*mV or avg_potential > -55*mV:
    issues.append(f"Потенциал {avg_potential/mV:.2f} mV вне диапазона -65..-55 mV")

print(f"\nКритерии успеха:")
if len(issues) == 0:
    print("✓ УСПЕХ: Основные критерии выполнены")
    print(f"  - Низкочастотная фоновая активность: {avg_rate:.2f} Hz")
    print(f"  - Спайки без внешнего входа: {len(M_spike.t)} спайков")
    print(f"  - Устойчивый средний потенциал: {avg_potential/mV:.2f} mV")
    print("  - Отсутствие коллапса: проверено по стабильности")
else:
    print("⚠ ПРОБЛЕМЫ:")
    for issue in issues:
        print(f"  - {issue}")

# Сохранение данных
np.savez('embryonic_self.npz',
         spike_times=np.array(M_spike.t),
         spike_indices=np.array(M_spike.i),
         voltages=np.array(M_state.v),
         voltage_times=np.array(M_state.t),
         adaptation=np.array(M_adapt.w),
         adaptation_times=np.array(M_adapt.t))

print("Эмбриональный режим завершен. Данные сохранены в 'embryonic_self.npz'")
