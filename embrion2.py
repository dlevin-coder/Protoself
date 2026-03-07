from brian2 import *
import numpy as np

print("Эмбриональный режим Σ_slow — исправленный с мультимодальными входами")

start_scope()

# -----------------------------
# Параметры нейронов
# -----------------------------
N = 50

tau_m = 50*ms
tau_w = 500*ms

EL = -70*mV
VT = -55*mV
Vr = -60*mV

a_adapt = 0.5*pA/mV  # адаптационный ток на мВ
b_adapt = 0.5*pA

R = 100*Mohm  # мембранное сопротивление

# -----------------------------
# Эндогенный ток и шум
# -----------------------------
I_endo_mean = 15*pA
I_endo_std  = 5*pA
tau_endo = 300*ms

sigma_noise = 1*pA
tau_noise = 50*ms

sigma_mod = 1.5*pA
tau_mod   = 300*ms

# -----------------------------
# Уравнения
# -----------------------------
eqs = '''
dv/dt = (EL - v + R*(I_noise + I_endo + alpha_A*I_mod_A + alpha_B*I_mod_B - w)) / tau_m : volt

dw/dt = (a_adapt*(v - EL) - w) / tau_w : amp

dI_noise/dt = -I_noise/tau_noise + sigma_noise*sqrt(2/tau_noise)*xi_noise : amp
dI_endo/dt  = -(I_endo - I_endo_mean)/tau_endo + I_endo_std*sqrt(2/tau_endo)*xi_endo : amp
dI_mod_A/dt = -I_mod_A/tau_mod + sigma_mod*sqrt(2/tau_mod)*xi_A : amp
dI_mod_B/dt = -I_mod_B/tau_mod + sigma_mod*sqrt(2/tau_mod)*xi_B : amp

alpha_A : 1
alpha_B : 1
'''

# -----------------------------
# Создание NeuronGroup
# -----------------------------
Sigma_slow = NeuronGroup(
    N,
    eqs,
    threshold='v > VT',
    reset='v = Vr; w += b_adapt',
    method='euler'
)

# Инициализация
Sigma_slow.v = EL
Sigma_slow.w = 0*pA
Sigma_slow.I_noise = 0*pA
Sigma_slow.I_endo = I_endo_mean
Sigma_slow.I_mod_A = 0*pA
Sigma_slow.I_mod_B = 0*pA
Sigma_slow.alpha_A = 1
Sigma_slow.alpha_B = 1

# -----------------------------
# Слабая внутренняя связность
# -----------------------------
S_rec = Synapses(
    Sigma_slow, Sigma_slow,
    on_pre='I_mod_A_post += 0.1*pA'
)
S_rec.connect(p=0.1)

# -----------------------------
# Мониторы
# -----------------------------
M_spike = SpikeMonitor(Sigma_slow)
M_state = StateMonitor(Sigma_slow, 'v', record=True)
M_adapt = StateMonitor(Sigma_slow, 'w', record=range(min(3, N)))
M_Iendo = StateMonitor(Sigma_slow, 'I_endo', record=range(min(3, N)))
M_IA = StateMonitor(Sigma_slow, 'I_mod_A', record=range(min(3, N)))
M_IB = StateMonitor(Sigma_slow, 'I_mod_B', record=range(min(3, N)))

# -----------------------------
# Эндогенный фоновый дрейв (дополнительно)
# -----------------------------
@network_operation(dt=10*ms)
def endogenous_drive():
    Sigma_slow.I_endo = I_endo_mean + I_endo_std * randn(N)

# -----------------------------
# Создаем сеть
# -----------------------------
net = Network()
net.add(Sigma_slow, S_rec, M_spike, M_state, M_adapt, M_Iendo, M_IA, M_IB, endogenous_drive)

# -----------------------------
# Запуск
# -----------------------------
print("Запуск эмбрионального режима...")
net.run(5*second)

# -----------------------------
# Результаты
# -----------------------------
print(f"\nРезультаты эмбрионального режима:")

# Спайки
print(f"Общее число спайков: {len(M_spike.t)}")
avg_rate = len(M_spike.t) / (N * 5.0)
print(f"Средняя частота: {avg_rate:.2f} Hz/нейрон")

# Потенциалы
if len(M_state.v) > 0:
    avg_v = np.mean(M_state.v)
    std_v = np.std(M_state.v)
    min_v = np.min(M_state.v)
    max_v = np.max(M_state.v)
    print(f"Потенциалы: средний {avg_v/mV:.2f} ± {std_v/mV:.2f} mV, диапазон {min_v/mV:.2f}..{max_v/mV:.2f} mV")

# Адаптация
if len(M_adapt.w) > 0:
    for i in range(min(3, len(M_adapt.w))):
        avg_w = np.mean(M_adapt.w[i])
        print(f"Адаптация нейрона {i}: {avg_w/pA:.2f} pA")

# -----------------------------
# Проверка критериев
# -----------------------------
issues = []
if avg_rate < 0.1:
    issues.append(f"Слишком низкая частота {avg_rate:.2f} Hz (< 0.1 Hz)")
if avg_rate > 20:
    issues.append(f"Слишком высокая частота {avg_rate:.2f} Hz (> 20 Hz)")
if avg_v < -65*mV or avg_v > -55*mV:
    issues.append(f"Потенциал {avg_v/mV:.2f} mV вне диапазона -65..-55 mV")

print(f"\nКритерии успеха:")
if len(issues) == 0:
    print("✓ УСПЕХ: все основные критерии выполнены")
else:
    print("⚠ ПРОБЛЕМЫ:")
    for issue in issues:
        print(f" - {issue}")

# -----------------------------
# Сохранение данных
# -----------------------------
np.savez('embryonic_self.npz',
         spike_times=np.array(M_spike.t),
         spike_indices=np.array(M_spike.i),
         voltages=np.array(M_state.v),
         voltage_times=np.array(M_state.t),
         adaptation=np.array(M_adapt.w),
         I_endo=np.array(M_Iendo.I_endo),
         I_mod_A=np.array(M_IA.I_mod_A),
         I_mod_B=np.array(M_IB.I_mod_B)
         )

print("Эмбриональный режим завершен. Данные сохранены в 'embryonic_self.npz'")