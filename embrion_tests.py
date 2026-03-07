from brian2 import *
import numpy as np

print("Эмбриональный режим Σ_slow — три теста по каналам A/B (OU-шума)")

# --------------------------
# Общие параметры
# --------------------------
N = 50
tau_m = 50*ms
tau_w = 500*ms
EL = -70*mV
VT = -60*mV
Vr = -65*mV
R = 150*Mohm

a_adapt = 0.5*pA/mV
b_adapt = 0.5*pA

# OU-процессы
I_endo_mean = 50*pA
I_endo_std  = 40*pA
tau_endo = 300*ms

sigma_mod = 30*pA
tau_mod = 300*ms

# Время симуляции
sim_time = 5*second

# --------------------------
# Функция запуска теста
# --------------------------
def run_test(alphaA, alphaB, filename, test_name):
    print(f"\n=== {test_name} ===")

    start_scope()

    eqs = f'''
    dv/dt = (EL - v + R*(I_endo + {alphaA}*I_mod_A + {alphaB}*I_mod_B - w)) / tau_m : volt
    dw/dt = (a_adapt*(v - EL) - w)/tau_w : amp

    dI_endo/dt = -(I_endo - I_endo_mean)/tau_endo + (I_endo_std*sqrt(2/tau_endo))*xi_endo : amp
    dI_mod_A/dt = -I_mod_A/tau_mod + (sigma_mod*sqrt(2/tau_mod))*xi_A : amp
    dI_mod_B/dt = -I_mod_B/tau_mod + (sigma_mod*sqrt(2/tau_mod))*xi_B : amp
    '''

    # Создание NeuronGroup
    Sigma_slow = NeuronGroup(
        N,
        model=eqs,
        threshold='v > VT',
        reset='v = Vr; w += b_adapt',
        method='euler'
    )

    # Инициализация
    Sigma_slow.v = EL
    Sigma_slow.w = 0*pA
    Sigma_slow.I_endo = I_endo_mean
    Sigma_slow.I_mod_A = 0*pA
    Sigma_slow.I_mod_B = 0*pA

    # Слабая внутренняя связность
    S_rec = Synapses(Sigma_slow, Sigma_slow, on_pre='v_post += 0.1*mV')
    S_rec.connect(p=0.1)

    # Мониторы
    M_spike = SpikeMonitor(Sigma_slow)
    M_v = StateMonitor(Sigma_slow, 'v', record=True)
    M_w = StateMonitor(Sigma_slow, 'w', record=range(min(3, N)))

    # Запуск сети
    net = Network(Sigma_slow, S_rec, M_spike, M_v, M_w)
    net.run(sim_time)

    # --------------------------
    # Результаты
    # --------------------------
    print(f"Общее число спайков: {len(M_spike.t)}")
    avg_rate = len(M_spike.t) / (N * sim_time/second)
    print(f"Средняя частота: {avg_rate:.2f} Hz/нейрон")

    avg_v = np.mean(M_v.v)
    std_v = np.std(M_v.v)
    min_v = np.min(M_v.v)
    max_v = np.max(M_v.v)
    print(f"Потенциал: средний {avg_v/mV:.2f} ± {std_v/mV:.2f} mV, диапазон {min_v/mV:.2f} .. {max_v/mV:.2f} mV")

    for i in range(min(3, N)):
        avg_w = np.mean(M_w.w[i])
        print(f"Адаптация нейрона {i}: {avg_w/pA:.2f} pA")

    # Сохранение данных
    np.savez(filename,
             spike_times=np.array(M_spike.t),
             spike_indices=np.array(M_spike.i),
             voltages=np.array(M_v.v),
             voltage_times=np.array(M_v.t),
             adaptation=np.array(M_w.w),
             adaptation_times=np.array(M_w.t))

    print(f"Данные сохранены в '{filename}'")

# --------------------------
# Тесты
# --------------------------
run_test(alphaA=0, alphaB=0, filename='test0.npz', test_name='Тест 0: Оба канала отключены')
run_test(alphaA=1, alphaB=0, filename='testA.npz', test_name='Тест 1: Только канал A')
run_test(alphaA=0, alphaB=1, filename='testB.npz', test_name='Тест 2: Только канал B')
run_test(alphaA=1, alphaB=1, filename='testAB.npz', test_name='Тест 3: Каналы A + B')