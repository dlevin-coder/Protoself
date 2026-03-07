from brian2 import *
import numpy as np

print("Эмбриональный режим Σ_slow — три теста по каналам A/B (OU-шума) — Шаг 4")

# --------------------------
# Общие параметры
# --------------------------
N = 50
tau_m = 50*ms
tau_w = 500*ms
EL = -70*mV
VT = -60*mV
VT_base = -60*mV  # базовый порог
Vr = -65*mV
R = 150*Mohm
tau_VT = 50*ms

a_adapt = 0.5*pA/mV
b_adapt = 0.5*pA

# OU-процессы
I_endo_mean = 50*pA
I_endo_std  = 40*pA
tau_endo = 300*ms

sigma_mod = 30*pA
tau_mod = 300*ms

# --------------------------
# Σ_fast параметры (наблюдатель)
# --------------------------
N_fast = 50

tau_m_fast = 10*ms
tau_w_fast = 100*ms

EL_fast = -70*mV
VT_fast = -60*mV
Vr_fast = -65*mV
R_fast = 100*Mohm

a_fast = 0.0*pA/mV
b_fast = 0.2*pA

sigma_fast = 20*pA     # масштаб шума
tau_fast = 50*ms
tau_fast_avg = 300*ms

# --------------------------
# Temporal pattern parameters
# --------------------------

pattern_A_amp = 150*pA
pattern_B_amp = 120*pA

pattern_A_period = 400*ms
pattern_B_period = 250*ms

pattern_A_width = 100*ms
pattern_B_width = 50*ms

# Время симуляции
sim_time = 5*second

# --------------------------
# Функция запуска теста
# --------------------------
def run_test(alphaA, alphaB, filename, test_name):
    print(f"\n=== {test_name} ===")
    start_scope()

    # --------------------------
    # Σ_fast — чистый наблюдатель
    # --------------------------
    eqs_fast = '''
    dv/dt = (EL_fast - v + R_fast*(I_fast_A + I_fast_B + I_pattern_A + I_pattern_B - w)) / tau_m_fast : volt
    dw/dt = (a_fast*(v - EL_fast) - w)/tau_w_fast : amp

    dI_fast_A/dt = -I_fast_A/tau_fast + (sigma_fast*sqrt(2/tau_fast))*xi_fA : amp
    dI_fast_B/dt = -I_fast_B/tau_fast + (sigma_fast*sqrt(2/tau_fast))*xi_fB : amp

    ds_fast/dt = -s_fast / tau_fast_avg : 1  # экспоненциальное усреднение спайков
    I_pattern_A : amp
    I_pattern_B : amp
    '''

    Sigma_fast = NeuronGroup(
        N_fast,
        model=eqs_fast,
        threshold='v > VT_fast',
        reset='v = Vr_fast; w += b_fast; s_fast += 1.0',
        method='euler'
    )

    Sigma_fast.v = EL_fast
    Sigma_fast.w = 0*pA
    Sigma_fast.I_fast_A = 0*pA
    Sigma_fast.I_fast_B = 0*pA
    Sigma_fast.s_fast = 0.0
    Sigma_fast.I_pattern_A = 0*pA
    Sigma_fast.I_pattern_B = 0*pA

    # --------------------------
    # Σ_slow — эмбриональный слой
    # --------------------------
    eqs_slow = '''
    dv/dt = (EL - v + R*(I_endo + alphaA*I_mod_A + alphaB*I_mod_B + I_pattern_A + I_pattern_B - w)) / tau_m : volt
    dw/dt = ((a_adapt*(v - EL) - w) + I_fast_mod) / tau_w : amp

    dVT_dyn/dt = (VT_base - VT_dyn)/tau_VT : volt

    dI_endo/dt = -(I_endo - I_endo_mean)/tau_endo + (I_endo_std*sqrt(2/tau_endo))*xi_endo : amp
    dI_mod_A/dt = -I_mod_A/tau_mod + (sigma_mod*sqrt(2/tau_mod))*xi_A : amp
    dI_mod_B/dt = -I_mod_B/tau_mod + (sigma_mod*sqrt(2/tau_mod))*xi_B : amp

    I_fast_mod : amp  # модифицирует dw
    I_pattern_A : amp
    I_pattern_B : amp
    '''

    Sigma_slow = NeuronGroup(
        N,
        model=eqs_slow,
        threshold='v > VT_dyn',
        reset='v = Vr; w += b_adapt',
        method='euler'
    )

    # Слабая внутренняя связность Σ_slow
    S_rec = Synapses(Sigma_slow, Sigma_slow, on_pre='v_post += 0.1*mV')
    S_rec.connect(p=0.1)

    # Инициализация
    Sigma_slow.v = EL
    Sigma_slow.w = 0*pA
    Sigma_slow.I_endo = I_endo_mean
    Sigma_slow.I_mod_A = 0*pA
    Sigma_slow.I_mod_B = 0*pA
    Sigma_slow.VT_dyn = VT_base
    Sigma_slow.I_fast_mod = 0*pA
    Sigma_slow.I_pattern_A = 0*pA
    Sigma_slow.I_pattern_B = 0*pA

    # --------------------------
    # Модуль Σ_fast → Σ_slow
    # --------------------------
    Delta_w = 0.05*pA  # маленький ток для мягкой модуляции

    @network_operation(dt=10*ms)
    def fast_to_slow_modulation():
        mean_fast = np.mean(Sigma_fast.s_fast)
        Sigma_slow.I_fast_mod = Delta_w * mean_fast

    @network_operation(dt=1*ms)
    def temporal_patterns():

        t_now = defaultclock.t

        # Pattern A
        phaseA = t_now % pattern_A_period
        if phaseA < pattern_A_width:
            Sigma_slow.I_pattern_A = pattern_A_amp
            Sigma_fast.I_pattern_A = pattern_A_amp
        else:
            Sigma_slow.I_pattern_A = 0*pA
            Sigma_fast.I_pattern_A = 0*pA

        # Pattern B
        phaseB = t_now % pattern_B_period
        if phaseB < pattern_B_width:
            Sigma_slow.I_pattern_B = pattern_B_amp
            Sigma_fast.I_pattern_B = pattern_B_amp
        else:
            Sigma_slow.I_pattern_B = 0*pA
            Sigma_fast.I_pattern_B = 0*pA
    # --------------------------
    # Мониторы
    # --------------------------
    M_spike = SpikeMonitor(Sigma_slow)
    M_v = StateMonitor(Sigma_slow, 'v', record=True)
    M_w = StateMonitor(Sigma_slow, 'w', record=range(min(3, N)))

    M_fast_spike = SpikeMonitor(Sigma_fast)
    M_fast_v = StateMonitor(Sigma_fast, 'v', record=range(min(10, N_fast)))
    M_VT = StateMonitor(Sigma_slow, 'VT_dyn', record=range(3))
    M_rate = PopulationRateMonitor(Sigma_slow)

    # --------------------------
    # Запуск сети
    # --------------------------
    net = Network(Sigma_slow, Sigma_fast, S_rec,
              M_spike, M_v, M_w, M_fast_spike, M_fast_v, M_VT, M_rate,
              fast_to_slow_modulation,
              temporal_patterns)
    net.run(sim_time)

    # --------------------------
    # Результаты и сохранение
    # --------------------------
    print(f"Общее число спайков: {len(M_spike.t)}")
    avg_rate = len(M_spike.t) / (N * sim_time/second)
    print(f"Средняя частота: {avg_rate:.2f} Hz/нейрон")

    avg_v = np.mean(M_v.v)
    std_v = np.std(M_v.v)
    min_v = np.min(M_v.v)
    max_v = np.max(M_v.v)
    print(f"Потенциал: средний {avg_v/mV:.2f} ± {std_v/mV:.2f} mV, диапазон {min_v/mV:.2f} .. {max_v/mV:.2f} mV")

    fast_rate = len(M_fast_spike.t) / (N_fast * sim_time/second)
    print(f"Σ_fast: средняя частота {fast_rate:.2f} Hz/нейрон")

    print("VT_dyn (нейрон 0):",
          np.mean(M_VT.VT_dyn[0])/mV,
          "±",
          np.std(M_VT.VT_dyn[0])/mV)

    for i in range(min(3, N)):
        avg_w = np.mean(M_w.w[i])
        print(f"Адаптация нейрона {i}: {avg_w/pA:.2f} pA")

    # --------------------------
    # Population synchrony
    # --------------------------

    rate = M_rate.smooth_rate(window='gaussian', width=5*ms) / Hz

    sync_index = np.var(rate) / np.mean(rate)

    print(f"Population synchrony index: {sync_index:.3f}")    

    # Сохранение в файл
    np.savez(filename,
             spike_times=np.array(M_spike.t),
             spike_indices=np.array(M_spike.i),
             voltages=np.array(M_v.v),
             voltage_times=np.array(M_v.t),
             adaptation=np.array(M_w.w),
             adaptation_times=np.array(M_w.t),
             population_rate=np.array(rate),
             synchrony=sync_index,
             fast_spikes=np.array(M_fast_spike.t),
             fast_indices=np.array(M_fast_spike.i),
             VT_dyn=np.array(M_VT.VT_dyn),
             VT_times=np.array(M_VT.t)
            )

    print(f"Данные сохранены в '{filename}'")


# --------------------------
# Тесты
# --------------------------
run_test(alphaA=0, alphaB=0, filename='test0.npz', test_name='Тест 0: Оба канала отключены')
run_test(alphaA=1, alphaB=0, filename='testA.npz', test_name='Тест 1: Только канал A')
run_test(alphaA=0, alphaB=1, filename='testB.npz', test_name='Тест 2: Только канал B')
run_test(alphaA=1, alphaB=1, filename='testAB.npz', test_name='Тест 3: Каналы A + B')