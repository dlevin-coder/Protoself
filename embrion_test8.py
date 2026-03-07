from brian2 import *
import numpy as np

print("Эмбриональный режим Σ_slow — три теста по каналам A/B (OU-шума) — Шаг 5")

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
from sklearn.decomposition import PCA

def run_test(alphaA, alphaB, use_body=True, filename='test.npz', test_name='Test'):
    print(f"\n=== {test_name} ===")
    start_scope()

    # --------------------------
    # Σ_fast (наблюдатель)
    # --------------------------
    eqs_fast = '''
    dv/dt = (EL_fast - v + R_fast*(I_fast_A + I_fast_B + I_pattern_A + I_pattern_B - w)) / tau_m_fast : volt
    dw/dt = (a_fast*(v - EL_fast) - w)/tau_w_fast : amp
    dI_fast_A/dt = -I_fast_A/tau_fast + (sigma_fast*sqrt(2/tau_fast))*xi_fA : amp
    dI_fast_B/dt = -I_fast_B/tau_fast + (sigma_fast*sqrt(2/tau_fast))*xi_fB : amp
    ds_fast/dt = -s_fast / tau_fast_avg : 1
    I_pattern_A : amp
    I_pattern_B : amp
    '''

    Sigma_fast = NeuronGroup(
        N_fast, eqs_fast, threshold='v>VT_fast',
        reset='v=Vr_fast; w+=b_fast; s_fast+=1.0', method='euler'
    )
    Sigma_fast.v = EL_fast
    Sigma_fast.w = 0*pA
    Sigma_fast.I_fast_A = 0*pA
    Sigma_fast.I_fast_B = 0*pA
    Sigma_fast.s_fast = 0.0
    Sigma_fast.I_pattern_A = 0*pA
    Sigma_fast.I_pattern_B = 0*pA

    # --------------------------
    # Σ_slow (эмбриональный слой)
    # --------------------------
    eqs_slow = '''
    dv/dt = (EL - v + R*(I_endo*(use_body) + alphaA*I_mod_A + alphaB*I_mod_B + I_pattern_A + I_pattern_B - w))/tau_m : volt
    dw/dt = ((a_adapt*(v - EL) - w) + I_fast_mod)/tau_w : amp
    dVT_dyn/dt = (VT_base - VT_dyn)/tau_VT : volt
    dI_endo/dt = -(I_endo - I_endo_mean)/tau_endo + (I_endo_std*sqrt(2/tau_endo))*xi_endo : amp
    dI_mod_A/dt = -I_mod_A/tau_mod + (sigma_mod*sqrt(2/tau_mod))*xi_A : amp
    dI_mod_B/dt = -I_mod_B/tau_mod + (sigma_mod*sqrt(2/tau_mod))*xi_B : amp
    I_fast_mod : amp
    I_pattern_A : amp
    I_pattern_B : amp
    use_body : 1  # флаг включения телесного фона
    '''

    Sigma_slow = NeuronGroup(
        N, eqs_slow, threshold='v>VT_dyn',
        reset='v=Vr; w+=b_adapt', method='euler'
    )
    Sigma_slow.v = EL
    Sigma_slow.w = 0*pA
    Sigma_slow.I_endo = I_endo_mean
    Sigma_slow.I_mod_A = 0*pA
    Sigma_slow.I_mod_B = 0*pA
    Sigma_slow.VT_dyn = VT_base
    Sigma_slow.I_fast_mod = 0*pA
    Sigma_slow.I_pattern_A = 0*pA
    Sigma_slow.I_pattern_B = 0*pA
    Sigma_slow.use_body = 1 if use_body else 0

    # --------------------------
    # Внутренняя слабая связь Σ_slow
    # --------------------------
    S_rec = Synapses(Sigma_slow, Sigma_slow, on_pre='v_post+=0.1*mV')
    S_rec.connect(p=0.1)

    # --------------------------
    # Модуляция Σ_fast → Σ_slow
    # --------------------------
    Delta_w = 0.05*pA
    @network_operation(dt=10*ms)
    def fast_to_slow_modulation():
        mean_fast = np.mean(Sigma_fast.s_fast)
        Sigma_slow.I_fast_mod = Delta_w * mean_fast

    # --------------------------
    # Паттерны
    # --------------------------
    @network_operation(dt=1*ms)
    def temporal_patterns():
        t_now = defaultclock.t
        phaseA = t_now % pattern_A_period
        phaseB = t_now % pattern_B_period

        Sigma_slow.I_pattern_A = pattern_A_amp if phaseA < pattern_A_width else 0*pA
        Sigma_slow.I_pattern_B = pattern_B_amp if phaseB < pattern_B_width else 0*pA
        Sigma_fast.I_pattern_A = Sigma_slow.I_pattern_A
        Sigma_fast.I_pattern_B = Sigma_slow.I_pattern_B

    # --------------------------
    # Мониторы
    # --------------------------
    M_spike = SpikeMonitor(Sigma_slow)
    M_v = StateMonitor(Sigma_slow, 'v', record=True)
    M_w = StateMonitor(Sigma_slow, 'w', record=range(min(3,N)))
    M_fast_spike = SpikeMonitor(Sigma_fast)
    M_rate = PopulationRateMonitor(Sigma_slow)

    # --------------------------
    # Запуск сети
    # --------------------------
    net = Network(Sigma_slow, Sigma_fast, S_rec,
                  M_spike, M_v, M_w, M_fast_spike, M_rate,
                  fast_to_slow_modulation, temporal_patterns)
    net.run(sim_time)

    # --------------------------
    # Метрики self
    # --------------------------
    # 1. Population synchrony
    rate = M_rate.smooth_rate(window='gaussian', width=5*ms)/Hz
    sync_index = np.var(rate)/np.mean(rate)

    # 2. Автокорреляция Σ_slow
    v_mean = np.mean(M_v.v, axis=0)
    v_mean_centered = v_mean - np.mean(v_mean)
    autocorr = np.correlate(v_mean_centered, v_mean_centered, mode='full')
    autocorr = autocorr[autocorr.size//2:] / np.max(autocorr)

    # 3. PCA мембранных потенциалов
    pca = PCA(n_components=3)
    V_flat = M_v.v.T  # время x нейроны
    pca.fit(V_flat)
    explained_variance = pca.explained_variance_ratio_

    # --------------------------
    # Вывод
    # --------------------------
    print(f"Общее число спайков: {len(M_spike.t)}")
    print(f"Средняя частота: {len(M_spike.t)/(N*sim_time/second):.2f} Hz/нейрон")
    print(f"Population synchrony index: {sync_index:.3f}")
    print(f"PCA explained variance (3 components): {explained_variance}")

    # --------------------------
    # Сохранение
    # --------------------------
    np.savez(filename,
             spike_times=np.array(M_spike.t),
             spike_indices=np.array(M_spike.i),
             voltages=np.array(M_v.v),
             voltage_times=np.array(M_v.t),
             adaptation=np.array(M_w.w),
             adaptation_times=np.array(M_w.t),
             population_rate=np.array(rate),
             synchrony=sync_index,
             autocorr=autocorr,
             pca_explained=explained_variance
             )
    print(f"Данные сохранены в '{filename}'")
# Тесты
run_test(alphaA=0, alphaB=0, use_body=True, filename='test_body_only.npz', test_name='Только телесный фон')
run_test(alphaA=0, alphaB=0, use_body=False, filename='test_no_body.npz', test_name='Без телесного фона')
run_test(alphaA=1, alphaB=1, use_body=True, filename='test_full.npz', test_name='Телесный фон + каналы A/B')    