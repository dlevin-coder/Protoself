from brian2 import *
import numpy as np
from sklearn.decomposition import PCA

print("Self-Recursion Version — Attractor Prediction Test")

# --------------------------
# Общие параметры
# --------------------------

N = 50
tau_m = 50*ms
tau_w = 500*ms

EL = -70*mV
VT = -60*mV
VT_base = -60*mV
Vr = -65*mV

R = 150*Mohm
tau_VT = 50*ms

a_adapt = 0.5*pA/mV
b_adapt = 0.5*pA

# --------------------------
# Endogenous OU (body noise)
# --------------------------

I_endo_mean = 50*pA
I_endo_std = 40*pA
tau_endo = 300*ms

sigma_mod = 30*pA
tau_mod = 300*ms

# --------------------------
# Observer layer Σ_fast
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

sigma_fast = 20*pA
tau_fast = 50*ms
tau_fast_avg = 300*ms

# --------------------------
# Attractor prediction
# --------------------------

tau_pred = 500*ms
prediction_gain = 0.2*pA/mV

# --------------------------
# Temporal patterns
# --------------------------

pattern_A_amp = 150*pA
pattern_B_amp = 120*pA

pattern_A_period = 400*ms
pattern_B_period = 250*ms

pattern_A_width = 100*ms
pattern_B_width = 50*ms

sim_time = 5*second


def run_test(alphaA, alphaB, use_body=True,
             use_prediction=True,
             filename="self_recursion_test.npz",
             test_name="Self recursion", *, prediction_gain_local=0.2*pA/mV):

    print(f"\n=== {test_name} ===")

    start_scope()

    # --------------------------
    # Σ_fast (observer)
    # --------------------------

    eqs_fast = '''
    dv/dt = (EL_fast - v + R_fast*(I_fast_A + I_fast_B +
             I_pattern_A + I_pattern_B - w)) / tau_m_fast : volt

    dw/dt = (a_fast*(v - EL_fast) - w)/tau_w_fast : amp

    dI_fast_A/dt = -I_fast_A/tau_fast +
        (sigma_fast*sqrt(2/tau_fast))*xi_fA : amp

    dI_fast_B/dt = -I_fast_B/tau_fast +
        (sigma_fast*sqrt(2/tau_fast))*xi_fB : amp

    ds_fast/dt = -s_fast / tau_fast_avg : 1

    I_pattern_A : amp
    I_pattern_B : amp
    '''

    Sigma_fast = NeuronGroup(
        N_fast,
        eqs_fast,
        threshold='v>VT_fast',
        reset='v=Vr_fast; w+=b_fast; s_fast+=1',
        method='euler'
    )

    Sigma_fast.v = EL_fast
    Sigma_fast.w = 0*pA
    Sigma_fast.s_fast = 0

    # --------------------------
    # Σ_slow (embodied layer)
    # --------------------------

    eqs_slow = '''

    dv/dt = (EL - v + R*(I_endo*(use_body)
      + alphaA*I_mod_A
      + alphaB*I_mod_B
      + I_pattern_A
      + I_pattern_B
      + I_fast_mod
      - w))/tau_m : volt

    dw/dt = (a_adapt*(v - EL) - w)/tau_w : amp

    dVT_dyn/dt = (VT_base - VT_dyn)/tau_VT : volt

    dI_endo/dt = -(I_endo - I_endo_mean)/tau_endo
       + (I_endo_std*sqrt(2/tau_endo))*xi_endo : amp

    dI_mod_A/dt = -I_mod_A/tau_mod
       + (sigma_mod*sqrt(2/tau_mod))*xi_A : amp

    dI_mod_B/dt = -I_mod_B/tau_mod
       + (sigma_mod*sqrt(2/tau_mod))*xi_B : amp

    I_fast_mod : amp
    I_pattern_A : amp
    I_pattern_B : amp
    use_body : 1
    '''

    Sigma_slow = NeuronGroup(
        N,
        eqs_slow,
        threshold='v>VT_dyn',
        reset='v=Vr; w+=b_adapt',
        method='euler'
    )

    Sigma_slow.v = EL
    Sigma_slow.w = 0*pA

    Sigma_slow.I_endo = I_endo_mean
    Sigma_slow.I_fast_mod = 0*pA

    Sigma_slow.VT_dyn = VT_base

    Sigma_slow.use_body = 1 if use_body else 0

    # --------------------------
    # recurrent slow network
    # --------------------------

    S_rec = Synapses(Sigma_slow, Sigma_slow,
                     on_pre='v_post += 0.1*mV')
    S_rec.connect(p=0.1)

    # --------------------------
    # Observer modulation
    # --------------------------

    Delta_w = 0.05*pA

    @network_operation(dt=10*ms)
    def fast_to_slow_modulation():

        mean_fast = np.mean(Sigma_fast.s_fast)

        Sigma_slow.I_fast_mod = Delta_w * mean_fast

    # --------------------------
    # Attractor prediction
    # --------------------------

    pred_state = 0*mV

    @network_operation(dt=10*ms)
    def attractor_prediction():

        nonlocal pred_state

        if not use_prediction:
            return

        current_state = np.mean(Sigma_slow.v)

        pred_state += (current_state - pred_state) * (10*ms/tau_pred)

        error = current_state - pred_state

        Sigma_slow.I_fast_mod += -prediction_gain_local * error

    # --------------------------
    # Temporal patterns
    # --------------------------

    @network_operation(dt=1*ms)
    def temporal_patterns():

        t_now = defaultclock.t

        phaseA = t_now % pattern_A_period
        phaseB = t_now % pattern_B_period

        valA = pattern_A_amp if phaseA < pattern_A_width else 0*pA
        valB = pattern_B_amp if phaseB < pattern_B_width else 0*pA

        Sigma_slow.I_pattern_A = valA
        Sigma_slow.I_pattern_B = valB

        Sigma_fast.I_pattern_A = valA
        Sigma_fast.I_pattern_B = valB

    # --------------------------
    # Monitors
    # --------------------------

    M_spike = SpikeMonitor(Sigma_slow)
    M_v = StateMonitor(Sigma_slow, 'v', record=True)
    M_rate = PopulationRateMonitor(Sigma_slow)

    # --------------------------
    # Run
    # --------------------------

    net = Network(
        Sigma_slow,
        Sigma_fast,
        S_rec,
        M_spike,
        M_v,
        M_rate,
        fast_to_slow_modulation,
        attractor_prediction,
        temporal_patterns
    )

    net.run(sim_time)

    # --------------------------
    # Metrics
    # --------------------------

    rate = M_rate.smooth_rate(window='gaussian', width=5*ms)/Hz

    synchrony = np.var(rate)/np.mean(rate)

    v_mean = np.mean(M_v.v, axis=0)

    v_center = v_mean - np.mean(v_mean)

    autocorr = np.correlate(v_center, v_center, mode='full')
    autocorr = autocorr[autocorr.size//2:]
    autocorr /= np.max(autocorr)

    pca = PCA(n_components=3)

    V = M_v.v.T

    pca.fit(V)

    explained = pca.explained_variance_ratio_

    # --------------------------
    # Output
    # --------------------------

    print("Spikes:", len(M_spike.t))
    print("Synchrony:", synchrony)
    print("PCA:", explained)
    print("Prediction gain:", prediction_gain_local)

    np.savez(
        filename,
        spikes=np.array(M_spike.t),
        voltages=np.array(M_v.v),
        rate=np.array(rate),
        synchrony=synchrony,
        autocorr=autocorr,
        pca=explained
    )

    print("Saved:", filename)


# --------------------------
# Experiments
# --------------------------

run_test(0,0,True,False,"baseline.npz","Baseline")

run_test(1,1,True,False,"observer_only.npz","Observer only")

run_test(1,1,True,True,"self_recursion.npz","Self recursion")