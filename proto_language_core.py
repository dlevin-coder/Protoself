# proto_language_train.py
from brian2 import *
import numpy as np

def train_proto_sequence(spike_indices, spike_times, N_input, N_neurons=4, duration_ms=300):
    """
    Минимальная SNN для обучения последовательностям аудио-спайков.
    Генеративный слой предсказывает следующий спайк, считаем prediction error.
    """
    start_scope()
    
    tau = 10*ms
    tau_pre = 20*ms
    tau_post = 20*ms
    Apre_val = 0.1
    Apost_val = -0.1
    
    # -------------------------
    # Входной слой
    # -------------------------
    inputs = SpikeGeneratorGroup(N_input, spike_indices, spike_times*ms)
    
    # -------------------------
    # Основной слой
    # -------------------------
    eqs = 'dv/dt = -v/tau : 1'
    G = NeuronGroup(N_neurons, model=eqs, threshold='v>1', reset='v=0', method='euler')
    G.v = 'rand()*0.1'
    
    # -------------------------
    # Генеративный слой
    # -------------------------
    gen_eqs = 'dv/dt = -v/tau : 1'
    G_gen = NeuronGroup(N_neurons, model=eqs, threshold='v>1', reset='v=0', method='euler')
    G_gen.v = 'rand()*0.1'
    
    # -------------------------
    # Синапсы вход→основной слой с STDP
    # -------------------------
    stdp_eqs = '''
    w : 1
    dApre/dt = -Apre/tau_pre : 1 (event-driven)
    dApost/dt = -Apost/tau_post : 1 (event-driven)
    '''
    on_pre = f'''
    v_post += w
    Apre += {Apre_val}
    w = clip(w + Apost, 0, 3)
    '''
    on_post = f'''
    Apost += {Apost_val}
    w = clip(w + Apre, 0, 3)
    '''
    syn_in = Synapses(inputs, G, model=stdp_eqs, on_pre=on_pre, on_post=on_post)
    syn_in.connect()
    syn_in.w = '0.5 + rand()*0.5'
    
    # -------------------------
    # Синапсы основной→генеративный (предсказание)
    # -------------------------
    syn_pred = Synapses(G, G_gen, model='w:1', on_pre='v_post += w')
    syn_pred.connect()
    syn_pred.w = '0.3 + rand()*0.3'
    
    # -------------------------
    # Мониторы
    # -------------------------
    spike_mon_main = SpikeMonitor(G)
    spike_mon_gen = SpikeMonitor(G_gen)
    
    # -------------------------
    # StateMonitor для LTM
    # -------------------------
    ltm = StateMonitor(G, 'v', record=True)
    
    # -------------------------
    # Функция для расчета prediction error
    # -------------------------
    def prediction_error():
        t_main = spike_mon_main.t/ms
        t_gen = spike_mon_gen.t/ms
        error = 0.0
        # сравниваем число спайков в коротких интервалах
        dt_window = 10  # ms
        for t0 in np.arange(0, duration_ms, dt_window):
            count_main = np.sum((t_main >= t0) & (t_main < t0+dt_window))
            count_gen = np.sum((t_gen >= t0) & (t_gen < t0+dt_window))
            error += abs(count_main - count_gen)
        return error
    
    # -------------------------
    # Запуск
    # -------------------------
    run(duration_ms*ms)
    
    results = {
        'main_spikes': spike_mon_main.count[:],
        'gen_spikes': spike_mon_gen.count[:],
        'LTM': np.mean(ltm.v, axis=1),
        'weights_input': syn_in.w[:],
        'weights_pred': syn_pred.w[:],
        'prediction_error': prediction_error()
    }
    
    return results

def make_spike_input(spike_indices, spike_times, N_input):
    return SpikeGeneratorGroup(
        N_input,
        spike_indices,
        spike_times * ms
    )

def run_snn_step(spike_indices, spike_times, N_input):
    start_scope()

    # --- вход ---
    inputs = SpikeGeneratorGroup(
        N_input,
        spike_indices,
        spike_times*ms
    )

    # --- нейроны ---
    G = NeuronGroup(
        4,
        'dv/dt = -v/(10*ms) : 1',
        threshold='v>1',
        reset='v=0',
        method='euler'
    )

    G_gen = NeuronGroup(
        4,
        'dv/dt = -v/(10*ms) : 1',
        threshold='v>1',
        reset='v=0',
        method='euler'
    )

    # --- синапсы ---
    syn = Synapses(inputs, G, on_pre='v_post += 0.5')
    syn.connect()

    syn_pred = Synapses(
    G, G_gen,
    on_pre='v_post += 0.3',
    delay=20*ms
    )
    syn_pred.connect()

    # --- мониторы ---
    sm_main = SpikeMonitor(G)
    sm_gen = SpikeMonitor(G_gen)

    net = Network(
        inputs, G, G_gen,
        syn, syn_pred,
        sm_main, sm_gen
    )

    net.run(300*ms)

    return {
        "spikes_main": sm_main.count[:],
        "spikes_gen": sm_gen.count[:],
        "prediction_error": abs(sm_main.num_spikes - sm_gen.num_spikes)
    }

if __name__ == "__main__":
    # Пример последовательности: 3 входных нейрона
    spike_idx = [0,1,2,0,1,2,0,1]
    spike_time = [0,10,20,30,40,50,60,70]
    res = train_proto_sequence(spike_idx, spike_time, N_input=3)
    
    print("Спайки основного слоя:", res['main_spikes'])
    print("Спайки генеративного слоя:", res['gen_spikes'])
    print("LTM:", res['LTM'])
    print("Prediction error:", res['prediction_error'])