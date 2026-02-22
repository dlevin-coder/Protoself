# proto_language_snn.py
from brian2 import *
import numpy as np

def train_proto_language_snn(spike_indices, spike_times, N_input, N_neurons=4, duration_ms=300):
    """
    Минимальная SNN для обучения прото-фонем: предсказание следующего спайка.
    
    spike_indices, spike_times: входные спайки
    N_input: число входных нейронов
    N_neurons: число нейронов основного слоя
    duration_ms: длительность симуляции
    """
    
    start_scope()
    
    # -------------------------
    # Параметры
    # -------------------------
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
    eqs = '''
    dv/dt = -v/tau : 1
    '''
    G = NeuronGroup(N_neurons, model=eqs, threshold='v>1', reset='v=0', method='euler')
    G.v = 'rand()*0.1'
    
    # -------------------------
    # Генеративный слой (предсказание)
    # -------------------------
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
    # LTM - просто аккумулируем среднюю активность
    # -------------------------
    ltm = StateMonitor(G, 'v', record=True)
    
    # -------------------------
    # Мониторы
    # -------------------------
    spike_mon_main = SpikeMonitor(G)
    spike_mon_gen = SpikeMonitor(G_gen)
    
    # -------------------------
    # Запуск
    # -------------------------
    run(duration_ms*ms)
    
    # -------------------------
    # Результаты
    # -------------------------
    results = {
        'main_spikes': spike_mon_main.count[:],
        'gen_spikes': spike_mon_gen.count[:],
        'LTM': np.mean(ltm.v, axis=1),
        'weights_input': syn_in.w[:],
        'weights_pred': syn_pred.w[:]
    }
    
    return results

if __name__ == "__main__":
    # Пример: простой spike train для теста
    spike_idx = [0,1,2,0,1,2,0]
    spike_time = [0,10,20,30,40,50,60]
    res = train_proto_language_snn(spike_idx, spike_time, N_input=3)
    print("Спайки основного слоя:", res['main_spikes'])
    print("Спайки генеративного слоя:", res['gen_spikes'])
    print("LTM:", res['LTM'])