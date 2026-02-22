# fixed_proto_language.py
from brian2 import *
import numpy as np

def train_predictive_snn(spike_indices, spike_times, N_input, N_neurons=4, duration_ms=300):
    """
    SNN с настоящим предсказательным обучением
    """
    start_scope()
    
    # ==========================
    # Параметры
    # ==========================
    tau = 10*ms
    prediction_delay = 30*ms  # предсказываем на 30ms вперед
    
    # ==========================
    # Входной слой
    # ==========================
    inputs = SpikeGeneratorGroup(N_input, spike_indices, spike_times*ms)
    
    # ==========================
    # Основной (сенсорный) слой
    # ==========================
    eqs_main = '''
    dv/dt = -v/tau : 1
    last_spike : second
    '''
    G_main = NeuronGroup(N_neurons, model=eqs_main, threshold='v>1', 
                        reset='v=0; last_spike = t', method='euler')
    G_main.v = 'rand()*0.1'
    
    # ==========================
    # Генеративный слой (предсказывает будущее)
    # ==========================
    eqs_gen = '''
    dv/dt = -v/tau : 1
    prediction_target : 1  # целевое значение для обучения
    '''
    G_gen = NeuronGroup(N_neurons, model=eqs_gen, threshold='v>1', reset='v=0', method='euler')
    G_gen.v = 'rand()*0.1'
    
    # ==========================
    # Синапсы: вход -> основной слой
    # ==========================
    syn_in = Synapses(inputs, G_main, on_pre='v_post += 0.8')
    syn_in.connect()
    
    # ==========================
    # Синапсы: основной -> генеративный (с задержкой!)
    # ==========================
    syn_pred = Synapses(G_main, G_gen, 
                       model='w : 1',
                       on_pre='''
                       v_post += w;
                       prediction_target_post = v_pre  # запоминаем текущее состояние
                       ''')
    syn_pred.connect()
    syn_pred.delay = prediction_delay  # КЛЮЧЕВОЙ МОМЕНТ!
    syn_pred.w = '0.5 + rand()*0.3'
    
    # ==========================
    # STDP для обучения предсказания
    # ==========================
    stdp_pred = '''
    w : 1
    dApre/dt = -Apre / (20*ms) : 1 (event-driven)
    dApost/dt = -Apost / (20*ms) : 1 (event-driven)
    '''
    
    on_pre_stdp = '''
    v_post += w
    Apre += 0.05
    w = clip(w + Apost, 0, 1)
    '''
    
    on_post_stdp = '''
    Apost += -0.05
    w = clip(w + Apre, 0, 1)
    '''
    
    # Подключаем STDP к предсказательным синапсам
    syn_learning = Synapses(G_main, G_gen, model=stdp_pred, 
                           on_pre=on_pre_stdp, on_post=on_post_stdp)
    syn_learning.connect()
    syn_learning.w = '0.3 + rand()*0.2'
    syn_learning.delay = prediction_delay
    
    # ==========================
    # Мониторы
    # ==========================
    spike_mon_main = SpikeMonitor(G_main)
    spike_mon_gen = SpikeMonitor(G_gen)
    state_mon_gen = StateMonitor(G_gen, ['v', 'prediction_target'], record=True)
    
    # ==========================
    # Запуск
    # ==========================
    run(duration_ms*ms)
    
    # ==========================
    # Расчет(prediction error)
    # ==========================
    # Сравниваем активность генеративного слоя с тем, 
    # что было в основном слое prediction_delay времени назад
    prediction_errors = []
    for i in range(len(state_mon_gen.t)):
        time_point = state_mon_gen.t[i]
        target_time = time_point - prediction_delay
        
        # Находим соответствующие значения
        if target_time >= 0*ms:
            # Здесь должна быть логика сравнения...
            # Для простоты считаем среднюю ошибку
            pred_v = state_mon_gen.v[:, i]
            target_v = state_mon_gen.prediction_target[:, i] 
            error = np.mean(np.abs(pred_v - target_v)) if len(target_v) > 0 else 0
            prediction_errors.append(error)
    
    avg_prediction_error = np.mean(prediction_errors) if prediction_errors else 0
    
    return {
        'main_spikes': spike_mon_main.count[:],
        'gen_spikes': spike_mon_gen.count[:],
        'prediction_error': avg_prediction_error,
        'weights': syn_learning.w[:],
        'timestamps': state_mon_gen.t/ms
    }

# ==========================
# Цикл обучения между эпохами
# ==========================
def train_across_epochs(N_input=4, epochs=5):
    error_history = []
    
    # Генерируем одну и ту же последовательность для обучения
    base_spike_indices = [0, 1, 2, 3, 0, 1, 2, 3]
    base_spike_times = [0, 20, 40, 60, 100, 120, 140, 160]
    
    for epoch in range(epochs):
        print(f"\n--- Эпоха {epoch+1} ---")
        
        result = train_predictive_snn(
            base_spike_indices, 
            base_spike_times, 
            N_input, 
            duration_ms=200
        )
        
        print(f"Ошибка предсказания: {result['prediction_error']:.4f}")
        print(f"Спайки основного слоя: {result['main_spikes']}")
        print(f"Спайки генеративного: {result['gen_spikes']}")
        
        error_history.append(result['prediction_error'])
        
        # Здесь можно добавить адаптацию весов между эпохами
    
    return error_history

if __name__ == "__main__":
    errors = train_across_epochs()
    print(f"\nИстория ошибок: {errors}")
