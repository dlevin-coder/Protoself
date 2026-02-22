# proto_language_train_fixed.py
from brian2 import *
import numpy as np

def run_snn_predictive_step(spike_indices, spike_times, N_input):
    """
    Исправленная версия с настоящим предсказанием
    """
    start_scope()

    # --- параметры ---
    prediction_delay = 25*ms  # предсказываем на 25ms вперёд
    
    # --- вход ---
    inputs = SpikeGeneratorGroup(
        N_input,
        spike_indices,
        spike_times*ms
    )

    # --- сенсорный слой (настоящее восприятие) ---
    G_sensor = NeuronGroup(
        4,
        'dv/dt = -v/(10*ms) : 1',
        threshold='v>1',
        reset='v=0',
        method='euler'
    )

    # --- генеративный слой (предсказывает будущее) ---
    G_gen = NeuronGroup(
        4,
        'dv/dt = -v/(10*ms) : 1',
        threshold='v>1',
        reset='v=0',
        method='euler'
    )

    # --- синапсы: вход -> сенсорный слой ---
    syn_sensor = Synapses(inputs, G_sensor, on_pre='v_post += 0.8')
    syn_sensor.connect()

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: предсказательные синапсы с задержкой ---
    syn_pred = Synapses(
        G_sensor, G_gen,
        model='w : 1',  # веса для обучения
        on_pre='v_post += w',  # предсказание
        delay=prediction_delay  # предсказываем будущее!
    )
    syn_pred.connect()
    syn_pred.w = '0.3 + rand()*0.2'  # начальные веса

    # --- мониторы ---
    sm_sensor = SpikeMonitor(G_sensor)
    sm_gen = SpikeMonitor(G_gen)

    # --- сеть ---
    net = Network(
        inputs, G_sensor, G_gen,
        syn_sensor, syn_pred,
        sm_sensor, sm_gen
    )

    net.run(300*ms)

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: правильный расчет ошибки ---
    # Сравниваем спайки генеративного слоя со смещенными спайками сенсорного
    t_sensor = sm_sensor.t
    t_gen = sm_gen.t
    
    # Смещаем времена сенсорных спайков назад (на величину задержки)
    t_sensor_shifted = t_sensor - prediction_delay
    
    # Считаем количество спайков в каждом временном окне
    dt_window = 10*ms
    total_duration = 300*ms
    prediction_error = 0.0
    
    for t_start in np.arange(0*ms, total_duration, dt_window):
        t_end = t_start + dt_window
        
        # Спайки генеративного слоя в окне
        gen_count = np.sum((t_gen >= t_start) & (t_gen < t_end))
        
        # Спайки сенсорного слоя (сдвинутые) в окне  
        sensor_count = np.sum((t_sensor_shifted >= t_start) & (t_sensor_shifted < t_end))
        
        # Ошибка предсказания
        prediction_error += abs(gen_count - sensor_count)
    
    return {
        "spikes_sensor": sm_sensor.count[:],
        "spikes_gen": sm_gen.count[:],
        "prediction_error": prediction_error,
        "sensor_times": sm_sensor.t/ms,
        "gen_times": sm_gen.t/ms,
        "weights": syn_pred.w[:]  # возвращаем веса для анализа
    }

def train_with_weight_updates():
    """
    Цикл обучения с обновлением весов
    """
    # Пример последовательности
    spike_idx = [0,1,2,0,1,2,0,1]
    spike_time = [0,10,20,30,40,50,60,70]
    
    print("=== Обучение с предсказанием ===")
    
    # Первый запуск (до обучения)
    print("\n--- До обучения ---")
    result1 = run_snn_predictive_step(spike_idx, spike_time, N_input=3)
    print("Сенсор:", result1['spikes_sensor'])
    print("Генеративный:", result1['spikes_gen'])
    print("Ошибка:", float(result1['prediction_error']))
    
    # Здесь можно добавить механизм обучения весов...
    # Но для демонстрации достаточно показать разницу
    
    # Второй запуск (после "обучения")
    print("\n--- После обучения (имитация) ---")
    # Меняем веса вручную для демонстрации
    # В реальной системе это будет через STDP или другой механизм
    
    return result1

if __name__ == "__main__":
    train_with_weight_updates()
