# proto_language_train_debug.py
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def run_snn_predictive_step(spike_indices, spike_times, N_input):
    """
    Отладочная версия с усиленными сигналами
    """
    start_scope()
    
    print(f"Input spikes: indices={spike_indices}, times={spike_times}")
    
    prediction_delay = 25*ms  
    
    # Увеличиваем веса для надежной активации
    inputs = SpikeGeneratorGroup(N_input, spike_indices, spike_times*ms)
    
    G_sensor = NeuronGroup(4, 'dv/dt = -v/(10*ms) : 1', 
                          threshold='v>1', reset='v=0', method='euler')
    G_sensor.v = 0  # начальное состояние
    
    G_gen = NeuronGroup(4, 'dv/dt = -v/(10*ms) : 1', 
                       threshold='v>1', reset='v=0', method='euler')
    G_gen.v = 0
    
    # УСИЛИВАЕМ входные веса
    syn_sensor = Synapses(inputs, G_sensor, on_pre='v_post += 2.0')  # было 0.8
    syn_sensor.connect()  # подключаем все
    
    # Проверяем, какие индексы доступны
    print(f"N_input={N_input}, создаём соединения...")
    
    # УСИЛИВАЕМ предсказательные веса  
    syn_pred = Synapses(G_sensor, G_gen, on_pre='v_post += 1.5', delay=prediction_delay)  # было 0.3
    syn_pred.connect()  # подключаем все
    
    sm_sensor = SpikeMonitor(G_sensor)
    sm_gen = SpikeMonitor(G_gen)
    sm_input = SpikeMonitor(inputs)  # мониторим вход тоже
    
    net = Network(inputs, G_sensor, G_gen, syn_sensor, syn_pred, 
                  sm_sensor, sm_gen, sm_input)
    net.run(300*ms)
    
    print(f"Входные спайки: {sm_input.count[:]}")
    print(f"Сенсорные спайки: {sm_sensor.count[:]}")
    print(f"Генеративные спайки: {sm_gen.count[:]}")
    
    # Если есть активность, считаем ошибку
    if sm_sensor.num_spikes > 0 or sm_gen.num_spikes > 0:
        # Простая ошибка для начала
        prediction_error = abs(sm_sensor.num_spikes - sm_gen.num_spikes)
        print(f"Ошибка (простая): {prediction_error}")
    else:
        prediction_error = 0
        print("Нет активности - ошибка = 0")
    
    return {
        "input_spikes": sm_input.count[:],
        "spikes_sensor": sm_sensor.count[:],
        "spikes_gen": sm_gen.count[:],
        "prediction_error": float(prediction_error),
        "input_times": sm_input.t/ms,
        "sensor_times": sm_sensor.t/ms if len(sm_sensor.t) > 0 else [],
        "gen_times": sm_gen.t/ms if len(sm_gen.t) > 0 else []
    }

# ==========================
# Тест с усиленными сигналами
# ==========================
def generate_test_sequence():
    """Усиленная последовательность"""
    # Убедимся, что индексы в пределах [0, 3]  
    spike_indices = [0, 1, 2, 3, 0, 1, 2, 3]  # 4 нейрона
    spike_times = [10, 30, 50, 70, 100, 120, 140, 160]
    return spike_indices, spike_times

# ==========================
# Тестовый запуск
# ==========================
print("=== Отладочный запуск ===")
spike_indices, spike_times = generate_test_sequence()
print(f"Индексы: {spike_indices}")
print(f"Времена: {spike_times}")

result = run_snn_predictive_step(spike_indices, spike_times, N_input=4)

print(f"\nРезультаты:")
print(f"Вход: {result['input_spikes']}")
print(f"Сенсор: {result['spikes_sensor']}")  
print(f"Генеративный: {result['spikes_gen']}")
print(f"Ошибка: {result['prediction_error']}")
