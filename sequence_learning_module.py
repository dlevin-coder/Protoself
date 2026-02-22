# sequence_learning_module.py
from brian2 import *
import numpy as np

def create_network_components(N_input, N_hidden=4, N_output=4):
    """
    Создание компонентов сети (без входного слоя)
    """
    # Параметры
    tau = 10*ms
    
    # Уравнения для нейронов (с конкретным значением tau)
    eqs_neuron = '''
    dv/dt = -v/(10*ms) : 1
    '''
    
    # Скрытый слой (основная обработка)
    hidden = NeuronGroup(N_hidden, model=eqs_neuron, threshold='v>1', 
                        reset='v=0', method='euler')
    hidden.v = 'rand()*0.1'
    
    # Выходной/предсказывающий слой
    output = NeuronGroup(N_output, model=eqs_neuron, threshold='v>1', 
                        reset='v=0', method='euler')
    output.v = 'rand()*0.1'
    
    # Синапсы вход→скрытый слой с STDP
    stdp_eqs = '''
    w : 1
    dApre/dt = -Apre/(20*ms) : 1 (event-driven)
    dApost/dt = -Apost/(20*ms) : 1 (event-driven)
    '''
    
    on_pre_stdp = '''
    v_post += w
    Apre += 0.1
    w = clip(w + Apost, 0, 3)
    '''
    
    on_post_stdp = '''
    Apost += -0.12
    w = clip(w + Apre, 0, 3)
    '''
    
    # Рекуррентные связи в скрытом слое
    syn_hidden_recurrent = Synapses(hidden, hidden, model='w:1', on_pre='v_post += w')
    syn_hidden_recurrent.connect(condition='i!=j')
    syn_hidden_recurrent.w = '0.2 + rand()*0.2'
    
    # Синапсы скрытый→выходной слой
    syn_hidden_output = Synapses(hidden, output, model='w:1', on_pre='v_post += w')
    syn_hidden_output.connect()
    syn_hidden_output.w = '0.3 + rand()*0.3'
    
    # Сохраняем все компоненты
    network_components = {
        'hidden': hidden,
        'output': output,
        'syn_hidden_recurrent': syn_hidden_recurrent,
        'syn_hidden_output': syn_hidden_output
    }
    
    return network_components

def run_single_sequence(network_components, N_input, spike_idx, spike_time, duration=None):
    """
    Запуск одной последовательности с созданием нового входного слоя каждый раз
    """
    # Создаем новую сеть для каждого запуска
    net = Network()
    
    # Создаем новый входной слой для каждой последовательности
    if len(spike_idx) > 0 and len(spike_time) > 0:
        inputs = SpikeGeneratorGroup(N_input, spike_idx, np.array(spike_time)*ms)
    else:
        inputs = SpikeGeneratorGroup(N_input, [], []*ms)
    
    # Добавляем все компоненты из network_components
    for key, component in network_components.items():
        net.add(component)
    
    # Добавляем входной слой
    net.add(inputs)
    
    # Создаем новые мониторы для этой сессии
    spike_mon_hidden = SpikeMonitor(network_components['hidden'])
    spike_mon_output = SpikeMonitor(network_components['output'])
    net.add(spike_mon_hidden, spike_mon_output)
    
    # Создаем синапсы вход→скрытый слой (новые для каждой сессии)
    syn_input_hidden = Synapses(inputs, network_components['hidden'], 
                               model='''
                               w : 1
                               dApre/dt = -Apre/(20*ms) : 1 (event-driven)
                               dApost/dt = -Apost/(20*ms) : 1 (event-driven)
                               ''', 
                               on_pre='''
                               v_post += w
                               Apre += 0.1
                               w = clip(w + Apost, 0, 3)
                               ''',
                               on_post='''
                               Apost += -0.12
                               w = clip(w + Apre, 0, 3)
                               ''')
    syn_input_hidden.connect()
    syn_input_hidden.w = '0.5 + rand()*0.5'
    net.add(syn_input_hidden)
    
    # Определяем длительность
    if duration is None:
        duration = (max(spike_time) + 100) if len(spike_time) > 0 else 200
    
    # Запускаем симуляцию
    net.run(duration*ms)
    
    return {
        'spike_mon_hidden': spike_mon_hidden,
        'spike_mon_output': spike_mon_output,
        'duration': duration,
        'syn_input_hidden': syn_input_hidden
    }

def analyze_prediction(spike_mon_hidden, spike_mon_output, target_times):
    """
    Анализ предсказательной способности сети
    """
    # Получаем времена спайков
    hidden_spikes = np.array(spike_mon_hidden.t/ms)
    output_spikes = np.array(spike_mon_output.t/ms)
    
    if len(target_times) == 0:
        return {'prediction_accuracy': 0.0, 'hidden_spike_count': 0, 'output_spike_count': 0}
    
    # Если нет спайков вообще - нулевая точность
    if len(output_spikes) == 0:
        return {'prediction_accuracy': 0.0, 'hidden_spike_count': len(hidden_spikes), 'output_spike_count': 0}
    
    # Определяем окно предсказания (после последнего входного спайка)
    target_end = max(target_times) if target_times else 0
    prediction_window_start = target_end
    prediction_window_end = target_end + 50  # 50ms после последнего входа
    
    # Считаем спайки в окне предсказания
    pred_spikes_in_window = output_spikes[
        (output_spikes >= prediction_window_start) & 
        (output_spikes <= prediction_window_end)
    ]
    
    # Базовая метрика: наличие спайков после входа
    prediction_score = min(len(pred_spikes_in_window) / max(1, len(target_times)), 1.0)
    
    return {
        'prediction_accuracy': prediction_score,
        'hidden_spike_count': len(hidden_spikes),
        'output_spike_count': len(output_spikes),
        'predicted_spikes': len(pred_spikes_in_window)
    }

def generate_spike_sequences(base_pattern, num_variants=3, noise_level=0.1):
    """
    Генерация вариантов последовательности с шумом
    """
    sequences = []
    
    for i in range(num_variants):
        # Добавляем небольшой шум к времени
        noisy_times = [max(0, t + np.random.normal(0, noise_level*5)) for t in base_pattern[1]]
        sequences.append((base_pattern[0][:], noisy_times))  # копируем индексы тоже
    
    return sequences

# Пример использования
if __name__ == "__main__":
    # Создаем базовые компоненты сети
    try:
        N_input = 3
        net_components = create_network_components(N_input, N_hidden=4, N_output=4)
        
        # Базовая последовательность (индексы нейронов, времена)
        base_sequence = ([0, 1, 2, 0, 1], [0, 10, 20, 30, 40])
        
        # Генерируем варианты с шумом
        sequences = generate_spike_sequences(base_sequence, num_variants=3)
        
        # Обучаем
        print("Начинаем обучение последовательностям...")
        results_history = []
        
        for seq_idx, (spike_idx, spike_time) in enumerate(sequences):
            print(f"Обучение последовательности {seq_idx + 1}")
            print(f"  Входные спайки: индексы={spike_idx}, времена={[round(t,1) for t in spike_time]}")
            
            # Запускаем последовательность
            result_data = run_single_sequence(net_components, N_input, spike_idx, spike_time)
            
            # Анализируем результаты
            result = analyze_prediction(
                result_data['spike_mon_hidden'],
                result_data['spike_mon_output'],
                spike_time
            )
            results_history.append(result)
            
            print(f"  Предсказательная точность: {result['prediction_accuracy']:.2f}")
            print(f"  Спайков в скрытом слое: {result['hidden_spike_count']}")
            print(f"  Спайков в выходном слое: {result['output_spike_count']}")
            print(f"  Предсказанных спайков: {result['predicted_spikes']}")
            
        # Выводим финальные результаты
        print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
        if results_history:
            avg_accuracy = np.mean([r['prediction_accuracy'] for r in results_history])
            print(f"Средняя предсказательная точность: {avg_accuracy:.3f}")
            
            for i, result in enumerate(results_history):
                print(f"Последовательность {i+1}: точность={result['prediction_accuracy']:.3f}, "
                      f"скрытые={result['hidden_spike_count']}, выходные={result['output_spike_count']}")
        else:
            print("Нет результатов для анализа")
            
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
