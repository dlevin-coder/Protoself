# improved_minimal_test.py - улучшенная минимальная система
from brian2 import *
import numpy as np

def test_improved_system():
    """Тест улучшенной минимальной SNN системы"""
    print("🧪 Тест улучшенной SNN системы")
    
    # Параметры
    defaultclock.dt = 1*ms  # Более точное временнóе разрешение
    
    # Создаем искусственные спайки для входа
    N_input = 8
    spike_times = []
    spike_indices = []
    
    # Более плотная генерация спайков
    for i in range(N_input):
        for t in range(5, 200, 15):  # каждые 15ms от 5ms до 200ms
            spike_times.append(t * ms)
            spike_indices.append(i)
    
    print(f"Генерируем {len(spike_indices)} входных спайков")
    input_layer = SpikeGeneratorGroup(N_input, spike_indices, spike_times)
    
    # Скрытый слой (наблюдатель) - более чувствительный
    hidden_eqs = '''
    dv/dt = (I_ext - v) / (15*ms) : 1
    dI_ext/dt = -I_ext / (5*ms) : 1
    '''
    hidden_layer = NeuronGroup(16, hidden_eqs, threshold='v>0.5', reset='v=0', method='euler')
    hidden_layer.v = 'rand() * 0.2'  # Начальное возбуждение
    
    # Выходной слой - тоже чувствительный
    output_eqs = '''
    dv/dt = (I_ext - v) / (10*ms) : 1
    dI_ext/dt = -I_ext / (3*ms) : 1
    '''
    output_layer = NeuronGroup(4, output_eqs, threshold='v>0.3', reset='v=0', method='euler')
    output_layer.v = 'rand() * 0.1'
    
    # Синапсы от входа к скрытым (сильные связи)
    input_syn = Synapses(input_layer, hidden_layer, 'w : 1', on_pre='I_ext_post += w')
    input_syn.connect(p=0.7)  # Больше связей
    input_syn.w = '1.2 + 0.3*rand()'  # Сильные веса
    
    # Рекуррентные связи в скрытом слое
    rec_syn = Synapses(hidden_layer, hidden_layer, 'w : 1', on_pre='I_ext_post += w')
    rec_syn.connect(condition='i!=j', p=0.4)
    rec_syn.w = '0.5*rand()'
    
    # От скрытого к выходному (очень сильные связи)
    output_syn = Synapses(hidden_layer, output_layer, 'w : 1', on_pre='I_ext_post += w')
    output_syn.connect(p=0.6)
    output_syn.w = '1.5 + 0.5*rand()'  # Очень сильные веса
    
    # Мониторы
    input_monitor = SpikeMonitor(input_layer)
    hidden_monitor = SpikeMonitor(hidden_layer)
    output_monitor = SpikeMonitor(output_layer)
    
    # Мониторы состояний для анализа
    hidden_state = StateMonitor(hidden_layer, 'v', record=True)
    output_state = StateMonitor(output_layer, 'v', record=True)
    
    # Сборка сети
    net = Network()
    net.add(input_layer, hidden_layer, output_layer)
    net.add(input_syn, rec_syn, output_syn)
    net.add(input_monitor, hidden_monitor, output_monitor)
    net.add(hidden_state, output_state)
    
    print("🚀 Запуск улучшенной системы...")
    net.run(200*ms)
    
    # Результаты
    print("📊 Результаты:")
    print(f"  Входные спайки: {input_monitor.num_spikes}")
    print(f"  Скрытые спайки: {hidden_monitor.num_spikes}")
    print(f"  Выходные спайки: {output_monitor.num_spikes}")
    
    # Детальные результаты
    if hidden_monitor.num_spikes > 0:
        print(f"\n  Первые 15 скрытых спайков:")
        for i in range(min(15, len(hidden_monitor.t))):
            print(f"    Neuron {hidden_monitor.i[i]}, Time {hidden_monitor.t[i]/ms:.1f}ms")
    
    if output_monitor.num_spikes > 0:
        print(f"\n  Выходные спайки:")
        for i in range(len(output_monitor.t)):
            print(f"    Neuron {output_monitor.i[i]}, Time {output_monitor.t[i]/ms:.1f}ms")
    else:
        print(f"\n  Нет выходных спайков - проверим напряжение:")
        if len(output_state.t) > 0:
            max_v = np.max([np.max(trace) for trace in output_state.v if len(trace) > 0])
            print(f"    Максимальное напряжение выходных нейронов: {max_v:.3f}")
    
    return net, input_monitor, hidden_monitor, output_monitor

def analyze_network_dynamics(net, hidden_monitor, output_monitor):
    """Анализ динамики сети"""
    print("\n📈 Анализ динамики сети:")
    
    # Частота спайков
    if len(hidden_monitor.t) > 0:
        duration = 200  # ms
        firing_rate = len(hidden_monitor.t) / (duration/1000)  # в Hz
        print(f"  Частота спайков скрытого слоя: {firing_rate:.1f} Hz")
    
    # Активность по нейронам
    if len(hidden_monitor.i) > 0:
        unique_neurons, counts = np.unique(hidden_monitor.i, return_counts=True)
        print(f"  Активные нейроны скрытого слоя: {len(unique_neurons)}/{len(hidden_monitor.source)}")
        print(f"  Самый активный нейрон: Neuron {unique_neurons[np.argmax(counts)]} ({np.max(counts)} спайков)")

if __name__ == "__main__":
    net, input_mon, hidden_mon, output_mon = test_improved_system()
    analyze_network_dynamics(net, hidden_mon, output_mon)
