# minimal_test.py - минимальная рабочая версия
from brian2 import *
import numpy as np

def test_minimal_system():
    """Тест минимальной рабочей системы"""
    print("🧪 Тест минимальной SNN системы")
    
    # Создаем простую сеть
    eqs = '''
    dv/dt = (I - v) / (10*ms) : 1
    dI/dt = -I / (5*ms) : 1
    '''
    
    # Входной слой - используем SpikeGeneratorGroup для реальных спайков
    # Создаем искусственные спайки
    N_input = 8
    spike_times = []
    spike_indices = []
    
    # Генерируем спайки на регулярных интервалах
    for i in range(N_input):
        for t in range(10, 200, 20):  # каждые 20ms от 10ms до 200ms
            spike_times.append(t * ms)
            spike_indices.append(i)
    
    input_layer = SpikeGeneratorGroup(N_input, spike_indices, spike_times)
    
    # Скрытый слой (наблюдатель)
    hidden_layer = NeuronGroup(16, eqs, threshold='v>1', reset='v=0', method='euler')
    hidden_layer.v = 'rand() * 0.1'
    
    # Выходной слой
    output_layer = NeuronGroup(4, eqs, threshold='v>1', reset='v=0', method='euler')
    output_layer.v = 'rand() * 0.1'
    
    # Синапсы от входа к скрытым
    input_syn = Synapses(input_layer, hidden_layer, 'w : 1', on_pre='I_post += w')
    input_syn.connect(p=0.5)
    input_syn.w = '0.8 + 0.2*rand()'  # Сильные веса для гарантии спайков
    
    # Рекуррентные связи в скрытом слое
    rec_syn = Synapses(hidden_layer, hidden_layer, 'w : 1', on_pre='I_post += w')
    rec_syn.connect(condition='i!=j', p=0.3)
    rec_syn.w = '0.3*rand()'
    
    # От скрытого к выходному
    output_syn = Synapses(hidden_layer, output_layer, 'w : 1', on_pre='I_post += w')
    output_syn.connect(p=0.4)
    output_syn.w = '0.5*rand()'
    
    # Мониторы
    input_monitor = SpikeMonitor(input_layer)
    hidden_monitor = SpikeMonitor(hidden_layer)
    output_monitor = SpikeMonitor(output_layer)
    
    # Сборка сети
    net = Network()
    net.add(input_layer, hidden_layer, output_layer)
    net.add(input_syn, rec_syn, output_syn)
    net.add(input_monitor, hidden_monitor, output_monitor)
    
    print("🚀 Запуск минимальной системы...")
    net.run(200*ms)
    
    # Результаты
    print("📊 Результаты:")
    print(f"  Входные спайки: {input_monitor.num_spikes}")
    print(f"  Скрытые спайки: {hidden_monitor.num_spikes}")
    print(f"  Выходные спайки: {output_monitor.num_spikes}")
    
    # Детальные результаты
    if hidden_monitor.num_spikes > 0:
        print(f"  Первые 10 скрытых спайков:")
        for i in range(min(10, len(hidden_monitor.t))):
            print(f"    Neuron {hidden_monitor.i[i]}, Time {hidden_monitor.t[i]/ms:.1f}ms")
    
    return net

if __name__ == "__main__":
    test_minimal_system()
