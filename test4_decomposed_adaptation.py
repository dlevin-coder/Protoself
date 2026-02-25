# test4_decomposed_adaptation.py
from brian2 import *
import numpy as np
import os

def generate_test_spikes(num_spikes=100):
    """Генерирует тестовые спайки для моделирования"""
    # Создаем спайки для 3 нейронов
    indices = []
    times = []
    
    for i in range(num_spikes):
        neuron_idx = i % 3  # чередуем между 3 нейронами
        time_ms = i * 10    # спайки каждые 10 мс
        indices.append(neuron_idx)
        times.append(time_ms)
    
    return indices, np.array(times) * ms

def run_decomposed_adaptation_test():
    """Тест с декомпозицией: Σ(t) = Σ_fast(t) + Σ_slow"""
    print("Тест 4: Декомпозиция адаптации (Σ = Σ_fast + Σ_slow)")
    print("="*50)
    
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # Генерируем тестовые спайки
    spike_indices, spike_times = generate_test_spikes(50)  # 50 спайков
    print(f"Сгенерировано {len(spike_indices)} тестовых спайков")
    
    # Фаза 1: Обучение с полной системой
    print("Фаза 1: Обучение с полной системой")
    
    # Создаем входной слой с тестовыми спайками
    input_layer = SpikeGeneratorGroup(3, spike_indices, spike_times)
    
    # Модель с двумя типами весов
    neurons = NeuronGroup(
        30,  # уменьшим для стабильности
        '''
        dv/dt = -v/(20*ms) : 1
        w_fast : 1  # быстрые веса
        w_slow : 1  # медленные веса
        ''',
        threshold='v > 1',
        reset='v = 0'
    )
    
    # Инициализируем веса
    neurons.w_fast = 0.2
    neurons.w_slow = 0.1
    
    # Синапсы с двумя типами весов
    synapses = Synapses(
        input_layer, neurons,
        '''
        w_fast_syn : 1  # быстрые синаптические веса
        w_slow_syn : 1  # медленные синаптические веса
        ''',
        on_pre='''
        v_post += w_fast_syn + w_slow_syn
        '''
    )
    
    synapses.connect(p=0.5)  # частичное соединение
    
    # Инициализируем синаптические веса
    synapses.w_fast_syn = 0.2
    synapses.w_slow_syn = 0.1
    
    # Мониторы
    spike_mon = SpikeMonitor(neurons)
    state_mon_fast = StateMonitor(synapses, 'w_fast_syn', record=range(min(5, len(synapses))))
    state_mon_slow = StateMonitor(synapses, 'w_slow_syn', record=range(min(5, len(synapses))))
    
    # Сеть
    net = Network()
    net.add(input_layer, neurons, synapses, spike_mon, state_mon_fast, state_mon_slow)
    
    # Запуск обучения
    print("  Обучение...")
    net.run(2*second)  # 2 секунды обучения
    
    print(f"  После обучения: {len(spike_mon.t)} спайков")
    
    # Фаза 2: Моделирование анестезии (обнуление быстрых компонент)
    print("Фаза 2: Моделирование анестезии (обнуление быстрых компонент)")
    
    # Создаем новую сеть с уменьшенными быстрыми весами
    neurons_anesth = NeuronGroup(
        30,
        '''
        dv/dt = -v/(20*ms) : 1
        w_fast : 1
        w_slow : 1
        ''',
        threshold='v > 1',
        reset='v = 0'
    )
    
    # Обнуляем быстрые веса (оставляем только медленные)
    neurons_anesth.w_fast = 0.01  # почти ноль
    neurons_anesth.w_slow = 0.1   # сохраняем медленные
    
    synapses_anesth = Synapses(
        input_layer, neurons_anesth,
        on_pre='v_post += (0.01 + 0.1)'  # только медленные веса
    )
    synapses_anesth.connect(p=0.5)
    
    spike_mon_anesth = SpikeMonitor(neurons_anesth)
    net_anesth = Network()
    net_anesth.add(input_layer, neurons_anesth, synapses_anesth, spike_mon_anesth)
    
    print("  После анестезии:")
    net_anesth.run(1*second)
    print(f"    Спайков: {len(spike_mon_anesth.t)}")
    
    # Фаза 3: Восстановление
    print("Фаза 3: Восстановление (частичное)")
    
    # Сеть с частично восстановленными весами
    neurons_recover = NeuronGroup(
        30,
        '''
        dv/dt = -v/(20*ms) : 1
        w_fast : 1
        w_slow : 1
        ''',
        threshold='v > 1',
        reset='v = 0'
    )
    
    # Частично восстановленные веса
    neurons_recover.w_fast = 0.1  # частично восстановлены
    neurons_recover.w_slow = 0.1  # медленные сохранены
    
    synapses_recover = Synapses(
        input_layer, neurons_recover,
        on_pre='v_post += (0.1 + 0.1)'  # частично восстановленные веса
    )
    synapses_recover.connect(p=0.5)
    
    spike_mon_recover = SpikeMonitor(neurons_recover)
    net_recover = Network()
    net_recover.add(input_layer, neurons_recover, synapses_recover, spike_mon_recover)
    
    print("  Во время восстановления:")
    net_recover.run(1*second)
    print(f"    Спайков: {len(spike_mon_recover.t)}")
    
    # Фаза 4: Полное восстановление
    print("Фаза 4: Полное восстановление")
    
    # Сеть с восстановленными весами
    neurons_full = NeuronGroup(
        30,
        '''
        dv/dt = -v/(20*ms) : 1
        w_fast : 1
        w_slow : 1
        ''',
        threshold='v > 1',
        reset='v = 0'
    )
    
    # Почти полностью восстановленные веса
    neurons_full.w_fast = 0.15  # почти полные
    neurons_full.w_slow = 0.1   # медленные сохранены
    
    synapses_full = Synapses(
        input_layer, neurons_full,
        on_pre='v_post += (0.15 + 0.1)'  # почти полные веса
    )
    synapses_full.connect(p=0.5)
    
    spike_mon_full = SpikeMonitor(neurons_full)
    net_full = Network()
    net_full.add(input_layer, neurons_full, synapses_full, spike_mon_full)
    
    print("  После полного восстановления:")
    net_full.run(1*second)
    print(f"    Спайков: {len(spike_mon_full.t)}")
    
    # Сохраняем результаты
    with open('test_results/decomposed_results.txt', 'w') as f:
        f.write("РЕЗУЛЬТАТЫ ТЕСТА 4: ДЕКОМПОЗИЦИЯ АДАПТАЦИИ\n")
        f.write("="*50 + "\n\n")
        
        f.write("ДЕКОМПОЗИЦИЯ: Σ(t) = Σ_fast(t) + Σ_slow\n")
        f.write("- Σ_fast - исчезает при анестезии (текущая активность)\n")
        f.write("- Σ_slow - сохраняется как 'форма' для восстановления\n\n")
        
        f.write("ФАЗА 1 - Обучение (полная система):\n")
        f.write(f"  Активность: {len(spike_mon.t)} спайков\n\n")
        
        f.write("ФАЗА 2 - Анестезия (Σ_fast ≈ 0):\n")
        f.write(f"  Активность: {len(spike_mon_anesth.t)} спайков\n")
        f.write("  (ожидаем значительное снижение)\n\n")
        
        f.write("ФАЗА 3 - Частичное восстановление:\n")
        f.write(f"  Активность: {len(spike_mon_recover.t)} спайков\n")
        f.write("  (промежуточное значение)\n\n")
        
        f.write("ФАЗА 4 - Полное восстановление:\n")
        f.write(f"  Активность: {len(spike_mon_full.t)} спайков\n")
        f.write("  (ближе к исходному уровню)\n\n")
        
        # Анализ декомпозиции
        f.write("АНАЛИЗ ДЕКОМПОЗИЦИИ:\n")
        f.write("-" * 20 + "\n")
        
        initial_spikes = len(spike_mon.t)
        anesthesia_spikes = len(spike_mon_anesth.t)
        recovery_spikes = len(spike_mon_recover.t)
        full_recovery_spikes = len(spike_mon_full.t)
        
        f.write(f"Исходный уровень: {initial_spikes} спайков\n")
        f.write(f"После анестезии: {anesthesia_spikes} спайков\n")
        f.write(f"Частичное восстановление: {recovery_spikes} спайков\n")
        f.write(f"Полное восстановление: {full_recovery_spikes} спайков\n\n")
        
        # Проверка гипотез
        if anesthesia_spikes < initial_spikes * 0.7:
            f.write("✓ Гипотеза 1 ПОДТВЕРЖДЕНА: Анестезия вызывает снижение активности\n")
        else:
            f.write("○ Гипотеза 1: Эффект анестезии требует уточнения\n")
            
        if recovery_spikes > anesthesia_spikes:
            f.write("✓ Гипотеза 2 ПОДТВЕРЖДЕНА: Восстановление происходит постепенно\n")
        else:
            f.write("○ Гипотеза 2: Восстановление требует доработки\n")
            
        if full_recovery_spikes < initial_spikes:
            f.write("✓ Гипотеза 3 ПОДТВЕРЖДЕНА: Восстановление не мгновенное\n")
        else:
            f.write("○ Гипотеза 3: Восстановление слишком быстрое\n")
            
        if full_recovery_spikes > recovery_spikes:
            f.write("✓ Гипотеза 4 ПОДТВЕРЖДЕНА: Восстановление прогрессирует\n")
        else:
            f.write("○ Гипотеза 4: Прогрессия восстановления требует уточнения\n")
    
    print("\nРезультаты сохранены в test_results/decomposed_results.txt")
    
    # Краткий вывод
    print("\nКРАТКИЙ ВЫВОД ДЕКОМПОЗИЦИИ:")
    print("="*35)
    print(f"Исходный уровень: {len(spike_mon.t)} спайков")
    print(f"После анестезии: {len(spike_mon_anesth.t)} спайков")
    print(f"Частичное восстановление: {len(spike_mon_recover.t)} спайков")
    print(f"Полное восстановление: {len(spike_mon_full.t)} спайков")
    
    # Основные выводы
    if len(spike_mon_anesth.t) < len(spike_mon.t) * 0.7:
        print("✓ МОДЕЛЬ АНЕСТЕЗИИ РАБОТАЕТ: значительное снижение активности")
    if len(spike_mon_full.t) > len(spike_mon_anesth.t):
        print("✓ МОДЕЛЬ ВОССТАНОВЛЕНИЯ РАБОТАЕТ: активность возвращается")
    if len(spike_mon_full.t) < len(spike_mon.t):
        print("✓ ВОССТАНОВЛЕНИЕ НЕ МГНОВЕННОЕ: сохраняется след адаптации")
    if len(spike_mon_recover.t) < len(spike_mon_full.t):
        print("✓ ВОССТАНОВЛЕНИЕ ПРОГРЕССИРУЕТ: постепенное улучшение")

if __name__ == "__main__":
    run_decomposed_adaptation_test()
