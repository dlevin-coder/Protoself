from brian2 import *
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import resample
import os

def create_and_analyze_ALE():
    """
    Создает и анализирует ALE-модель с барабанным ритмом
    """
    
    # Создаем директорию для результатов
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Загрузка аудио файла
    print("Загрузка аудио файла...")
    try:
        sample_rate, audio_data = wavfile.read('drums_5sec.wav')
        print(f"Аудио загружено: {sample_rate} Hz, длительность: {len(audio_data)/sample_rate:.2f} секунд")
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        # Создаем тестовый сигнал
        sample_rate = 44100
        duration = 5
        t = np.linspace(0, duration, sample_rate * duration)
        audio_data = np.zeros_like(t)
        # Простой ритм: kick каждые 500ms
        for i in range(0, int(duration*1000), 500):
            idx = int(i * sample_rate / 1000)
            if idx < len(audio_data) - 1000:
                # Создаем kick-звук
                env = np.exp(-np.linspace(0, 0.1, 1000) * 20)
                audio_data[idx:idx+1000] += np.sin(2*np.pi*60*np.linspace(0, 0.1, 1000)) * env
        audio_data = (audio_data * 32767).astype(np.int16)
        print("Создан тестовый барабанный паттерн")
    
    # Обработка стерео в моно если нужно
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Нормализация
    audio_data = audio_data.astype(float)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Даунсэмплинг до 1000 Hz для Brian2
    target_rate = 1000
    num_samples = int(len(audio_data) * target_rate / sample_rate)
    if num_samples > 0:
        audio_resampled = resample(audio_data, min(num_samples, 5000))
    else:
        audio_resampled = np.zeros(5000)
    
    # Создание спайкового потока из аудио
    def audio_to_spikes(audio_signal, num_channels=3):
        """Преобразует аудиосигнал в спайки для нескольких каналов"""
        spike_times_list = []
        spike_indices_list = []
        
        # Создаем несколько каналов с разными порогами
        thresholds = np.linspace(0.1, 0.5, num_channels)
        
        for channel, threshold in enumerate(thresholds):
            # Находим точки превышения порога
            above_threshold = np.abs(audio_signal) > threshold
            # Простое детектирование пиков
            peaks = []
            for i in range(1, len(above_threshold)-1):
                if above_threshold[i] and not above_threshold[i-1]:  # rising edge
                    peaks.append(i)
            
            # Добавляем рефрактерный период
            filtered_peaks = []
            last_peak = -100
            for peak in peaks:
                if peak - last_peak > 50:  # 50ms рефрактерный период
                    filtered_peaks.append(peak)
                    last_peak = peak
            
            # Конвертируем в спайки
            for peak in filtered_peaks:
                time_ms = (peak / target_rate) * 1000  # конвертируем в миллисекунды
                spike_times_list.append(time_ms)
                spike_indices_list.append(channel)
                
        return np.array(spike_indices_list), np.array(spike_times_list) * ms
    
    # Генерируем спайки
    spike_indices, spike_times = audio_to_spikes(audio_resampled, num_channels=3)
    
    print(f"Сгенерировано {len(spike_times)} спайков")
    
    # Параметры модели
    N_input = 3      
    N_hidden = 60    
    N_output = 30    
    tau = 20*ms      
    eta = 0.01       
    Delta = 5*ms     
    
    # Входной слой
    input_layer = SpikeGeneratorGroup(N_input, spike_indices, spike_times)
    
    # Задержанный вход
    delayed_input = NeuronGroup(
        N_input,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.5',
        reset='v = 0'
    )
    
    # Связь с задержкой
    input_to_delay = Synapses(
        input_layer, delayed_input,
        on_pre='v += 1',
        delay=Delta
    )
    input_to_delay.connect(j='i')
    
    # Генеративная сеть
    gen_network = NeuronGroup(
        N_hidden,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 1',
        reset='v = 0'
    )
    
    # Выходной слой
    output_layer = NeuronGroup(
        N_output,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 1',
        reset='v = 0'
    )
    
    # Адаптивный фильтр (ALE)
    adaptive_filter = Synapses(
        delayed_input, gen_network,
        '''
        w : 1
        dapre/dt = -apre/(20*ms) : 1 (clock-driven)
        dapost/dt = -apost/(20*ms) : 1 (clock-driven)
        ''',
        on_pre='''
        v_post += w
        apre += 1
        w = clip(w + 0.01 * apost, 0, 3)
        ''',
        on_post='''
        apost += 1
        w = clip(w + 0.01 * apre, 0, 3)
        '''
    )
    
    adaptive_filter.connect(p=0.5)
    adaptive_filter.w = '0.2 + rand() * 0.3'
    
    # Связь генеративной сети с выходом
    gen_to_output = Synapses(
        gen_network, output_layer,
        on_pre='v_post += 0.3'
    )
    gen_to_output.connect(p=0.3)
    
    # Мониторинг
    spike_mon_input = SpikeMonitor(input_layer)
    spike_mon_delayed = SpikeMonitor(delayed_input)
    spike_mon_gen = SpikeMonitor(gen_network)
    spike_mon_output = SpikeMonitor(output_layer)
    state_mon_weights = StateMonitor(adaptive_filter, 'w', record=range(min(10, len(adaptive_filter))))
    
    # Сборка сети
    net = Network()
    net.add(input_layer, delayed_input, gen_network, output_layer)
    net.add(input_to_delay, adaptive_filter, gen_to_output)
    net.add(spike_mon_input, spike_mon_delayed, spike_mon_gen, 
            spike_mon_output, state_mon_weights)
    
    return net, spike_mon_input, spike_mon_gen, spike_mon_output, state_mon_weights

def save_results_to_file(input_mon, gen_mon, output_mon, weight_mon):
    """
    Сохраняет результаты в текстовый файл
    """
    with open('results/experiment_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ALE\n")
        f.write("="*50 + "\n\n")
        
        # Входные данные
        f.write("ВХОДНОЙ ПОТОК:\n")
        f.write(f"  Спайков: {len(input_mon.t)}\n")
        f.write(f"  Активных каналов: {len(np.unique(input_mon.i))}\n")
        if len(input_mon.t) > 1:
            isi = np.diff(input_mon.t)
            f.write(f"  Средний межспайковой интервал: {np.mean(isi)/ms:.1f} мс\n")
        f.write("\n")
        
        # Генеративная сеть
        f.write("ГЕНЕРАТИВНАЯ СЕТЬ:\n")
        f.write(f"  Спайков: {len(gen_mon.t)}\n")
        f.write(f"  Активных нейронов: {len(np.unique(gen_mon.i))}\n")
        f.write("\n")
        
        # Выходной слой
        f.write("ВЫХОДНОЙ СЛОЙ (SELF-LIKE ПОВЕДЕНИЕ):\n")
        f.write(f"  Спайков: {len(output_mon.t)}\n")
        f.write(f"  Активных нейронов: {len(np.unique(output_mon.i))}\n")
        
        if len(output_mon.t) > 0:
            total_time = float(max(output_mon.t)/second)
            window_size = 1.0
            num_windows = int(total_time / window_size)
            
            f.write(f"  Анализ в {num_windows} окнах по {window_size} секунды:\n")
            
            if num_windows > 1:
                # Разбиваем на окна
                windows_activity = {}
                for i in range(num_windows):
                    start_time = i * window_size * second
                    end_time = (i + 1) * window_size * second
                    mask = (output_mon.t >= start_time) & (output_mon.t < end_time)
                    active_neurons = set(output_mon.i[mask])
                    windows_activity[i] = active_neurons
                
                # Анализ схожести
                similarities = []
                for i in range(len(windows_activity)):
                    for j in range(i+1, len(windows_activity)):
                        set1 = windows_activity[i]
                        set2 = windows_activity[j]
                        if len(set1) > 0 or len(set2) > 0:
                            intersection = len(set1.intersection(set2))
                            union = len(set1.union(set2))
                            if union > 0:
                                similarity = intersection / union
                                similarities.append(similarity)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    f.write(f"  Средняя схожесть окон: {avg_similarity*100:.1f}%\n")
                    
                    if avg_similarity > 0.2:
                        f.write("  ✓ НАБЛЮДАЕТСЯ SELF-LIKE ПОВЕДЕНИЕ\n")
                    elif avg_similarity > 0.05:
                        f.write("  ○ Умеренное self-like поведение\n")
                    else:
                        f.write("  ✗ Self-like поведение отсутствует\n")
        
        # Веса
        f.write("\nАДАПТАЦИЯ ВЕСОВ:\n")
        if hasattr(weight_mon, 'w') and len(weight_mon.w) > 0:
            initial_weights = [w[0] for w in weight_mon.w if len(w) > 0]
            final_weights = [w[-1] for w in weight_mon.w if len(w) > 0]
            
            if initial_weights and final_weights:
                avg_initial = np.mean(initial_weights)
                avg_final = np.mean(final_weights)
                f.write(f"  Средние веса: {avg_initial:.2f} → {avg_final:.2f}\n")
                if avg_final > avg_initial:
                    f.write("  ✓ Веса адаптировались - обучение произошло\n")
                else:
                    f.write("  ○ Веса не изменились значительно\n")

def save_plots(input_mon, gen_mon, output_mon, weight_mon):
    """
    Сохраняет графики в файлы PNG
    """
    # График входных спайков
    plt.figure(figsize=(15, 3))
    if len(input_mon.t) > 0:
        plt.scatter(input_mon.t/ms, input_mon.i, s=20, c='black', alpha=0.7)
    plt.title('Входные спайки (барабанный ритм)')
    plt.ylabel('Канал')
    plt.xlabel('Время (мс)')
    plt.savefig('results/input_spikes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График генеративной сети
    plt.figure(figsize=(15, 3))
    if len(gen_mon.t) > 0:
        plt.scatter(gen_mon.t/ms, gen_mon.i, s=5, c='red', alpha=0.6)
    plt.title('Генеративная сеть (обучение)')
    plt.ylabel('Нейрон')
    plt.xlabel('Время (мс)')
    plt.savefig('results/generative_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График выходного слоя
    plt.figure(figsize=(15, 3))
    if len(output_mon.t) > 0:
        plt.scatter(output_mon.t/ms, output_mon.i, s=10, c='green', alpha=0.7)
    plt.title('Выходной слой (self-like поведение)')
    plt.ylabel('Нейрон')
    plt.xlabel('Время (мс)')
    plt.savefig('results/output_layer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График эволюции весов
    plt.figure(figsize=(15, 3))
    if hasattr(weight_mon, 'w') and len(weight_mon.w) > 0:
        for i in range(min(len(weight_mon.w), 10)):
            if len(weight_mon.w[i]) > 0:
                plt.plot(weight_mon.t/ms, weight_mon.w[i], alpha=0.7, linewidth=1)
    plt.title('Эволюция весов (STDP адаптация)')
    plt.ylabel('Вес')
    plt.xlabel('Время (мс)')
    plt.savefig('results/weight_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment_with_saving():
    """
    Запуск эксперимента с сохранением результатов
    """
    print("Создание и анализ ALE-модели...")
    
    # Создание модели
    result = create_and_analyze_ALE()
    if result is None:
        return
    
    net, input_mon, gen_mon, output_mon, weight_mon = result
    
    # Запуск симуляции
    print("Запуск симуляции (6 циклов по 5 секунд)...")
    for cycle in range(3):
        print(f"Цикл {cycle + 1}/3")
        net.run(5*second)
    
    # Сохранение результатов
    print("Сохранение результатов...")
    save_results_to_file(input_mon, gen_mon, output_mon, weight_mon)
    save_plots(input_mon, gen_mon, output_mon, weight_mon)
    
    print("\nРезультаты сохранены в директории 'results':")
    print("- experiment_results.txt: текстовый анализ")
    print("- input_spikes.png: график входных спайков")
    print("- generative_network.png: активность генеративной сети")
    print("- output_layer.png: self-like поведение")
    print("- weight_evolution.png: адаптация весов")
    
    print("\n✅ Эксперимент завершен! Проверь директорию results.")

# Запуск
if __name__ == "__main__":
    run_experiment_with_saving()
