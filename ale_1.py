from brian2 import *
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import resample

def create_ALE_with_drums():
    """
    Создает ALE-модель с барабанным ритмом
    """
    
    # Загрузка аудио файла
    print("Загрузка аудио файла...")
    try:
        sample_rate, audio_data = wavfile.read('drums_5sec.wav')
        print(f"Аудио загружено: {sample_rate} Hz, длительность: {len(audio_data)/sample_rate:.2f} секунд")
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        print("Создаем тестовый барабанный паттерн...")
        # Создаем тестовый ритмичный сигнал
        sample_rate = 44100
        duration = 5  # 5 секунд
        t = np.linspace(0, duration, sample_rate * duration)
        # Простой барабанный паттерн
        audio_data = np.zeros_like(t)
        # Kick барабаны
        for i in range(0, int(duration), 1):
            idx = int(i * sample_rate)
            if idx < len(audio_data):
                audio_data[idx:idx+1000] += np.sin(2*np.pi*60*np.linspace(0, 0.02, 1000)) * np.exp(-np.linspace(0, 0.02, 1000)*50)
        # Hi-hats
        for i in range(0, int(duration*4), 1):
            idx = int(i * sample_rate/4)
            if idx < len(audio_data):
                audio_data[idx] = 0.5
        
        audio_data = (audio_data * 32767).astype(np.int16)
    
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
        audio_resampled = resample(audio_data, min(num_samples, 5000))  # ограничиваем до 5 секунд
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
    
    if len(spike_times) == 0:
        # Если нет спайков, создаем тестовые
        print("Создаем тестовые спайки...")
        spike_indices = np.array([0, 0, 1, 1, 2, 2] * 20)  # 20 повторений
        spike_times = np.array([i*250 for i in range(120)]) * ms  # каждые 250ms
    
    print(f"Сгенерировано {len(spike_times)} спайков")
    
    # Параметры модели
    N_input = 3      # количество входных каналов
    N_hidden = 60    # размер генеративной сети
    N_output = 30    # размер выходного слоя
    tau = 20*ms      # временная константа
    tau_pre = 20*ms  # STDP pre
    tau_post = 20*ms # STDP post
    eta = 0.01       # скорость обучения
    Delta = 5*ms     # задержка для ALE
    
    # Входной слой
    input_layer = SpikeGeneratorGroup(N_input, spike_indices, spike_times)
    
    # Задержанный вход (копия с задержкой)
    delayed_input = NeuronGroup(
        N_input,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.5',
        reset='v = 0'
    )
    
    # Связь с задержкой
    input_to_delay = Synapses(
        input_layer, delayed_input,
        on_pre='v += 1'
    )
    input_to_delay.connect(j='i')
    
    # Добавляем задержку через параметр delay
    input_to_delay.delay = Delta
    
    # Генеративная сеть (предсказатель)
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
    
    # Адаптивный фильтр (ALE - STDP learning)
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
    
    # Полносвязное соединение
    adaptive_filter.connect(p=0.5)
    adaptive_filter.w = '0.2 + rand() * 0.3'  # начальные веса
    
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

def run_experiment():
    """
    Запуск эксперимента с барабанным ритмом
    """
    print("Создание модели...")
    try:
        net, input_mon, gen_mon, output_mon, weight_mon = create_ALE_with_drums()
    except Exception as e:
        print(f"Ошибка создания модели: {e}")
        return None, None, None, None, None
    
    print("Запуск симуляции...")
    # Запускаем несколько циклов для обучения
    try:
        for cycle in range(3):
            print(f"Цикл {cycle + 1}/3")
            net.run(5*second)
    except Exception as e:
        print(f"Ошибка во время симуляции: {e}")
        return net, input_mon, gen_mon, output_mon, weight_mon
    
    print("Анализ результатов...")
    # Визуализация
    try:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Входные спайки
        if len(input_mon.t) > 0:
            axes[0].scatter(input_mon.t/ms, input_mon.i, s=1, c='black', alpha=0.7)
        axes[0].set_title('Входные спайки (барабанный ритм)')
        axes[0].set_ylabel('Канал')
        
        # Активность генеративной сети
        if len(gen_mon.t) > 0:
            axes[1].scatter(gen_mon.t/ms, gen_mon.i, s=1, c='red', alpha=0.6)
        axes[1].set_title('Генеративная сеть (предсказания)')
        axes[1].set_ylabel('Нейрон')
        
        # Выходной слой
        if len(output_mon.t) > 0:
            axes[2].scatter(output_mon.t/ms, output_mon.i, s=2, c='green', alpha=0.7)
        axes[2].set_title('Выходной слой (self-like поведение)')
        axes[2].set_ylabel('Нейрон')
        
        # Эволюция весов
        if hasattr(weight_mon, 't') and len(weight_mon.t) > 0 and len(weight_mon.w) > 0:
            for i in range(min(len(weight_mon.w), 10)):
                axes[3].plot(weight_mon.t/ms, weight_mon.w[i], alpha=0.7, linewidth=0.8)
        axes[3].set_title('Эволюция весов (STDP адаптация)')
        axes[3].set_ylabel('Вес')
        axes[3].set_xlabel('Время (мс)')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
    
    return net, input_mon, gen_mon, output_mon, weight_mon

# Запуск эксперимента
if __name__ == "__main__":
    print("Запуск ALE-модели с барабанным ритмом")
    print("=====================================")
    
    try:
        net, input_mon, gen_mon, output_mon, weight_mon = run_experiment()
        if net is not None:
            print("\n✅ Эксперимент завершен успешно!")
        else:
            print("\n❌ Эксперимент не удался")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
