# visualize_audio_spikes.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from audio_spiketrain import audio_to_spiketrain_loop
import os

def compare_waveform_spikes(audio_file, N_input=3, threshold=0.2):
    """Сравнение waveform и spike raster с сохранением в файл"""
    
    # Создаем директорию для результатов если её нет
    os.makedirs('results', exist_ok=True)
    
    # Загружаем аудио для визуализации waveform
    sr, audio = wavfile.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
    
    # Берем только первые 100 мс для наглядности
    samples_for_display = int(0.1 * sr)  # 100ms
    audio_segment = audio[:samples_for_display]
    time_axis = np.linspace(0, 0.1, len(audio_segment))
    
    # Генерируем спайки для той же части
    inputs = audio_to_spiketrain_loop(audio_file, N_input, threshold, repeat=1)
    
    # Создаем subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform
    ax1.plot(time_axis * 1000, audio_segment, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform')
    ax1.grid(True, alpha=0.3)
    
    # Добавляем пороговые линии
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    ax1.axhline(y=-threshold, color='r', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Spike raster - получаем данные правильно
    try:
        # Для SpikeGeneratorGroup используем внутренние переменные
        if hasattr(inputs, '_spike_time') and hasattr(inputs, '_neuron_index'):
            spike_times = inputs._spike_time
            spike_indices = inputs._neuron_index
            
            # Фильтруем спайки в пределах 100мс
            max_time = 0.1  # 100ms в секундах
            mask = spike_times <= max_time
            if np.any(mask):
                ax2.scatter(spike_times[mask] * 1000, spike_indices[mask], c='red', s=10)
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('Neuron Index')
                ax2.set_title('Spike Raster (corresponding to waveform)')
                ax2.set_xlim(0, 100)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No spikes in this time window', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Spike Raster (no spikes found)')
        else:
            # Альтернативный способ - создаем временный монитор
            temp_monitor = SpikeMonitor(inputs)
            # Запускаем короткую симуляцию для получения спайков
            temp_duration = 100*ms
            temp_net = Network(inputs, temp_monitor)
            temp_net.run(temp_duration)
            
            if len(temp_monitor.t) > 0:
                ax2.scatter(temp_monitor.t/ms, temp_monitor.i, c='red', s=10)
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('Neuron Index')
                ax2.set_title('Spike Raster (corresponding to waveform)')
                ax2.set_xlim(0, 100)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No spikes generated', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Spike Raster')
                
    except Exception as e:
        print(f"Warning: Could not get spike data: {e}")
        ax2.text(0.5, 0.5, 'Could not generate spike raster', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Spike Raster')
    
    plt.tight_layout()
    
    # Сохраняем в файл
    output_file = 'results/waveform_vs_spikes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен в: {output_file}")
    
    # Также сохраняем данные в текстовый файл
    save_comparison_data(time_axis, audio_segment, threshold)

def save_comparison_data(time_axis, audio_segment, threshold):
    """Сохранение данных для анализа в текстовый файл"""
    
    with open('results/comparison_data.txt', 'w') as f:
        f.write("=== Audio-Spikes Comparison Data ===\n\n")
        
        f.write("Audio segment (first 100ms):\n")
        f.write("Time(ms)\tAmplitude\n")
        for i, (time, amp) in enumerate(zip(time_axis[:20]*1000, audio_segment[:20])):  # Первые 20 точек
            f.write(f"{time:.3f}\t{amp:.6f}\n")
        
        f.write(f"\nThreshold: {threshold}\n")
        f.write("\nNote: Spike data requires running simulation to extract properly.\n")

def analyze_spike_statistics():
    """Анализ статистики спайков - вызывается после основной симуляции"""
    pass  # Эта функция будет использоваться в основном скрипте

# Простая версия для тестирования
if __name__ == "__main__":
    try:
        compare_waveform_spikes("drums_5sec.wav", N_input=3, threshold=0.2)
        print("Визуализация завершена успешно!")
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
        # Создаем простой отчет даже при ошибке
        os.makedirs('results', exist_ok=True)
        with open('results/error_report.txt', 'w') as f:
            f.write(f"Visualization error: {str(e)}\n")
        print("Создан отчет об ошибке в results/error_report.txt")
