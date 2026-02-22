# Обновленный audio_spiketrain.py с rate coding
import numpy as np
from brian2 import *
from scipy.io import wavfile

def audio_to_spiketrain_loop(audio_file, N_input, threshold=0.2, repeat=1, coding_type='threshold'):
    """
    coding_type: 'threshold', 'rate', or 'onset'
    """
    
    # Загрузка аудио
    sr, audio = wavfile.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
    
    if repeat > 1:
        audio = np.tile(audio, repeat)
    
    if coding_type == 'threshold':
        return _threshold_coding(audio, sr, N_input, threshold)
    elif coding_type == 'rate':
        return _rate_coding(audio, sr, N_input)
    elif coding_type == 'onset':
        return _onset_coding(audio, sr, N_input)
    else:
        raise ValueError(f"Unknown coding type: {coding_type}")

def _threshold_coding(audio, sr, N_input, threshold):
    """Пороговое кодирование"""
    spike_indices = []
    spike_times = []
    dt = 1.0 / sr

    for t, sample in enumerate(audio):
        if abs(sample) > threshold:
            i = int((abs(sample) / 1.0) * (N_input - 1))
            i = min(max(i, 0), N_input - 1)
            spike_indices.append(i)
            spike_times.append(t * dt)

    return _create_spike_generator(N_input, spike_indices, spike_times, sr)

def _rate_coding(audio, sr, N_input, window_size_ms=10):
    """Rate coding: частота спайков пропорциональна средней амплитуде в окне"""
    spike_indices = []
    spike_times = []
    
    window_size = int(window_size_ms * sr / 1000)
    dt = 1.0 / sr
    
    for t in range(0, len(audio) - window_size, window_size):
        window = audio[t:t+window_size]
        avg_amplitude = np.mean(np.abs(window))
        
        # Число спайков пропорционально амплитуде (максимум 10 спайков на окно)
        num_spikes = int(avg_amplitude * 10)
        
        # Распределяем спайки по нейронам
        for spike_idx in range(num_spikes):
            neuron_idx = spike_idx % N_input
            # Спайк в середине окна
            spike_time = (t + window_size/2) * dt
            spike_indices.append(neuron_idx)
            spike_times.append(spike_time)
    
    return _create_spike_generator(N_input, spike_indices, spike_times, sr)

def _onset_coding(audio, sr, N_input):
    """Onset coding: спайки только при переходах через порог"""
    spike_indices = []
    spike_times = []
    dt = 1.0 / sr
    threshold = 0.1
    
    prev_sample = 0
    for t, sample in enumerate(audio):
        # Проверяем переход через порог
        if (prev_sample < threshold <= sample) or (prev_sample > -threshold >= sample):
            i = int((abs(sample) / 1.0) * (N_input - 1))
            i = min(max(i, 0), N_input - 1)
            spike_indices.append(i)
            spike_times.append(t * dt)
        prev_sample = sample
    
    return _create_spike_generator(N_input, spike_indices, spike_times, sr)

def _create_spike_generator(N_input, spike_indices, spike_times, sr):
    """Вспомогательная функция для создания SpikeGeneratorGroup"""
    if len(spike_times) == 0:
        print("WARNING: empty spike train")
        return SpikeGeneratorGroup(N_input, [], [] * second)
    
    spike_indices = np.array(spike_indices, dtype=int)
    spike_times = np.array(spike_times) * second
    
    # Сортировка
    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    spike_indices = spike_indices[order]
    
    # Проверки
    assert len(spike_times) == len(spike_indices)
    if len(spike_indices) > 0:
        assert spike_indices.min() >= 0
        assert spike_indices.max() < N_input
    
    dt_spike = (1.0 / sr) * second
    return SpikeGeneratorGroup(N_input, spike_indices, spike_times, dt=dt_spike)
