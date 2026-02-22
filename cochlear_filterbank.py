# cochlear_filterbank.py
import numpy as np
from brian2 import *
from scipy import signal

def create_cochlear_filterbank(audio_file, n_bands=16):
    """
    Создает банк фильтров, имитирующих улитку
    """
    # Загрузка аудио
    from scipy.io import wavfile
    sr, audio = wavfile.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    
    # Полосно-пропускающие фильтры (гамма-тоновые)
    center_freqs = np.logspace(np.log10(100), np.log10(4000), n_bands)
    
    filtered_signals = []
    for freq in center_freqs:
        # Простая реализация - можно улучшить
        try:
            sos = signal.butter(4, [max(50, freq*0.8), min(sr/2-100, freq*1.2)], 
                              btype='band', fs=sr, output='sos')
            filtered = signal.sosfilt(sos, audio)
            filtered_signals.append(filtered)
        except:
            # Fallback если фильтр не работает
            filtered_signals.append(audio * 0.1)
    
    return np.array(filtered_signals), sr

def cochlear_spiking_encoding(filtered_signals, sr, threshold=0.1):
    """
    Rate + timing кодирование для каждой полосы
    """
    n_bands, n_samples = filtered_signals.shape
    spike_trains = []
    
    dt = 1.0 / sr
    
    for band_idx, signal_band in enumerate(filtered_signals):
        spike_times = []
        neuron_indices = []
        
        # Rate coding based on envelope
        try:
            analytic_signal = signal.hilbert(signal_band)
            envelope = np.abs(analytic_signal)
        except:
            envelope = np.abs(signal_band)  # fallback
        
        # Простое спайковое кодирование
        for t in range(0, len(envelope), int(sr/100)):  # каждые 10ms
            amplitude = envelope[t] if t < len(envelope) else 0
            if amplitude > threshold:
                # Число спайков пропорционально амплитуде
                n_spikes = int(amplitude * 10)
                for _ in range(n_spikes):
                    spike_times.append(t * dt)
                    neuron_indices.append(band_idx)
        
        spike_trains.append({
            'times': np.array(spike_times) * second,
            'indices': np.array(neuron_indices)
        })
    
    return spike_trains
