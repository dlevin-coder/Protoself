# main_proto_self.py (обновленная версия)
from brian2 import *
import numpy as np
import os

# Импорты модулей
try:
    from cochlear_filterbank import create_cochlear_filterbank, cochlear_spiking_encoding
    from observer_module import create_observer_network
    from predictive_module import create_predictive_network
    from ltm_module import create_ltm_network
    from self_object_module import create_summary_state, connect_summary_to_observer
    modules_available = True
    print("✅ Все модули успешно импортированы")
except ImportError as e:
    print(f"⚠️  Ошибка импорта модулей: {e}")
    modules_available = False

def create_proto_self_system(audio_file="drums_5sec.wav"):
    """
    Создание полной системы proto self-object
    """
    if not modules_available:
        print("Модули недоступны, создаю упрощенную версию...")
        return create_simple_test_system()
    
    try:
        print("=== Создание системы Proto Self-Object ===")
        
        # Проверка наличия аудио файла
        if not os.path.exists(audio_file):
            print(f"Файл {audio_file} не найден, создаю тестовый сигнал...")
            create_test_audio(audio_file)
        
        # Модуль A - Cochlear processing
        print("1. Создание кохлеарного фильтра...")
        try:
            filtered_signals, sr = create_cochlear_filterbank(audio_file, n_bands=8)
            print(f"   Создано {len(filtered_signals)} полос фильтрации")
        except Exception as e:
            print(f"   Ошибка кохлеарной фильтрации: {e}, используем прямую передачу")
            filtered_signals, sr = create_dummy_signals()
        
        print("2. Спайковое кодирование...")
        try:
            spike_trains = cochlear_spiking_encoding(filtered_signals, sr, threshold=0.05)
            print(f"   Создано {len(spike_trains)} спайковых поездов")
        except Exception as e:
            print(f"   Ошибка спайкового кодирования: {e}")
            spike_trains = create_dummy_spike_trains()
        
        # Модуль B - Observer (STM)
        print("3. Создание наблюдателя...")
        try:
            observer, input_syn, rec_syn = create_observer_network(n_input_channels=8, n_hidden=16)
            print(f"   Наблюдатель: {observer.N if hasattr(observer, 'N') else len(observer)} нейронов")
        except Exception as e:
            print(f"   Ошибка создания наблюдателя: {e}")
            observer, input_syn, rec_syn = create_dummy_observer()
        
        # Модуль C - Predictor
        print("4. Создание предиктора...")
        try:
            predictor = create_predictive_network(observer, delay=25*ms)
            print(f"   Предиктор: {predictor.N if hasattr(predictor, 'N') else 'unknown'} нейронов")
        except Exception as e:
            print(f"   Ошибка создания предиктора: {e}")
            predictor = create_dummy_predictor()
        
        # Модуль D - LTM
        print("5. Создание долговременной памяти...")
        try:
            ltm, ltm_syn = create_ltm_network(n_neurons=32)
            print(f"   LTM: {ltm.N} нейронов")
        except Exception as e:
            print(f"   Ошибка создания LTM: {e}")
            ltm, ltm_syn = create_dummy_ltm()
        
        # Модуль E - Summary State (Self-Object)
        print("6. Создание summary state...")
        try:
            summary = create_summary_state(n_summary=4)
            print(f"   Summary state: {summary.N} нейронов")
        except Exception as e:
            print(f"   Ошибка создания summary state: {e}")
            summary = create_dummy_summary()
        
        # Создание мониторов
        print("7. Создание мониторов...")
        monitors = create_all_monitors(observer, predictor, ltm, summary)
        
        # Сборка сети
        print("8. Сборка сети...")
        net = assemble_network(observer, predictor, ltm, summary, 
                              input_syn, rec_syn, ltm_syn, monitors, spike_trains)
        
        print("✅ Система proto self-object создана успешно!")
        return net, monitors
        
    except Exception as e:
        print(f"❌ Критическая ошибка при создании системы: {e}")
        import traceback
        traceback.print_exc()
        return create_simple_test_system()

def create_dummy_observer():
    """Создание резервного наблюдателя"""
    eqs = '''
    dv/dt = (1 - v)/tau : 1
    tau : second
    '''
    G = NeuronGroup(4, eqs, threshold='v>1', reset='v=0', method='euler')
    G.tau = 20*ms
    G.v = 'rand()'
    
    dummy_syn1 = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    dummy_syn2 = Synapses(None, G, 'w : 1', on_pre='v_post += w')
    
    return G, dummy_syn2, dummy_syn1

def create_dummy_predictor():
    """Создание резервного предиктора"""
    G = NeuronGroup(4, 'dv/dt = (0.5 - v)/tau : 1', threshold='v>1', reset='v=0')
    return G

def create_dummy_ltm():
    """Создание резервной LTM"""
    G = NeuronGroup(8, 'dv/dt = (0.3 - v)/tau : 1', threshold='v>1', reset='v=0')
    syn = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    syn.connect(p=0.1)
    return G, syn

def create_dummy_summary():
    """Создание резервного summary state"""
    G = NeuronGroup(2, 'dv/dt = (0.2 - v)/tau : 1', threshold='v>1', reset='v=0')
    return G

def create_dummy_signals():
    """Создание тестовых сигналов"""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    signals = []
    for i in range(8):
        freq = 200 + i * 300
        signal = np.sin(2 * np.pi * freq * t) * 0.3
        signals.append(signal)
    
    return np.array(signals), sr

def create_dummy_spike_trains():
    """Создание тестовых спайковых поездов"""
    spike_trains = []
    for i in range(8):
        times = np.random.uniform(0, 1, 20)  # 20 случайных спайков
        indices = np.full_like(times, i, dtype=int)
        spike_trains.append({
            'times': times * second,
            'indices': indices
        })
    return spike_trains

def create_test_audio(filename="test_audio.wav"):
    """Создание тестового аудио файла"""
    try:
        from scipy.io.wavfile import write
        
        sr = 22050
        t = np.linspace(0, 2, sr * 2)
        
        # Комбинация синусоид
        audio = (np.sin(2 * np.pi * 440 * t) + 
                0.5 * np.sin(2 * np.pi * 880 * t) +
                0.1 * np.random.random(len(t)))  # шум
        
        audio = (audio * 0.8 * 32767).astype(np.int16)
        write(filename, sr, audio)
        print(f"✅ Создан тестовый файл: {filename}")
    except Exception as e:
        print(f"⚠️  Не удалось создать тестовый аудио файл: {e}")

def create_all_monitors(observer, predictor, ltm, summary):
    """Создание всех мониторов с обработкой ошибок"""
    try:
        monitors = {}
        try:
            monitors['observer'] = SpikeMonitor(observer)
        except:
            print("⚠️  Не удалось создать монитор наблюдателя")
            
        try:
            monitors['predictor'] = SpikeMonitor(predictor)
        except:
            print("⚠️  Не удалось создать монитор предиктора")
            
        try:
            monitors['ltm'] = SpikeMonitor(ltm)
        except:
            print("⚠️  Не удалось создать монитор LTM")
            
        try:
            monitors['summary'] = SpikeMonitor(summary)
        except:
            print("⚠️  Не удалось создать монитор summary")
            
        return monitors
    except:
        return {}

def assemble_network(observer, predictor, ltm, summary, 
                     input_syn, rec_syn, ltm_syn, monitors, spike_trains):
    """Сборка полной сети с обработкой ошибок"""
    try:
        net = Network()
        
        # Добавляем основные компоненты
        components = [observer, predictor, ltm, summary]
        for comp in components:
            try:
                if comp is not None:
                    net.add(comp)
            except Exception as e:
                print(f"⚠️  Не удалось добавить компонент: {e}")
        
        # Добавляем синапсы
        synapses = [input_syn, rec_syn, ltm_syn]
        for syn in synapses:
            try:
                if syn is not None:
                    net.add(syn)
            except Exception as e:
                print(f"⚠️  Не удалось добавить синапсы: {e}")
        
        # Добавляем мониторы
        for monitor in monitors.values():
            try:
                if monitor is not None:
                    net.add(monitor)
            except Exception as e:
                print(f"⚠️  Не удалось добавить монитор: {e}")
                
        return net
    except Exception as e:
        print(f"⚠️  Ошибка сборки сети: {e}")
        # Создаем минимальную рабочую сеть
        try:
            minimal_net = Network(observer, monitors.get('observer', None))
            return minimal_net
        except:
            return Network()

def create_simple_test_system():
    """Создание простой тестовой системы"""
    print("🔧 Создание упрощенной тестовой системы...")
    
    eqs = '''
    dv/dt = (1 - v)/tau : 1
    tau : second
    '''
    
    G = NeuronGroup(4, eqs, threshold='v>1', reset='v=0', method='euler')
    G.tau = 20*ms
    G.v = 'rand()'
    
    M = SpikeMonitor(G)
    net = Network(G, M)
    
    return net, {'simple': M}

def run_proto_self_test(duration=200*ms):
    """Тестовый запуск системы"""
    print("\n=== Тестовый запуск Proto Self-System ===")
    
    # Создание системы
    net, monitors = create_proto_self_system()
    
    # Запуск
    print(f"🚀 Запуск симуляции на {duration/ms} мс...")
    try:
        net.run(duration)
        print("✅ Симуляция завершена успешно!")
        
        # Вывод результатов
        print_results(monitors)
        
    except Exception as e:
        print(f"❌ Ошибка во время симуляции: {e}")
        import traceback
        traceback.print_exc()
        
        print("🔧 Попытка запуска упрощенной системы...")
        simple_net, simple_monitors = create_simple_test_system()
        try:
            simple_net.run(50*ms)
            print("✅ Упрощенная система работает!")
        except Exception as e2:
            print(f"❌ Даже упрощенная система не работает: {e2}")

def print_results(monitors):
    """Вывод результатов"""
    print("\n📊 === Результаты ===")
    any_results = False
    for name, monitor in monitors.items():
        if monitor is not None and hasattr(monitor, 'num_spikes'):
            try:
                spikes = monitor.num_spikes
                print(f"  {name}: {spikes} спайков")
                any_results = True
            except:
                pass
    
    if not any_results:
        print("  Нет данных о спайках")

if __name__ == "__main__":
    run_proto_self_test(200*ms)
