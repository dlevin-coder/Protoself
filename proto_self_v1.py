# proto_self_v1.py - первая версия proto self-object
from brian2 import *
import numpy as np

def create_proto_self_v1():
    """Первая версия proto self-object системы"""
    print("🧠 Создание Proto Self-Object v1.0")
    
    # Параметры
    defaultclock.dt = 1*ms
    
    # === МОДУЛЬ A: Сенсорный поток ===
    print("🎤 Модуль A: Сенсорный поток")
    
    # Создаем искусственные спайки (имитация cochlear processing)
    N_input = 16  # 16 частотных полос
    spike_times, spike_indices = generate_artificial_audio_spikes(N_input, duration=500)
    print(f"   Создано {len(spike_indices)} входных спайков")
    
    input_layer = SpikeGeneratorGroup(N_input, spike_indices, spike_times)
    
    # === МОДУЛЬ B: Наблюдатель (STM) ===
    print("👀 Модуль B: Наблюдатель (STM)")
    
    # Рекуррентная сеть с интеграцией во времени
    stm_eqs = '''
    dv/dt = (I_sensory + I_prediction + I_self - v) / (20*ms) : 1
    dI_sensory/dt = -I_sensory / (5*ms) : 1
    dI_prediction/dt = -I_prediction / (10*ms) : 1
    dI_self/dt = -I_self / (50*ms) : 1
    '''
    
    stm_layer = NeuronGroup(32, stm_eqs, threshold='v>1', reset='v=0', method='euler')
    stm_layer.v = 'rand() * 0.1'
    
    # === МОДУЛЬ C: Предиктор ===
    print("🔮 Модуль C: Предиктор")
    
    # Копия STM с задержкой
    predictor_eqs = '''
    dv/dt = (I_stm + I_error - v) / (15*ms) : 1
    dI_stm/dt = -I_stm / (8*ms) : 1
    dI_error/dt = -I_error / (12*ms) : 1
    '''
    
    predictor_layer = NeuronGroup(16, predictor_eqs, threshold='v>0.8', reset='v=0', method='euler')
    predictor_layer.v = 'rand() * 0.1'
    
    # === МОДУЛЬ D: Долговременная память ===
    print("💾 Модуль D: Долговременная память (LTM)")
    
    ltm_eqs = '''
    dv/dt = (I_input - v) / (100*ms) : 1  # Очень медленная динамика
    dI_input/dt = -I_input / (50*ms) : 1
    '''
    
    ltm_layer = NeuronGroup(64, ltm_eqs, threshold='v>1.5', reset='v=0', method='euler')
    ltm_layer.v = 'rand() * 0.2'
    
    # Медленные рекуррентные связи (LTM)
    ltm_syn = Synapses(ltm_layer, ltm_layer, 'w : 1', on_pre='I_input_post += w')
    ltm_syn.connect(condition='i!=j', p=0.1)
    ltm_syn.w = '0.3*rand()'
    
    # === МОДУЛЬ E: Summary State (Proto Self) ===
    print("🎯 Модуль E: Summary State (Proto Self)")
    
    summary_eqs = '''
    dv/dt = (I_stm + I_ltm - v) / (200*ms) : 1  # Очень медленно меняется
    dI_stm/dt = -I_stm / (100*ms) : 1
    dI_ltm/dt = -I_ltm / (150*ms) : 1
    '''
    
    summary_layer = NeuronGroup(8, summary_eqs, threshold='v>1', reset='v=0', method='euler')
    summary_layer.v = 'rand() * 0.1'
    
    # === СОЕДИНЕНИЯ ===
    print("🔗 Создание соединений")
    
    # A → B (сенсорный вход в наблюдатель)
    sensory_to_stm = Synapses(input_layer, stm_layer, 'w : 1', on_pre='I_sensory_post += w')
    sensory_to_stm.connect(p=0.6)
    sensory_to_stm.w = '1.0 + 0.5*rand()'
    
    # B → C (наблюдатель в предиктор)
    stm_to_predictor = Synapses(stm_layer, predictor_layer, 'w : 1', on_pre='I_stm_post += w')
    stm_to_predictor.connect(p=0.5)
    stm_to_predictor.w = '0.8 + 0.4*rand()'
    
    # C → B (предикция обратно в наблюдатель)
    predictor_to_stm = Synapses(predictor_layer, stm_layer, 'w : 1', on_pre='I_prediction_post += w')
    predictor_to_stm.connect(p=0.4)
    predictor_to_stm.w = '0.6 + 0.3*rand()'
    
    # B → D (STM в LTM)
    stm_to_ltm = Synapses(stm_layer, ltm_layer, 'w : 1', on_pre='I_input_post += w')
    stm_to_ltm.connect(p=0.3)
    stm_to_ltm.w = '0.5 + 0.2*rand()'
    
    # D → E (LTM в summary state)
    ltm_to_summary = Synapses(ltm_layer, summary_layer, 'w : 1', on_pre='I_ltm_post += w')
    ltm_to_summary.connect(p=0.2)
    ltm_to_summary.w = '0.4 + 0.2*rand()'
    
    # E → B (summary state обратно в наблюдатель)
    summary_to_stm = Synapses(summary_layer, stm_layer, 'w : 1', on_pre='I_self_post += w')
    summary_to_stm.connect(p=0.5)
    summary_to_stm.w = '0.7 + 0.3*rand()'
    
    # === МОНИТОРЫ ===
    print("📊 Создание мониторов")
    
    monitors = {
        'input': SpikeMonitor(input_layer),
        'stm': SpikeMonitor(stm_layer),
        'predictor': SpikeMonitor(predictor_layer),
        'ltm': SpikeMonitor(ltm_layer),
        'summary': SpikeMonitor(summary_layer)
    }
    
    # === СБОРКА СЕТИ ===
    print("⚙️  Сборка сети")
    
    net = Network()
    net.add(input_layer, stm_layer, predictor_layer, ltm_layer, summary_layer)
    net.add(sensory_to_stm, stm_to_predictor, predictor_to_stm, 
            stm_to_ltm, ltm_to_summary, summary_to_stm, ltm_syn)
    net.add(*monitors.values())
    
    return net, monitors

def generate_artificial_audio_spikes(N_channels, duration=500):
    """Генерация искусственных аудио спайков"""
    spike_times = []
    spike_indices = []
    
    # Создаем реалистичные паттерны спайков
    for i in range(N_channels):
        # Разные частоты для разных каналов
        base_rate = 5 + i * 2  # от 5 до 35 Hz базовой активности
        # Периодические всплески
        for burst_start in range(20, duration, 80 + i*5):
            # Всплеск активности
            for t in range(burst_start, min(burst_start+15, duration), 3):
                if np.random.rand() < 0.7:  # 70% вероятность спайка
                    spike_times.append(t * ms)
                    spike_indices.append(i)
            # Базовая активность между всплесками
            for t in range(burst_start+15, min(burst_start+65, duration), 10):
                if np.random.rand() < (base_rate/1000):  # Вероятность по частоте
                    spike_times.append(t * ms)
                    spike_indices.append(i)
    
    return spike_times, spike_indices

def run_proto_self_test():
    """Тест запуска proto self-object"""
    net, monitors = create_proto_self_v1()
    
    print("🚀 Запуск Proto Self-Object...")
    net.run(500*ms)  # 500ms для формирования self-object
    
    # Анализ результатов
    print("\n📊 Результаты Proto Self-Object:")
    for name, monitor in monitors.items():
        print(f"  {name.upper()}: {monitor.num_spikes} спайков")
    
    # Анализ активности summary state
    summary_monitor = monitors['summary']
    if summary_monitor.num_spikes > 0:
        print(f"\n🎯 Summary State (Self-Object) активен!")
        print(f"   Частота: {summary_monitor.num_spikes / 0.5:.1f} Hz")
        print(f"   Активные нейроны: {len(np.unique(summary_monitor.i))}/{len(summary_monitor.source)}")
    else:
        print(f"\n⚠️  Summary State (Self-Object) не активен")
    
    return net, monitors

if __name__ == "__main__":
    net, monitors = run_proto_self_test()
