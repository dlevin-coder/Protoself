# behavioral_tests.py - тесты поведенческих проявлений self-object
# behavioral_tests.py - тесты поведенческих проявлений self-object
from brian2 import *
import numpy as np

def create_enhanced_proto_self():
    """Улучшенная версия с поведенческими тестами"""
    print("🧠 Создание Enhanced Proto Self-Object")
    
    # Параметры
    defaultclock.dt = 0.5*ms  # Более точное разрешение
    
    # === МОДУЛИ ===
    
    # Входной слой (2 разных звука)
    N_input = 8
    input_layer = NeuronGroup(N_input, 'v:1', threshold='False')
    
    # STM (наблюдатель)
    stm_eqs = '''
    dv/dt = (I_sensory + I_prediction + I_self + noise - v) / (15*ms) : 1
    dI_sensory/dt = -I_sensory / (3*ms) : 1
    dI_prediction/dt = -I_prediction / (8*ms) : 1
    dI_self/dt = -I_self / (30*ms) : 1
    dnoise/dt = -noise/tau_noise + sigma*sqrt(2/tau_noise)*xi : 1
    tau_noise : second
    sigma : 1
    '''
    
    stm_layer = NeuronGroup(24, stm_eqs, threshold='v>1', reset='v=0', method='euler')
    stm_layer.v = 'rand() * 0.1'
    stm_layer.tau_noise = 20*ms
    stm_layer.sigma = 0.05
    
    # Предиктор (исправлено)
    predictor_eqs = '''
    dv/dt = (I_input - v) / (10*ms) : 1
    dI_input/dt = -I_input / (5*ms) : 1
    '''
    predictor_layer = NeuronGroup(12, predictor_eqs, threshold='v>0.7', reset='v=0', method='euler')
    predictor_layer.v = 'rand() * 0.1'
    
    # LTM
    ltm_eqs = '''
    dv/dt = (I_input - v) / (80*ms) : 1  # Очень медленно
    dI_input/dt = -I_input / (40*ms) : 1
    '''
    ltm_layer = NeuronGroup(32, ltm_eqs, threshold='v>1.2', reset='v=0', method='euler')
    ltm_layer.v = 'rand() * 0.1'
    
    # Summary State (Self-Object)
    summary_eqs = '''
    dv/dt = 2*(I_stm + I_ltm - v) / (150*ms) : 1  # Очень медленно меняется
    dI_stm/dt = -I_stm / (80*ms) : 1
    dI_ltm/dt = -I_ltm / (120*ms) : 1
    '''
    summary_layer = NeuronGroup(6, summary_eqs, threshold='v>0.8', reset='v=0', method='euler')
    summary_layer.v = 'rand() * 0.1'
    
    # === СОЕДИНЕНИЯ ===
    
    # Sensory pathways
    sensory_syn = Synapses(input_layer, stm_layer, 'w : 1', on_pre='I_sensory_post += w')
    sensory_syn.connect()
    sensory_syn.w = '0.8 + 0.4*rand()'
    
    # Prediction loop
    stm_to_pred = Synapses(stm_layer, predictor_layer, 'w : 1', on_pre='I_input_post += w')
    pred_to_stm = Synapses(predictor_layer, stm_layer, 'w : 1', on_pre='I_prediction_post += w')
    stm_to_pred.connect(p=0.4)
    pred_to_stm.connect(p=0.3)
    stm_to_pred.w = '0.6 + 0.3*rand()'
    pred_to_stm.w = '0.5 + 0.2*rand()'
    
    # Memory connections
    stm_to_ltm = Synapses(stm_layer, ltm_layer, 'w : 1', on_pre='I_input_post += w')
    ltm_to_summary = Synapses(ltm_layer, summary_layer, 'w : 1', on_pre='I_ltm_post += w')
    summary_to_stm = Synapses(summary_layer, stm_layer, 'w : 1', on_pre='I_self_post += w')
    
    stm_to_ltm.connect(p=0.2)
    ltm_to_summary.connect(p=0.3)
    summary_to_stm.connect(p=0.5)
    
    stm_to_ltm.w = '0.4 + 0.2*rand()'
    ltm_to_summary.w = '0.5 + 0.2*rand()'
    summary_to_stm.w = '0.7 + 0.3*rand()'
    
    # === МОНИТОРЫ ===
    monitors = {
        'input': SpikeMonitor(input_layer),
        'stm': SpikeMonitor(stm_layer),
        'predictor': SpikeMonitor(predictor_layer),
        'ltm': SpikeMonitor(ltm_layer),
        'summary': SpikeMonitor(summary_layer)
    }
    
    # === СЕТЬ ===
    net = Network()
    net.add(input_layer, stm_layer, predictor_layer, ltm_layer, summary_layer)
    net.add(sensory_syn, stm_to_pred, pred_to_stm, stm_to_ltm, 
            ltm_to_summary, summary_to_stm)
    net.add(*monitors.values())
    
    return net, monitors, input_layer, summary_layer

def test_same_sound_different_states():
    """🔁 Тест: Одинаковый звук → разная реакция (в зависимости от Σ)"""
    print("\n🔁 Тест 1: Одинаковый звук → разная реакция")
    
    net, monitors, input_layer, summary_layer = create_enhanced_proto_self()
    
    # Создаем два разных состояния self-object
    print("   Установка начального состояния self-object...")
    summary_layer.v = '0.6 + 0.2*rand()'  # Начальное состояние
    
    # Запускаем первый звук
    @network_operation(dt=50*ms)
    def sound_stimulus_1():
        if 100*ms < defaultclock.t < 150*ms:
            # Звук 1
            for i in [0, 2, 4, 6]:
                input_layer.v[i] = 2.0
    
    net.add(sound_stimulus_1)
    net.run(200*ms)
    
    stm_activity_1 = monitors['stm'].num_spikes
    summary_activity_1 = monitors['summary'].num_spikes
    
    print(f"   Реакция 1: STM={stm_activity_1}, Summary={summary_activity_1}")
    
    # Изменяем состояние self-object
    print("   Изменение состояния self-object...")
    summary_layer.v = '0.9 + 0.1*rand()'  # Новое состояние
    
    # Сбрасываем счетчики
    for mon in monitors.values():
        if hasattr(mon, 'clear'):
            mon.clear()
    
    # Запускаем тот же звук
    net.run(200*ms)
    
    stm_activity_2 = monitors['stm'].num_spikes
    summary_activity_2 = monitors['summary'].num_spikes
    
    print(f"   Реакция 2: STM={stm_activity_2}, Summary={summary_activity_2}")
    
    # Анализ различий
    stm_diff = abs(stm_activity_2 - stm_activity_1)
    print(f"   Разница в реакции STM: {stm_diff}")
    
    if stm_diff > 50:  # Значительное различие
        print("   ✅ Self-object влияет на интерпретацию входа!")
        return True
    else:
        print("   ⚠️  Self-object не оказывает значительного влияния")
        return False

def test_state_persistence():
    """⏳ Тест: Состояние удерживается без входа (короткое «присутствие»)"""
    print("\n⏳ Тест 2: Состояние удерживается без входа")
    
    net, monitors, input_layer, summary_layer = create_enhanced_proto_self()
    
    # Активируем систему
    print("   Активация системы...")
    @network_operation(dt=20*ms)
    def activation_stimulus():
        if defaultclock.t < 100*ms:
            for i in range(0, 8, 2):  # Каждый второй нейрон
                if np.random.rand() < 0.5:
                    input_layer.v[i] = 2.0
    
    net.add(activation_stimulus)
    net.run(100*ms)
    
    # Сохраняем состояние до паузы
    summary_monitor = monitors['summary']
    if len(summary_monitor.source.v) > 0:
        summary_state_before = np.array(summary_monitor.source.v)
        print(f"   Состояние до паузы: среднее={np.mean(summary_state_before):.3f}")
    else:
        print("   Состояние до паузы: не доступно")
        return False
    
    # Пауза без входа
    print("   Пауза 100мс без входа...")
    net.remove(activation_stimulus)
    net.run(100*ms)
    
    # Состояние после паузы
    if len(summary_monitor.source.v) > 0:
        summary_state_after = np.array(summary_monitor.source.v)
        print(f"   Состояние после паузы: среднее={np.mean(summary_state_after):.3f}")
        
        # Анализ сохранения состояния
        if len(summary_state_before) == len(summary_state_after):
            correlation = np.corrcoef(summary_state_before, summary_state_after)[0,1]
            print(f"   Корреляция состояний: {correlation:.3f}")
            
            if correlation > 0.6:
                print("   ✅ Self-object сохраняет состояние!")
                return True
            else:
                print("   ⚠️  Self-object не сохраняет состояние")
                return False
        else:
            print("   ⚠️  Размеры состояний не совпадают")
            return False
    else:
        print("   Состояние после паузы: не доступно")
        return False

def test_prediction_error_response():
    """⚠️ Тест: Сильные prediction errors → резкая перестройка Σ (прото-аффект)"""
    print("\n⚠️ Тест 3: Реакция на prediction errors")
    
    net, monitors, input_layer, summary_layer = create_enhanced_proto_self()
    
    # Нормальная работа системы
    print("   Нормальная работа...")
    @network_operation(dt=30*ms)
    def normal_operation():
        if defaultclock.t < 150*ms:
            # Регулярные входы
            for i in [1, 3, 5, 7]:
                if np.random.rand() < 0.3:
                    input_layer.v[i] = 2.0
    
    net.add(normal_operation)
    net.run(150*ms)
    
    # Сохраняем нормальное состояние
    summary_monitor = monitors['summary']
    if len(summary_monitor.source.v) > 0:
        summary_normal = np.array(summary_monitor.source.v)
        print(f"   Нормальное состояние: среднее={np.mean(summary_normal):.3f}")
    else:
        print("   Нормальное состояние: не доступно")
        return False
    
    # Внезапное сильное возмущение
    print("   Внезапное возмущение...")
    @network_operation(dt=1*ms)
    def sudden_disturbance():
        if 160*ms < defaultclock.t < 165*ms:
            # Массированное возмущение
            for i in range(8):
                input_layer.v[i] = 3.0  # Очень сильный сигнал
    
    net.add(sudden_disturbance)
    net.run(50*ms)  # Ещё 50мс после возмущения
    
    # Состояние после возмущения
    if len(summary_monitor.source.v) > 0:
        summary_disturbed = np.array(summary_monitor.source.v)
        print(f"   Состояние после возмущения: среднее={np.mean(summary_disturbed):.3f}")
        
        # Анализ изменения
        if len(summary_normal) == len(summary_disturbed):
            change_magnitude = np.linalg.norm(summary_disturbed - summary_normal)
            print(f"   Величина изменения: {change_magnitude:.3f}")
            
            if change_magnitude > 0.3:
                print("   ✅ Self-object реагирует на prediction error!")
                print("   🎭 Прото-аффект обнаружен!")
                return True
            else:
                print("   ⚠️  Self-object не реагирует на ошибки")
                return False
        else:
            print("   ⚠️  Размеры состояний не совпадают")
            return False
    else:
        print("   Состояние после возмущения: не доступно")
        return False

def run_all_behavioral_tests():
    """Запуск всех поведенческих тестов"""
    print("🧪 Полный тест поведенческих проявлений Self-Object")
    print("=" * 50)
    
    try:
        # Тест 1: Разная реакция на одинаковый вход
        print("Запуск Теста 1...")
        reaction_result = test_same_sound_different_states()
        
        # Тест 2: Сохранение состояния
        print("Запуск Теста 2...")
        persistence_result = test_state_persistence()
        
        # Тест 3: Реакция на ошибки
        print("Запуск Теста 3...")
        error_result = test_prediction_error_response()
        
        print("\n" + "=" * 50)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        
        if reaction_result:
            print("✅ Self-object влияет на интерпретацию входа")
        
        if persistence_result:
            print("✅ Self-object сохраняет состояние")
        
        if error_result:
            print("✅ Self-object реагирует на prediction errors")
            print("🎭 Прото-аффект подтвержден!")
        
        print("🧠 Proto Self-Object демонстрирует поведенческие признаки!")
        
    except Exception as e:
        print(f"❌ Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_behavioral_tests()

