# affective_perception_working.py - рабочая версия аффективно-модулированной перцепции
from brian2 import *
import numpy as np

def create_working_affective_system():
    """Создание работающей системы с аффективной модуляцией"""
    print("🧠 Создание работающей аффективной системы")
    
    defaultclock.dt = 1*ms
    
    # Упрощенные нейроны с гарантированной активностью
    N_sensory = 4
    N_predictor = 4
    N_summary = 2
    
    sensory = NeuronGroup(N_sensory, 'v:1', threshold='False')
    
    # Предиктор с аффективной модуляцией - БОЛЕЕ ВОЗБУДИМЫЙ
    predictor_eqs = '''
    dv/dt = (I_in * (1 + affective_gain) - v) / (15*ms) : 1
    dI_in/dt = -I_in / (8*ms) : 1
    affective_gain : 1
    '''
    predictor = NeuronGroup(N_predictor, predictor_eqs, threshold='v>0.2', reset='v=0', method='euler')  # НИЗКИЙ ПОРОГ
    predictor.v = 0.05
    predictor.affective_gain = 0.0
    
    # Summary (аффективное состояние) - СДЕЛАЕМ ЕГО АКТИВНЫМ
    summary_eqs = '''
    dv/dt = (drive - v) / (30*ms) : 1
    drive : 1
    '''
    summary = NeuronGroup(N_summary, summary_eqs, threshold='v>0.1', reset='v=0', method='euler')  # НИЗКИЙ ПОРОГ
    summary.v = 0.05
    summary.drive = 0.0
    
    # Соединения - УСИЛЕННЫЕ
    sens_to_pred = Synapses(sensory, predictor, 'w : 1', on_pre='I_in_post += w')
    sens_to_pred.connect()
    sens_to_pred.w = '2.0 + rand()'  # СИЛЬНЫЕ ВЕСА
    
    # Аффективная модуляция
    @network_operation(dt=2*ms)
    def update_affection():
        if len(summary.v) > 0:
            mean_v = np.mean(summary.v)
            # Применяем ко всем нейронам предиктора
            predictor.affective_gain = 1.0 * mean_v  # СИЛЬНАЯ МОДУЛЯЦИЯ
            # print(f"Аффект: {mean_v:.3f}, Модуляция: {predictor.affective_gain[0]:.3f}")

    # Мониторы
    monitors = {
        'sensory': SpikeMonitor(sensory),
        'predictor': SpikeMonitor(predictor),
        'summary': SpikeMonitor(summary),
        'summary_v': StateMonitor(summary, 'v', record=True),
        'pred_gain': StateMonitor(predictor, 'affective_gain', record=True),
        'pred_v': StateMonitor(predictor, 'v', record=True)
    }
    
    net = Network()
    net.add(sensory, predictor, summary)
    net.add(sens_to_pred, update_affection)
    net.add(*monitors.values())
    
    return net, monitors, sensory, summary, predictor

def test_working_affective_perception():
    """Тест работающей аффективной перцепции"""
    print("\n🎯 ТЕСТ: Работающая аффективная перцепция")
    print("=" * 40)
    
    net, monitors, sensory, summary, predictor = create_working_affective_system()
    
    # === ТЕСТ 1: Радостное состояние ===
    print("😊 Радостное состояние:")
    summary.drive = 1.0  # СИЛЬНАЯ радость
    net.run(50*ms)  # ДАЕМ ВРЕМЯ СТАБИЛИЗИРОВАТЬСЯ
    
    happy_v = np.mean(summary.v) if len(summary.v) > 0 else 0
    happy_gain = np.mean(predictor.affective_gain) if len(predictor.affective_gain) > 0 else 0
    print(f"   Σ активность: {happy_v:.3f}")
    print(f"   Аффективная модуляция: {happy_gain:.3f}")
    
    # Стимул - БОЛЕЕ ИНТЕНСИВНЫЙ
    @network_operation(dt=1*ms)
    def stimulus_happy():
        if defaultclock.t < 150*ms:
            for i in range(4):
                sensory.v[i] = 3.0 + np.random.rand()  # СИЛЬНЫЙ СТИМУЛ

    net.add(stimulus_happy)
    net.run(100*ms)
    net.remove(stimulus_happy)
    
    happy_spikes = monitors['predictor'].num_spikes
    print(f"   Предсказания при радости: {happy_spikes}")
    
    # === ТЕСТ 2: Грустное состояние ===
    print("\n😢 Грустное состояние:")
    summary.drive = -0.8  # СИЛЬНАЯ грусть
    net.run(50*ms)  # СТАБИЛИЗАЦИЯ
    
    sad_v = np.mean(summary.v) if len(summary.v) > 0 else 0
    sad_gain = np.mean(predictor.affective_gain) if len(predictor.affective_gain) > 0 else 0
    print(f"   Σ активность: {sad_v:.3f}")
    print(f"   Аффективная модуляция: {sad_gain:.3f}")
    
    # Тот же стимул
    @network_operation(dt=1*ms)
    def stimulus_sad():
        if defaultclock.t < 300*ms:
            for i in range(4):
                sensory.v[i] = 3.0 + np.random.rand()  # ТОТ ЖЕ СТИМУЛ

    net.add(stimulus_sad)
    net.run(100*ms)
    
    sad_spikes = monitors['predictor'].num_spikes - happy_spikes  # Только новые спайки
    print(f"   Предсказания при грусти: {sad_spikes}")
    
    # === АНАЛИЗ ===
    print("\n🔍 АНАЛИЗ:")
    print(f"   Разница в аффекте: {abs(happy_v - sad_v):.3f}")
    print(f"   Разница в модуляции: {abs(happy_gain - sad_gain):.3f}")
    print(f"   Разница в предсказаниях: {abs(happy_spikes - sad_spikes)}")
    print(f"   Радостные предсказания: {happy_spikes}")
    print(f"   Грустные предсказания: {sad_spikes}")
    
    # Оценка
    if abs(happy_v - sad_v) > 0.2 and abs(happy_spikes - sad_spikes) > 10:
        print("\n🎉 УСПЕХ!")
        print("🎭 АФФЕКТ ВЛИЯЕТ НА ПЕРЦЕПЦИЮ!")
        print("⚡ Self-object модулирует генеративную модель!")
        return True
    elif abs(happy_v - sad_v) > 0.1 and abs(happy_spikes - sad_spikes) > 5:
        print("\n👍 ХОРОШО!")
        print("🎭 Есть аффективная модуляция перцепции!")
        return True
    else:
        print("\n⚠️  Недостаточная модуляция")
        return False

if __name__ == "__main__":
    test_working_affective_perception()
