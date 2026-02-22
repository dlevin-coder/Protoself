# minimal_affective_test.py - минимальный рабочий тест аффективной модуляции
from brian2 import *
import numpy as np

def test_minimal_affective_modulation():
    """Минимальный тест аффективной модуляции"""
    print("🔬 МИНИМАЛЬНЫЙ ТЕСТ АФФЕКТИВНОЙ МОДУЛЯЦИИ")
    print("=" * 45)
    
    defaultclock.dt = 1*ms
    
    # Аффективное состояние (Σ)
    summary_eqs = '''
    dv/dt = (drive - v) / (20*ms) : 1
    drive : 1  # Прямое аффективное управление
    '''
    summary = NeuronGroup(1, summary_eqs, threshold='v>0.5', reset='v=0', method='euler')
    summary.v = 0.3
    summary.drive = 0.0
    
    # Генеративная модель с аффективной модуляцией
    predictor_eqs = '''
    dv/dt = (stimulus * (1 + affective_mod) - v) / (10*ms) : 1
    stimulus : 1  # Внешний стимул
    affective_mod : 1  # Аффективная модуляция
    '''
    predictor = NeuronGroup(1, predictor_eqs, threshold='v>0.8', reset='v=0', method='euler')
    predictor.v = 0.1
    predictor.stimulus = 0.0
    predictor.affective_mod = 0.0
    
    # Мониторы
    sum_mon = StateMonitor(summary, 'v', record=0)
    pred_mon = StateMonitor(predictor, ['v', 'affective_mod'], record=0)
    pred_spike = SpikeMonitor(predictor)
    
    # Сеть
    net = Network(summary, predictor, sum_mon, pred_mon, pred_spike)
    
    # === ТЕСТ 1: Радостное состояние ===
    print("😊 ТЕСТ 1: Радостное состояние")
    summary.drive = 1.5  # Сильная радость
    net.run(100*ms)
    
    happy_state = float(sum_mon.v[0][-1]) if len(sum_mon.v[0]) > 0 else 0
    print(f"   Аффективное состояние: {happy_state:.3f}")
    
    # Применяем аффективную модуляцию
    mod_value = 0.8 * happy_state
    predictor.affective_mod = mod_value
    print(f"   Аффективная модуляция: {float(mod_value):.3f}")
    
    # Подаем стимул
    predictor.stimulus = 2.0
    net.run(50*ms)
    
    happy_spikes = pred_spike.num_spikes
    print(f"   Спайки при радости: {happy_spikes}")
    
    # Сохраняем модуляцию для радости
    happy_mod = float(mod_value)
    
    # === ТЕСТ 2: Грустное состояние ===
    print("\n😢 ТЕСТ 2: Грустное состояние")
    summary.drive = -1.2  # Сильная грусть
    net.run(100*ms)
    
    sad_state = float(sum_mon.v[0][-1]) if len(sum_mon.v[0]) > 0 else 0
    print(f"   Аффективное состояние: {sad_state:.3f}")
    
    # Применяем аффективную модуляцию
    mod_value2 = 0.8 * sad_state
    predictor.affective_mod = mod_value2
    print(f"   Аффективная модуляция: {float(mod_value2):.3f}")
    
    # Подаем тот же стимул
    predictor.stimulus = 2.0
    net.run(50*ms)
    
    sad_spikes = pred_spike.num_spikes - happy_spikes
    print(f"   Спайки при грусти: {sad_spikes}")
    
    # === АНАЛИЗ ===
    print("\n🔍 РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"   Разница в аффекте: {abs(happy_state - sad_state):.3f}")
    print(f"   Разница в модуляции: {abs(happy_mod - float(mod_value2)):.3f}")
    print(f"   Радостные спайки: {happy_spikes}")
    print(f"   Грустные спайки: {sad_spikes}")
    
    # Проверяем зависимости
    affect_difference = abs(happy_state - sad_state)
    mod_difference = abs(happy_mod - float(mod_value2))
    spike_difference = abs(happy_spikes - sad_spikes)
    
    if affect_difference > 0.5 and spike_difference > 2:
        print("\n🎉 ВОСХИТИТЕЛЬНО!")
        print("🎭 АФФЕКТИВНАЯ МОДУЛЯЦИЯ ПОДТВЕРЖДЕНА!")
        print("⚡ Self-object влияет на генеративную модель!")
        return True
    elif affect_difference > 0.2 and spike_difference > 0:
        print("\n👍 ХОРОШО!")
        print("🎭 Есть аффективная модуляция!")
        return True
    else:
        print("\n⚠️  Модуляция слабо выражена")
        return False

if __name__ == "__main__":
    test_minimal_affective_modulation()
