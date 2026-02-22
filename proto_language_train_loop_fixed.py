# proto_language_train_loop_fixed.py
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def run_snn_predictive_step(spike_indices, spike_times, N_input):
    """
    Исправленная версия с правильными единицами измерения
    """
    start_scope()
    
    prediction_delay = 25*ms  
    
    inputs = SpikeGeneratorGroup(N_input, spike_indices, spike_times*ms)
    
    G_sensor = NeuronGroup(4, 'dv/dt = -v/(10*ms) : 1', 
                          threshold='v>1', reset='v=0', method='euler')
    
    G_gen = NeuronGroup(4, 'dv/dt = -v/(10*ms) : 1', 
                       threshold='v>1', reset='v=0', method='euler')
    
    syn_sensor = Synapses(inputs, G_sensor, on_pre='v_post += 0.8')
    syn_sensor.connect()
    
    # КЛЮЧ: задержка для предсказания
    syn_pred = Synapses(G_sensor, G_gen, on_pre='v_post += 0.3', delay=prediction_delay)
    syn_pred.connect()
    
    sm_sensor = SpikeMonitor(G_sensor)
    sm_gen = SpikeMonitor(G_gen)
    
    net = Network(inputs, G_sensor, G_gen, syn_sensor, syn_pred, sm_sensor, sm_gen)
    net.run(300*ms)
    
    # Правильный расчет ошибки с учетом задержки
    t_sensor = sm_sensor.t
    t_gen = sm_gen.t
    t_sensor_shifted = t_sensor - prediction_delay  # сдвигаем на величину предсказания
    
    # Сравниваем в окнах (исправляем единицы измерения)
    dt_window = 20.0  # в миллисекундах как число
    total_duration = 300.0  # в миллисекундах как число
    prediction_error = 0.0
    
    # Конвертируем времена в миллисекунды для расчетов
    t_sensor_ms = np.asarray(t_sensor/ms)
    t_gen_ms = np.asarray(t_gen/ms)
    t_sensor_shifted_ms = np.asarray(t_sensor_shifted/ms)
    
    for t_start in np.arange(0.0, total_duration, dt_window):
        t_end = t_start + dt_window
        gen_count = np.sum((t_gen_ms >= t_start) & (t_gen_ms < t_end))
        sensor_count = np.sum((t_sensor_shifted_ms >= t_start) & (t_sensor_shifted_ms < t_end))
        prediction_error += abs(gen_count - sensor_count)
    
    return {
        "spikes_sensor": sm_sensor.count[:],
        "spikes_gen": sm_gen.count[:],
        "prediction_error": prediction_error
    }

# ==========================
# Параметры
# ==========================
N_input = 4
epochs = 10

# ==========================
# Генерация тестовой последовательности
# ==========================
def generate_test_sequence():
    """Стандартная последовательность для обучения"""
    spike_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    spike_times = [10, 30, 50, 80, 100, 120, 150, 170, 190]
    return spike_indices, spike_times

# ==========================
# Логирование
# ==========================
error_history = []

# ==========================
# Основной цикл обучения
# ==========================
print("=== Цикл обучения с предсказанием ===")
spike_indices, spike_times = generate_test_sequence()

for epoch in range(epochs):
    out = run_snn_predictive_step(spike_indices, spike_times, N_input)
    
    print(
        f"Эпоха {epoch}:",
        "сенсор:", out["spikes_sensor"], 
        "генеративный:", out["spikes_gen"],
        "ошибка:", float(out["prediction_error"])
    )
    
    error_history.append(float(out["prediction_error"]))

# ==========================
# Сохранение результатов
# ==========================
np.save("prediction_error_history.npy", error_history)

# ==========================
# График ошибки
# ==========================
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), error_history, 'o-', linewidth=2, markersize=8)
plt.xlabel("Эпоха")
plt.ylabel("Ошибка предсказания")
plt.title("Прогресс обучения SNN (предсказательный режим)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prediction_error_progress.png")

print(f"\n✅ Готово! Сохранено:")
print(f"- История ошибок: prediction_error_history.npy")
print(f"- График: prediction_error_progress.png")
print(f"- Последняя ошибка: {error_history[-1]:.2f}")

# ==========================
# Анализ результата
# ==========================
if len(error_history) > 0 and error_history[-1] > 0:
    print("🎯 Система работает в предсказательном режиме!")
    print("   (Ошибка > 0 означает, что генеративный слой не просто копирует)")
else:
    print("⚠️  Система все еще работает как copy-loop")
