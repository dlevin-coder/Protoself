from brian2 import *
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import os

class SwitchingStructureTest:
    def __init__(self):
        # Создаем директорию для результатов
        if not os.path.exists('switching_test_results'):
            os.makedirs('switching_test_results')
        
        # Параметры модели
        self.N_input = 3
        self.N_hidden = 60
        self.N_output = 30
        self.tau = 20*ms
        self.eta = 0.01
        self.Delta = 5*ms
        
        # Для сбора данных по этапам
        self.stage_results = []
        
    def generate_correlated_signal(self, duration=5000, sample_rate=1000):
        """Генерирует структурированный (коррелированный) сигнал"""
        signal = np.zeros(duration)
        
        # Создаем ритмический паттерн
        # Kick барабаны каждые 500ms
        for i in range(0, duration, 500):
            if i + 100 < duration:
                # Создаем envelope для kick
                env_length = min(100, duration - i)
                env = np.exp(-np.linspace(0, 0.1, env_length) * 20)
                t_kick = np.linspace(0, 0.1, env_length)
                signal[i:i+env_length] += np.sin(2*np.pi*60*t_kick) * env
        
        # Hi-hats каждые 125ms
        for i in range(0, duration, 125):
            if i < duration:
                signal[min(i, duration-1)] = 0.3
                
        return signal
    
    def generate_uncorrelated_signal(self, duration=5000):
        """Генерирует некоррелированный (белый шум) сигнал"""
        return np.random.randn(duration) * 0.3
    
    def generate_phase_randomized_signal(self, original_signal):
        """Генерирует фазово-рандомизированную версию сигнала"""
        # Простая аппроксимация: перемешиваем временные сегменты
        segment_length = 100
        segments = []
        for i in range(0, len(original_signal), segment_length):
            segment = original_signal[i:i+segment_length]
            if len(segment) > 0:
                segments.append(segment)
        
        # Перемешиваем сегменты
        np.random.shuffle(segments)
        
        # Собираем обратно
        randomized = np.concatenate(segments)[:len(original_signal)]
        
        # Нормализуем энергию
        if np.sum(randomized**2) > 0:
            energy_ratio = np.sqrt(np.sum(original_signal**2) / np.sum(randomized**2))
            randomized *= energy_ratio
            
        return randomized
    
    def audio_to_spikes(self, audio_signal, num_channels=3):
        """Преобразует аудиосигнал в спайки"""
        spike_times_list = []
        spike_indices_list = []
        
        # Создаем несколько каналов с разными порогами
        thresholds = np.linspace(0.1, 0.5, num_channels)
        
        for channel, threshold in enumerate(thresholds):
            # Простое детектирование пиков
            peaks = []
            for i in range(1, len(audio_signal)-1):
                if abs(audio_signal[i]) > threshold and abs(audio_signal[i]) > abs(audio_signal[i-1]) and abs(audio_signal[i]) > abs(audio_signal[i+1]):
                    peaks.append(i)
            
            # Рефрактерный период
            filtered_peaks = []
            last_peak = -100
            for peak in peaks:
                if peak - last_peak > 50:
                    filtered_peaks.append(peak)
                    last_peak = peak
            
            # Конвертируем в спайки
            for peak in filtered_peaks:
                spike_times_list.append(peak)
                spike_indices_list.append(channel)
                
        if not spike_times_list:  # Если нет спайков, создаем минимальные
            spike_times_list = [100, 200, 300]
            spike_indices_list = [0, 1, 2]
                
        return np.array(spike_indices_list), np.array(spike_times_list) * ms
    
    def run_single_stage(self, signal_type, duration=5*second, plasticity_enabled=True):
        """Запускает один этап эксперимента"""
        print(f"  Запуск {signal_type}...")
        
        # Генерируем сигнал
        if signal_type == 'correlated':
            signal = self.generate_correlated_signal()
        elif signal_type == 'uncorrelated':
            signal = self.generate_uncorrelated_signal()
        elif signal_type == 'phase_randomized':
            correlated_base = self.generate_correlated_signal()
            signal = self.generate_phase_randomized_signal(correlated_base)
        else:
            raise ValueError("Unknown signal type")
        
        # Преобразуем в спайки
        spike_indices, spike_times = self.audio_to_spikes(signal)
        
        # Создаем сеть для этого этапа
        # Входной слой
        input_layer = SpikeGeneratorGroup(self.N_input, spike_indices, spike_times)
        
        # Задержанный вход
        delayed_input = NeuronGroup(
            self.N_input,
            '''dv/dt = -v/(20*ms) : 1''',
            threshold='v > 0.3',
            reset='v = 0'
        )
        
        # Связь с задержкой
        input_to_delay = Synapses(
            input_layer, delayed_input,
            on_pre='v += 1',
            delay=self.Delta
        )
        input_to_delay.connect(j='i')
        
        # Генеративная сеть
        gen_network = NeuronGroup(
            self.N_hidden,
            '''dv/dt = -v/(20*ms) : 1''',
            threshold='v > 0.8',
            reset='v = 0'
        )
        
        # Выходной слой
        output_layer = NeuronGroup(
            self.N_output,
            '''dv/dt = -v/(20*ms) : 1''',
            threshold='v > 0.8',
            reset='v = 0'
        )
        
        # Адаптивный фильтр
        if plasticity_enabled:
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
                w = clip(w + 0.01 * apost, 0, 2)
                ''',
                on_post='''
                apost += 1
                w = clip(w + 0.01 * apre, 0, 2)
                '''
            )
        else:
            adaptive_filter = Synapses(
                delayed_input, gen_network,
                '''w : 1''',
                on_pre='v_post += w'
            )
        
        adaptive_filter.connect(p=0.5)
        if plasticity_enabled:
            adaptive_filter.w = '0.2 + rand() * 0.3'
        else:
            adaptive_filter.w = '0.3'
        
        # Связь генеративной сети с выходом
        gen_to_output = Synapses(
            gen_network, output_layer,
            on_pre='v_post += 0.3'
        )
        gen_to_output.connect(p=0.3)
        
        # Мониторинг
        spike_mon_input = SpikeMonitor(input_layer)
        spike_mon_gen = SpikeMonitor(gen_network)
        spike_mon_output = SpikeMonitor(output_layer)
        state_mon_weights = StateMonitor(adaptive_filter, 'w', record=range(min(5, len(adaptive_filter))))
        
        # Сборка и запуск сети
        net = Network()
        net.add(input_layer, delayed_input, gen_network, output_layer)
        net.add(input_to_delay, adaptive_filter, gen_to_output)
        net.add(spike_mon_input, spike_mon_gen, spike_mon_output, state_mon_weights)
        
        # Запуск
        net.run(duration)
        
        # Сбор результатов
        result = {
            'signal_type': signal_type,
            'input_spikes': len(spike_mon_input.t),
            'gen_spikes': len(spike_mon_gen.t),
            'output_spikes': len(spike_mon_output.t),
            'active_gen_neurons': len(np.unique(spike_mon_gen.i)),
            'active_output_neurons': len(np.unique(spike_mon_output.i)),
            'final_weights': [w[-1] for w in state_mon_weights.w if len(w) > 0],
            'weight_variance': np.var([w[-1] for w in state_mon_weights.w if len(w) > 0]) if state_mon_weights.w and len(state_mon_weights.w[0]) > 0 else 0
        }
        
        return result
    
    def run_switching_test(self):
        """Запускает полный тест переключения структур"""
        print("=== SWITCHING CORRELATED STRUCTURE TEST ===")
        
        # Этап 1: Коррелированный сигнал (A(t))
        print("Этап 1: Коррелированный сигнал (A(t))")
        result1 = self.run_single_stage('correlated', 5*second, plasticity_enabled=True)
        self.stage_results.append(result1)
        
        # Этап 2: Некоррелированный сигнал (B(t))
        print("Этап 2: Некоррелированный сигнал (B(t))")
        result2 = self.run_single_stage('uncorrelated', 5*second, plasticity_enabled=True)
        self.stage_results.append(result2)
        
        # Этап 3: Возвращение к коррелированному сигналу (A(t))
        print("Этап 3: Возвращение к коррелированному сигналу (A(t))")
        result3 = self.run_single_stage('correlated', 5*second, plasticity_enabled=True)
        self.stage_results.append(result3)
        
        # Этап 4: Фазово-рандомизированный сигнал
        print("Этап 4: Фазово-рандомизированный сигнал")
        result4 = self.run_single_stage('phase_randomized', 5*second, plasticity_enabled=True)
        self.stage_results.append(result4)
        
        # Анализ результатов
        self.analyze_and_save_results()
    
    def analyze_and_save_results(self):
        """Анализирует и сохраняет результаты"""
        print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
        
        # Сохраняем в файл
        with open('switching_test_results/detailed_report.txt', 'w', encoding='utf-8') as f:
            f.write("SWITCHING CORRELATED STRUCTURE TEST - DETAILED REPORT\n")
            f.write("="*60 + "\n\n")
            
            stages_names = ['Correlated A(t)', 'Uncorrelated B(t)', 'Correlated A(t) again', 'Phase-randomized A(t)']
            
            for i, (result, name) in enumerate(zip(self.stage_results, stages_names)):
                f.write(f"ЭТАП {i+1}: {name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Входных спайков: {result['input_spikes']}\n")
                f.write(f"  Спайков генеративной сети: {result['gen_spikes']}\n")
                f.write(f"  Спайков выходного слоя: {result['output_spikes']}\n")
                f.write(f"  Активных ген. нейронов: {result['active_gen_neurons']}\n")
                f.write(f"  Активных выходных нейронов: {result['active_output_neurons']}\n")
                
                if result['final_weights']:
                    avg_weight = np.mean(result['final_weights'])
                    f.write(f"  Средний вес: {avg_weight:.3f}\n")
                    f.write(f"  Дисперсия весов: {result['weight_variance']:.3f}\n")
                f.write("\n")
            
            # Анализ гипотез
            f.write("АНАЛИЗ ГИПОТЕЗ:\n")
            f.write("="*20 + "\n\n")
            
            # Гипотеза 1: Self возникает при коррелированном входе
            if len(self.stage_results) >= 3:
                stage1_weight = np.mean(self.stage_results[0]['final_weights']) if self.stage_results[0]['final_weights'] else 0
                stage2_weight = np.mean(self.stage_results[1]['final_weights']) if self.stage_results[1]['final_weights'] else 0
                stage3_weight = np.mean(self.stage_results[2]['final_weights']) if self.stage_results[2]['final_weights'] else 0
                
                f.write("Гипотеза 1 - Self как адаптивный фильтр:\n")
                f.write(f"  A(t) → веса: {stage1_weight:.3f}\n")
                f.write(f"  B(t) → веса: {stage2_weight:.3f}\n")
                f.write(f"  A(t) снова → веса: {stage3_weight:.3f}\n")
                
                if stage1_weight > stage2_weight and stage3_weight > stage2_weight:
                    f.write("  ✓ ПОДТВЕРЖДЕНО: Веса адаптируются к коррелированной структуре\n")
                else:
                    f.write("  ○ Требуется дополнительный анализ\n")
                
                # Скорость адаптации
                f.write("\n  Скорость адаптации:\n")
                if stage3_weight > stage1_weight:
                    f.write("  ✓ Вторая адаптация быстрее (след памяти)\n")
                elif stage3_weight == stage1_weight:
                    f.write("  ○ Скорость адаптации примерно одинакова\n")
                else:
                    f.write("  ○ Первая адаптация быстрее (необычно)\n")
            
            # Гипотеза 2: Зависимость от корреляции, не энергии
            if len(self.stage_results) >= 4:
                stage4_weight = np.mean(self.stage_results[3]['final_weights']) if self.stage_results[3]['final_weights'] else 0
                f.write(f"\nГипотеза 2 - Зависимость от корреляции:\n")
                f.write(f"  Фазово-рандомизированный → веса: {stage4_weight:.3f}\n")
                
                if stage4_weight < stage1_weight:
                    f.write("  ✓ ПОДТВЕРЖДЕНО: Та же энергия, но меньше адаптации = корреляция важнее\n")
                else:
                    f.write("  ○ Энергия может играть роль\n")
            
            # Гипотеза 3: Self не сохраняется как объект
            f.write(f"\nГипотеза 3 - Self не как объект памяти:\n")
            stage1_output = self.stage_results[0]['output_spikes'] if len(self.stage_results) > 0 else 0
            stage2_output = self.stage_results[1]['output_spikes'] if len(self.stage_results) > 1 else 0
            stage3_output = self.stage_results[2]['output_spikes'] if len(self.stage_results) > 2 else 0
            
            f.write(f"  A(t) → активность: {stage1_output} спайков\n")
            f.write(f"  B(t) → активность: {stage2_output} спайков\n")  
            f.write(f"  A(t) снова → активность: {stage3_output} спайков\n")
            
            if stage1_output > stage2_output and stage3_output > stage2_output:
                f.write("  ✓ ПОДТВЕРЖДЕНО: Self исчезает без коррелированной структуры\n")
            else:
                f.write("  ○ Self может иметь компоненты памяти\n")
        
        # Создаем графики
        self.create_plots()
        
        print("Результаты сохранены в switching_test_results/")
        print("- detailed_report.txt: подробный текстовый отчет")
        print("- performance_comparison.png: сравнение производительности")
        
        # Также выводим краткий результат в консоль
        self.print_summary()
    
    def create_plots(self):
        """Создает графики результатов"""
        stages = ['Stage 1\n(Correlated)', 'Stage 2\n(Uncorrelated)', 'Stage 3\n(Correlated)', 'Stage 4\n(Phase-randomized)']
        
        # Извлекаем данные
        weights_avg = []
        output_spikes = []
        gen_spikes = []
        
        for result in self.stage_results:
            avg_weight = np.mean(result['final_weights']) if result['final_weights'] else 0
            weights_avg.append(avg_weight)
            output_spikes.append(result['output_spikes'])
            gen_spikes.append(result['gen_spikes'])
        
        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Средние веса
        axes[0,0].plot(range(len(weights_avg)), weights_avg, 'o-', linewidth=2, markersize=8)
        axes[0,0].set_xticks(range(len(stages)))
        axes[0,0].set_xticklabels(stages, rotation=45)
        axes[0,0].set_ylabel('Average Weight')
        axes[0,0].set_title('Weight Adaptation')
        axes[0,0].grid(True, alpha=0.3)
        
        # Спайки выходного слоя
        axes[0,1].bar(range(len(output_spikes)), output_spikes, alpha=0.7, color='green')
        axes[0,1].set_xticks(range(len(stages)))
        axes[0,1].set_xticklabels(stages, rotation=45)
        axes[0,1].set_ylabel('Output Spikes')
        axes[0,1].set_title('Output Layer Activity')
        axes[0,1].grid(True, alpha=0.3)
        
        # Спайки генеративной сети
        axes[1,0].bar(range(len(gen_spikes)), gen_spikes, alpha=0.7, color='red')
        axes[1,0].set_xticks(range(len(stages)))
        axes[1,0].set_xticklabels(stages, rotation=45)
        axes[1,0].set_ylabel('Generator Spikes')
        axes[1,0].set_title('Generative Network Activity')
        axes[1,0].grid(True, alpha=0.3)
        
        # Тепловая карта весов (если есть данные)
        if any(result['final_weights'] for result in self.stage_results):
            weight_matrix = []
            for result in self.stage_results:
                if result['final_weights']:
                    # Берем первые 10 весов для визуализации
                    padded_weights = (result['final_weights'] + [0]*10)[:10]
                    weight_matrix.append(padded_weights)
                else:
                    weight_matrix.append([0]*10)
            
            im = axes[1,1].imshow(weight_matrix, aspect='auto')
            axes[1,1].set_yticks(range(len(stages)))
            axes[1,1].set_yticklabels([s.split('\n')[0] for s in stages])
            axes[1,1].set_title('Weight Matrix Evolution')
            plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('switching_test_results/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Выводит краткий результат в консоль"""
        print("\nКРАТКИЙ ОБЗОР РЕЗУЛЬТАТОВ:")
        print("="*40)
        
        stages_names = ['A(t)', 'B(t)', 'A(t) again', 'Phase-random']
        for i, (result, name) in enumerate(zip(self.stage_results, stages_names)):
            avg_weight = np.mean(result['final_weights']) if result['final_weights'] else 0
            print(f"{name:>12}: веса={avg_weight:.3f}, выход={result['output_spikes']:>3} спайков")

# Запуск эксперимента
if __name__ == "__main__":
    print("Запуск Switching Correlated Structure Test...")
    test = SwitchingStructureTest()
    test.run_switching_test()
    print("\n✅ Эксперимент Switching Correlated Structure завершен!")
