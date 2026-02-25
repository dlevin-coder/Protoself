from brian2 import *
import numpy as np
import os

def generate_correlated_signal(duration=5000):
    """Generates correlated signal"""
    signal = np.zeros(duration)
    
    # Rhythmic pattern
    for i in range(0, duration, 500):  # kick drums
        if i + 100 < duration:
            env = np.exp(-np.linspace(0, 0.1, min(100, duration-i)) * 20)
            t_segment = np.linspace(0, 0.1, len(env))
            signal[i:i+len(env)] += np.sin(2*np.pi*60*t_segment) * env
    
    for i in range(0, duration, 125):  # hi-hats
        if i < duration:
            signal[i] = 0.3
            
    return signal

def audio_to_spikes(audio_signal, num_channels=3):
    """Converts signal to spikes"""
    spike_times_list = []
    spike_indices_list = []
    
    thresholds = np.linspace(0.1, 0.5, num_channels)
    
    for channel, threshold in enumerate(thresholds):
        peaks = []
        for i in range(1, len(audio_signal)-1):
            if (abs(audio_signal[i]) > threshold and 
                abs(audio_signal[i]) > abs(audio_signal[i-1]) and 
                abs(audio_signal[i]) > abs(audio_signal[i+1])):
                peaks.append(i)
        
        filtered_peaks = []
        last_peak = -100
        for peak in peaks:
            if peak - last_peak > 50:
                filtered_peaks.append(peak)
                last_peak = peak
        
        for peak in filtered_peaks:
            spike_times_list.append(peak)
            spike_indices_list.append(channel)
    
    if not spike_times_list:
        spike_times_list = [100, 200, 300]
        spike_indices_list = [0, 1, 2]
                
    return np.array(spike_indices_list), np.array(spike_times_list) * ms

def run_correlated_test():
    """Run test with correlated signal"""
    print("Test 1: Correlated signal (A(t))")
    
    # Generate signal
    signal = generate_correlated_signal()
    spike_indices, spike_times = audio_to_spikes(signal)
    
    # Create network
    input_layer = SpikeGeneratorGroup(3, spike_indices, spike_times)
    
    delayed_input = NeuronGroup(
        3, '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.3', reset='v = 0'
    )
    
    input_to_delay = Synapses(input_layer, delayed_input, on_pre='v += 1', delay=5*ms)
    input_to_delay.connect(j='i')
    
    gen_network = NeuronGroup(
        60, '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8', reset='v = 0'
    )
    
    output_layer = NeuronGroup(
        30, '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8', reset='v = 0'
    )
    
    # Adaptive filter with plasticity
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
    
    adaptive_filter.connect(p=0.5)
    adaptive_filter.w = '0.2 + rand() * 0.3'
    
    gen_to_output = Synapses(gen_network, output_layer, on_pre='v_post += 0.3')
    gen_to_output.connect(p=0.3)
    
    # Monitors
    spike_mon_input = SpikeMonitor(input_layer)
    spike_mon_gen = SpikeMonitor(gen_network)
    spike_mon_output = SpikeMonitor(output_layer)
    state_mon_weights = StateMonitor(adaptive_filter, 'w', record=range(5))
    
    # Network
    net = Network()
    net.add(input_layer, delayed_input, gen_network, output_layer)
    net.add(input_to_delay, adaptive_filter, gen_to_output)
    net.add(spike_mon_input, spike_mon_gen, spike_mon_output, state_mon_weights)
    
    # Run
    net.run(5*second)
    
    # Results
    print(f"Input spikes: {len(spike_mon_input.t)}")
    print(f"Generative network spikes: {len(spike_mon_gen.t)}")
    print(f"Output layer spikes: {len(spike_mon_output.t)}")
    
    if len(state_mon_weights.w) > 0 and len(state_mon_weights.w[0]) > 0:
        final_weights = [w[-1] for w in state_mon_weights.w if len(w) > 0]
        avg_weight = np.mean(final_weights) if final_weights else 0
        print(f"Average weight: {avg_weight:.3f}")
    
    # Save results
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    with open('test_results/correlated_results.txt', 'w') as f:
        f.write("Test 1 Results: Correlated signal\n")
        f.write("="*40 + "\n")
        f.write(f"Input spikes: {len(spike_mon_input.t)}\n")
        f.write(f"Generative network spikes: {len(spike_mon_gen.t)}\n")
        f.write(f"Output layer spikes: {len(spike_mon_output.t)}\n")
        if len(state_mon_weights.w) > 0 and len(state_mon_weights.w[0]) > 0:
            final_weights = [w[-1] for w in state_mon_weights.w if len(w) > 0]
            avg_weight = np.mean(final_weights) if final_weights else 0
            f.write(f"Average weight: {avg_weight:.3f}\n")
    
    print("Results saved to test_results/correlated_results.txt")

if __name__ == "__main__":
    run_correlated_test()
