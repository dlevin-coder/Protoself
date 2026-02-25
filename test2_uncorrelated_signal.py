from brian2 import *
import numpy as np
import os

def generate_uncorrelated_signal(duration=5000):
    """Generates uncorrelated signal"""
    return np.random.randn(duration) * 0.3

def audio_to_spikes(audio_signal, num_channels=3):
    """Converts signal to spikes"""
    spike_times_list = []
    spike_indices_list = []
    
    thresholds = np.linspace(0.1, 0.5, num_channels)
    
    for channel, threshold in enumerate(thresholds):
        above_threshold = np.abs(audio_signal) > threshold
        peaks = []
        for i in range(1, len(above_threshold)-1):
            if above_threshold[i] and not above_threshold[i-1]:
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

def run_uncorrelated_test():
    """Run test with uncorrelated signal"""
    print("Test 2: Uncorrelated signal (B(t))")
    
    # Generate signal
    signal = generate_uncorrelated_signal()
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
    
    with open('test_results/uncorrelated_results.txt', 'w') as f:
        f.write("Test 2 Results: Uncorrelated signal\n")
        f.write("="*40 + "\n")
        f.write(f"Input spikes: {len(spike_mon_input.t)}\n")
        f.write(f"Generative network spikes: {len(spike_mon_gen.t)}\n")
        f.write(f"Output layer spikes: {len(spike_mon_output.t)}\n")
        if len(state_mon_weights.w) > 0 and len(state_mon_weights.w[0]) > 0:
            final_weights = [w[-1] for w in state_mon_weights.w if len(w) > 0]
            avg_weight = np.mean(final_weights) if final_weights else 0
            f.write(f"Average weight: {avg_weight:.3f}\n")
    
    print("Results saved to test_results/uncorrelated_results.txt")

if __name__ == "__main__":
    run_uncorrelated_test()
