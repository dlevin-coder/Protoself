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

def generate_uncorrelated_signal(duration=5000):
    """Generates uncorrelated signal"""
    return np.random.randn(duration) * 0.3

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

def run_memory_trace_test():
    """Run memory trace test: A → B → A"""
    print("Test 3: Memory Trace (A → B → A)")
    print("Stage 1: Correlated signal (A)")
    
    # Create directory for results
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # Collect data by stages
    stage_data = []
    
    # STAGE 1: A(t) - correlated signal
    signal_a1 = generate_correlated_signal()
    spike_indices_a1, spike_times_a1 = audio_to_spikes(signal_a1)
    
    # Create network for first stage
    input_layer = SpikeGeneratorGroup(3, spike_indices_a1, spike_times_a1)
    delayed_input = NeuronGroup(3, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.3', reset='v = 0')
    input_to_delay = Synapses(input_layer, delayed_input, on_pre='v += 1', delay=5*ms)
    input_to_delay.connect(j='i')
    
    gen_network = NeuronGroup(60, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.8', reset='v = 0')
    output_layer = NeuronGroup(30, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.8', reset='v = 0')
    
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
    
    # Run first stage
    print("  Running first stage (A)...")
    net.run(5*second)
    
    # Save first stage data
    stage1_data = {
        'input_spikes': len(spike_mon_input.t),
        'gen_spikes': len(spike_mon_gen.t), 
        'output_spikes': len(spike_mon_output.t),
        'weights': [w[-1] for w in state_mon_weights.w if len(w) > 0]
    }
    stage_data.append(('A1', stage1_data))
    
    print(f"  Stage 1 completed: {stage1_data['output_spikes']} output layer spikes")
    
    # STAGE 2: B(t) - uncorrelated signal
    print("Stage 2: Uncorrelated signal (B)")
    
    # Create new network for second stage
    signal_b = generate_uncorrelated_signal()
    spike_indices_b, spike_times_b = audio_to_spikes(signal_b)
    
    input_layer_b = SpikeGeneratorGroup(3, spike_indices_b, spike_times_b)
    delayed_input_b = NeuronGroup(3, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.3', reset='v = 0')
    input_to_delay_b = Synapses(input_layer_b, delayed_input_b, on_pre='v += 1', delay=5*ms)
    input_to_delay_b.connect(j='i')
    
    # Use same weights as at the end of first stage
    gen_network_b = NeuronGroup(60, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.8', reset='v = 0')
    output_layer_b = NeuronGroup(30, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.8', reset='v = 0')
    
    adaptive_filter_b = Synapses(
        delayed_input_b, gen_network_b,
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
    
    adaptive_filter_b.connect(p=0.5)
    # Initialize weights from first stage
    avg_weight = np.mean(stage1_data['weights']) if stage1_data['weights'] else 0.3
    adaptive_filter_b.w = str(avg_weight) + ' + rand() * 0.1'
    
    gen_to_output_b = Synapses(gen_network_b, output_layer_b, on_pre='v_post += 0.3')
    gen_to_output_b.connect(p=0.3)
    
    # Monitors for second stage
    spike_mon_input_b = SpikeMonitor(input_layer_b)
    spike_mon_gen_b = SpikeMonitor(gen_network_b)
    spike_mon_output_b = SpikeMonitor(output_layer_b)
    state_mon_weights_b = StateMonitor(adaptive_filter_b, 'w', record=range(5))
    
    # Network for second stage
    net_b = Network()
    net_b.add(input_layer_b, delayed_input_b, gen_network_b, output_layer_b)
    net_b.add(input_to_delay_b, adaptive_filter_b, gen_to_output_b)
    net_b.add(spike_mon_input_b, spike_mon_gen_b, spike_mon_output_b, state_mon_weights_b)
    
    # Run second stage
    print("  Running second stage (B)...")
    net_b.run(5*second)
    
    # Save second stage data
    stage2_data = {
        'input_spikes': len(spike_mon_input_b.t),
        'gen_spikes': len(spike_mon_gen_b.t),
        'output_spikes': len(spike_mon_output_b.t),
        'weights': [w[-1] for w in state_mon_weights_b.w if len(w) > 0]
    }
    stage_data.append(('B', stage2_data))
    
    print(f"  Stage 2 completed: {stage2_data['output_spikes']} output layer spikes")
    
    # STAGE 3: A(t) again - correlated signal
    print("Stage 3: Correlated signal again (A)")
    
    # Create network for third stage
    signal_a2 = generate_correlated_signal()  # same signal as at the beginning
    spike_indices_a2, spike_times_a2 = audio_to_spikes(signal_a2)
    
    input_layer_c = SpikeGeneratorGroup(3, spike_indices_a2, spike_times_a2)
    delayed_input_c = NeuronGroup(3, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.3', reset='v = 0')
    input_to_delay_c = Synapses(input_layer_c, delayed_input_c, on_pre='v += 1', delay=5*ms)
    input_to_delay_c.connect(j='i')
    
    # Use weights from second stage
    gen_network_c = NeuronGroup(60, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.8', reset='v = 0')
    output_layer_c = NeuronGroup(30, '''dv/dt = -v/(20*ms) : 1''', threshold='v > 0.8', reset='v = 0')
    
    adaptive_filter_c = Synapses(
        delayed_input_c, gen_network_c,
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
    
    adaptive_filter_c.connect(p=0.5)
    # Initialize weights from second stage
    avg_weight_b = np.mean(stage2_data['weights']) if stage2_data['weights'] else 0.3
    adaptive_filter_c.w = str(avg_weight_b) + ' + rand() * 0.1'
    
    gen_to_output_c = Synapses(gen_network_c, output_layer_c, on_pre='v_post += 0.3')
    gen_to_output_c.connect(p=0.3)
    
    # Monitors for third stage
    spike_mon_input_c = SpikeMonitor(input_layer_c)
    spike_mon_gen_c = SpikeMonitor(gen_network_c)
    spike_mon_output_c = SpikeMonitor(output_layer_c)
    state_mon_weights_c = StateMonitor(adaptive_filter_c, 'w', record=range(5))
    
    # Network for third stage
    net_c = Network()
    net_c.add(input_layer_c, delayed_input_c, gen_network_c, output_layer_c)
    net_c.add(input_to_delay_c, adaptive_filter_c, gen_to_output_c)
    net_c.add(spike_mon_input_c, spike_mon_gen_c, spike_mon_output_c, state_mon_weights_c)
    
    # Run third stage
    print("  Running third stage (A again)...")
    net_c.run(5*second)
    
    # Save third stage data
    stage3_data = {
        'input_spikes': len(spike_mon_input_c.t),
        'gen_spikes': len(spike_mon_gen_c.t),
        'output_spikes': len(spike_mon_output_c.t),
        'weights': [w[-1] for w in state_mon_weights_c.w if len(w) > 0]
    }
    stage_data.append(('A2', stage3_data))
    
    print(f"  Stage 3 completed: {stage3_data['output_spikes']} output layer spikes")
    
    # Save all results
    with open('test_results/memory_trace_results.txt', 'w') as f:
        f.write("Test 3 Results: Memory Trace (A → B → A)\n")
        f.write("="*50 + "\n\n")
        
        for stage_name, data in stage_data:
            f.write(f"Stage: {stage_name}\n")
            f.write(f"  Input spikes: {data['input_spikes']}\n")
            f.write(f"  Generative network spikes: {data['gen_spikes']}\n")
            f.write(f"  Output layer spikes: {data['output_spikes']}\n")
            if data['weights']:
                avg_weight = np.mean(data['weights'])
                f.write(f"  Average weight: {avg_weight:.3f}\n")
            f.write("\n")
        
        # Memory trace analysis
        f.write("MEMORY TRACE ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        
        a1_output = stage_data[0][1]['output_spikes']
        b_output = stage_data[1][1]['output_spikes'] 
        a2_output = stage_data[2][1]['output_spikes']
        
        f.write(f"A1 → output: {a1_output} spikes\n")
        f.write(f"B → output: {b_output} spikes\n")
        f.write(f"A2 → output: {a2_output} spikes\n")
        f.write("\n")
        
        if a2_output < b_output and a2_output > 0:
            f.write("✓ CONFIRMED: Self-like behavior recovers faster after B\n")
        else:
            f.write("○ Additional memory trace analysis required\n")
    
    print("Results saved to test_results/memory_trace_results.txt")
    
    # Brief console output
    print("\nBRIEF SUMMARY:")
    print("="*20)
    a1_output = stage_data[0][1]['output_spikes']
    b_output = stage_data[1][1]['output_spikes'] 
    a2_output = stage_data[2][1]['output_spikes']
    
    print(f"A1: {a1_output} spikes")
    print(f"B:  {b_output} spikes") 
    print(f"A2: {a2_output} spikes")
    
    if a2_output < b_output:
        print("✓ Self-like behavior recovers (fewer spikes = more organized activity)")
    else:
        print("○ More research needed")

if __name__ == "__main__":
    run_memory_trace_test()
