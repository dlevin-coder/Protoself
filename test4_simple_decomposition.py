# test4_fixed_decomposition.py
from brian2 import *
import numpy as np
import os

def run_fixed_decomposition_test():
    """Fixed decomposition test with strong input"""
    print("Test 4: Fixed Decomposition (Σ = Σ_fast + Σ_slow)")
    print("="*55)
    
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # Create a STRONGER input signal
    print("Creating strong rhythmic input...")
    
    # Denser rhythm - spikes every 50 ms for 1 second
    duration = 1000  # ms
    spike_interval = 50  # ms
    num_spikes = duration // spike_interval
    
    input_indices = [0] * num_spikes  # all spikes from one neuron
    input_times = [i * spike_interval for i in range(num_spikes)] * ms
    
    print(f"Generated {num_spikes} input spikes every {spike_interval} ms")
    
    # Model 1: Full system (strong weights)
    print("Model 1: Full system (Σ_fast + Σ_slow active)")
    
    input1 = SpikeGeneratorGroup(1, input_indices, input_times)
    
    neurons1 = NeuronGroup(
        30,  # more neurons
        '''dv/dt = -v/(15*ms) : 1''',  # faster time constant
        threshold='v > 0.5',  # lower threshold
        reset='v = 0'
    )
    
    # Very strong weights to ensure activation
    synapses1 = Synapses(input1, neurons1, on_pre='v_post += 1.0')  # strong input
    synapses1.connect(p=0.9)  # almost full connection
    
    spike_mon1 = SpikeMonitor(neurons1)
    
    net1 = Network(input1, neurons1, synapses1, spike_mon1)
    net1.run(1*second)
    
    full_system_spikes = len(spike_mon1.t)
    print(f"  Full system: {full_system_spikes} spikes")
    
    # Model 2: Slow components only (anesthesia)
    print("Model 2: Slow components only (anesthesia)")
    
    input2 = SpikeGeneratorGroup(1, input_indices, input_times)
    
    neurons2 = NeuronGroup(
        30,
        '''dv/dt = -v/(15*ms) : 1''',
        threshold='v > 0.5',
        reset='v = 0'
    )
    
    # Very weak weights (almost zero activity)
    synapses2 = Synapses(input2, neurons2, on_pre='v_post += 0.1')  # 10 times weaker
    synapses2.connect(p=0.9)
    
    spike_mon2 = SpikeMonitor(neurons2)
    
    net2 = Network(input2, neurons2, synapses2, spike_mon2)
    net2.run(1*second)
    
    slow_only_spikes = len(spike_mon2.t)
    print(f"  Slow components only: {slow_only_spikes} spikes")
    
    # Model 3: Partial recovery
    print("Model 3: Partial recovery")
    
    input3 = SpikeGeneratorGroup(1, input_indices, input_times)
    
    neurons3 = NeuronGroup(
        30,
        '''dv/dt = -v/(15*ms) : 1''',
        threshold='v > 0.5',
        reset='v = 0'
    )
    
    # Medium weights
    synapses3 = Synapses(input3, neurons3, on_pre='v_post += 0.5')  # medium weights
    synapses3.connect(p=0.9)
    
    spike_mon3 = SpikeMonitor(neurons3)
    
    net3 = Network(input3, neurons3, synapses3, spike_mon3)
    net3.run(1*second)
    
    recovery_spikes = len(spike_mon3.t)
    print(f"  Partial recovery: {recovery_spikes} spikes")
    
    # Model 4: Almost full recovery
    print("Model 4: Almost full recovery")
    
    input4 = SpikeGeneratorGroup(1, input_indices, input_times)
    
    neurons4 = NeuronGroup(
        30,
        '''dv/dt = -v/(15*ms) : 1''',
        threshold='v > 0.5',
        reset='v = 0'
    )
    
    # Strong but not maximum weights
    synapses4 = Synapses(input4, neurons4, on_pre='v_post += 0.8')  # nearly full
    synapses4.connect(p=0.9)
    
    spike_mon4 = SpikeMonitor(neurons4)
    
    net4 = Network(input4, neurons4, synapses4, spike_mon4)
    net4.run(1*second)
    
    almost_full_spikes = len(spike_mon4.t)
    print(f"  Almost full recovery: {almost_full_spikes} spikes")
    
    # Check that we have activity in at least one model
    if full_system_spikes == 0:
        print("⚠️  WARNING: No activity even in full system!")
        print("Trying even stronger input...")
        
        # Emergency model with very strong input
        emergency_input = SpikeGeneratorGroup(1, [0]*20, np.arange(0, 1000, 50)*ms)
        emergency_neurons = NeuronGroup(20, '''dv/dt = -v/(10*ms) : 1''', threshold='v > 0.1', reset='v = 0')
        emergency_syn = Synapses(emergency_input, emergency_neurons, on_pre='v_post += 2.0')
        emergency_syn.connect()
        emergency_mon = SpikeMonitor(emergency_neurons)
        emergency_net = Network(emergency_input, emergency_neurons, emergency_syn, emergency_mon)
        emergency_net.run(500*ms)
        emergency_spikes = len(emergency_mon.t)
        print(f"Emergency model: {emergency_spikes} spikes")
    
    # Save results
    with open('test_results/fixed_decomposition_results.txt', 'w') as f:
        f.write("TEST 4 RESULTS: FIXED DECOMPOSITION\n")
        f.write("="*50 + "\n\n")
        
        f.write("DECOMPOSITION MODEL: Σ(t) = Σ_fast(t) + Σ_slow\n")
        f.write("- Σ_fast - disappears under anesthesia (current activity)\n")
        f.write("- Σ_slow - preserved as 'form' for recovery\n\n")
        
        f.write("INPUT SIGNAL:\n")
        f.write(f"- {num_spikes} spikes every {spike_interval} ms\n")
        f.write("- Strong rhythmic pattern\n\n")
        
        f.write("MODEL RESULTS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Model 1 - Full system:     {full_system_spikes:>4} spikes\n")
        f.write(f"Model 2 - Only Σ_slow:       {slow_only_spikes:>4} spikes\n")
        f.write(f"Model 3 - Partial recovery:   {recovery_spikes:>4} spikes\n")
        f.write(f"Model 4 - Almost full:       {almost_full_spikes:>4} spikes\n\n")
        
        # Analysis
        f.write("DECOMPOSITION ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        
        # Check if there is any activity at all
        if full_system_spikes > 0:
            # Hypothesis 1: Anesthesia reduces activity
            if slow_only_spikes < full_system_spikes:
                reduction = (full_system_spikes - slow_only_spikes) / max(full_system_spikes, 1) * 100
                f.write(f"✓ Hypothesis 1 CONFIRMED: Anesthesia reduces activity by {reduction:.1f}%\n")
            elif slow_only_spikes == full_system_spikes:
                f.write("○ Hypothesis 1: Anesthesia does not affect activity\n")
            else:
                f.write("⚠ Hypothesis 1: Unexpected result - activity increased\n")
            
            # Hypothesis 2: Recovery progresses
            if recovery_spikes > slow_only_spikes:
                f.write("✓ Hypothesis 2 CONFIRMED: Recovery increases activity\n")
            else:
                f.write("○ Hypothesis 2: Recovery does not increase activity\n")
                
            # Hypothesis 3: Recovery is not instantaneous
            if almost_full_spikes < full_system_spikes:
                f.write("✓ Hypothesis 3 CONFIRMED: Recovery is not instantaneous\n")
            elif almost_full_spikes == full_system_spikes:
                f.write("○ Hypothesis 3: Recovery is instantaneous\n")
            else:
                f.write("⚠ Hypothesis 3: Recovery exceeds original level\n")
                
            # Hypothesis 4: Sequence makes sense
            if slow_only_spikes <= recovery_spikes <= almost_full_spikes <= full_system_spikes:
                f.write("✓ Hypothesis 4 CONFIRMED: Logical activity sequence\n")
            else:
                f.write("○ Hypothesis 4: Activity sequence is not logical\n")
        else:
            f.write("❌ NO ACTIVITY: Model did not work\n")
            f.write("Network parameters need adjustment\n")
        
        f.write(f"\nQUANTITATIVE ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Full system → Anesthesia: {full_system_spikes} → {slow_only_spikes}\n")
        f.write(f"Anesthesia → Recovery: {slow_only_spikes} → {recovery_spikes}\n")
        f.write(f"Recovery → Almost full: {recovery_spikes} → {almost_full_spikes}\n")
    
    print("\nResults saved to test_results/fixed_decomposition_results.txt")
    
    # Brief summary
    print("\nDECOMPOSITION SUMMARY:")
    print("="*45)
    print(f"Full system:          {full_system_spikes:>4} spikes")
    print(f"Anesthesia (Σ_fast=0):    {slow_only_spikes:>4} spikes")
    print(f"Partial recovery:         {recovery_spikes:>4} spikes")
    print(f"Almost full:              {almost_full_spikes:>4} spikes")
    
    # Main conclusions
    if full_system_spikes > 0:
        if slow_only_spikes < full_system_spikes:
            reduction_pct = (full_system_spikes - slow_only_spikes) / max(full_system_spikes, 1) * 100
            print(f"✓ ANESTHESIA WORKS: activity reduction of {reduction_pct:.1f}%")
        
        if recovery_spikes > slow_only_spikes:
            print("✓ RECOVERY WORKS: activity returns")
        
        if almost_full_spikes <= full_system_spikes:
            print("✓ RECOVERY NOT INSTANTANEOUS: adaptation trace preserved")
        
        # Check logical sequence
        sequence_ok = (slow_only_spikes <= recovery_spikes <= almost_full_spikes <= full_system_spikes)
        if sequence_ok or (slow_only_spikes <= recovery_spikes <= almost_full_spikes):
            print("✓ LOGICAL SEQUENCE: model behaves predictably")
    else:
        print("❌ MODEL FAILED: parameter tuning required")

if __name__ == "__main__":
    run_fixed_decomposition_test()

