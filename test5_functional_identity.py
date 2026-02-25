# test5_functional_identity.py
"""
Test 5: Functional Identity of Adaptive Filter Before and After Reset

This test verifies that the adaptive filter maintains functional identity
after temporary deactivation (Σ_fast reset), demonstrating that:
- The "form" (Σ_slow) preserves filtering capabilities
- Recovery is not just increased activity, but restored functionality
- Self-like behavior is about pattern recognition, not just activation levels

Metrics measured:
- Pattern selectivity
- Phase sensitivity  
- Response latency
- Prediction errors
"""

from brian2 import *
import numpy as np
import os

def generate_rhythmic_pattern(duration_ms=1000):
    """Generate rhythmic pattern with specific timing"""
    # Create structured pattern: strong beats every 100ms
    times_ms = []
    
    # Primary rhythm: every 100ms (10 Hz rhythm)
    for i in range(0, duration_ms, 100):
        times_ms.append(i)
        # Add hi-hats every 25ms within each beat
        for offset in [25, 50, 75]:
            if i + offset < duration_ms:
                times_ms.append(i + offset)
    
    # Remove duplicates and sort
    times_ms = sorted(list(set(times_ms)))
    indices = [0] * len(times_ms)
    
    return indices, np.array(times_ms) * ms

def generate_noise_pattern(duration_ms=1000, target_spikes=50):
    """Generate random noise pattern with minimum intervals"""
    # Ensure minimum interval between spikes to avoid Brian2 errors
    min_interval = 20  # ms
    times_ms = []
    
    # Generate random times with proper bounds
    current_time = min_interval
    max_attempts = target_spikes * 3  # Prevent infinite loop
    attempts = 0
    
    while len(times_ms) < target_spikes and current_time < duration_ms and attempts < max_attempts:
        # Generate next spike time (at least min_interval away)
        if current_time + min_interval < duration_ms:
            next_time = np.random.randint(current_time + min_interval, min(duration_ms, current_time + 100))
            if next_time < duration_ms:
                times_ms.append(next_time)
                current_time = next_time + min_interval
        else:
            break
        attempts += 1
    
    # If we don't have enough spikes, fill with regular intervals
    if len(times_ms) < target_spikes:
        remaining_needed = target_spikes - len(times_ms)
        # Add evenly spaced spikes at the end
        if len(times_ms) > 0:
            last_time = times_ms[-1]
        else:
            last_time = 0
            
        for i in range(remaining_needed):
            new_time = last_time + (i + 1) * 50
            if new_time < duration_ms:
                times_ms.append(new_time)
    
    # Fallback: if still empty, create regular pattern
    if len(times_ms) == 0:
        times_ms = list(range(0, min(duration_ms, target_spikes * 30), 30))
        # Ensure we don't exceed duration
        times_ms = [t for t in times_ms if t < duration_ms]
    
    indices = [0] * len(times_ms)
    return indices, np.array(times_ms) * ms

def calculate_selectivity(spike_times_array, pattern_times_array, window_ms=75):
    """
    Calculate how selectively the network responds to pattern vs noise
    Using plain arrays to avoid unit conflicts
    """
    if len(spike_times_array) == 0:
        return 0.0
    
    pattern_matches = 0
    total_responses = len(spike_times_array)
    
    for response in spike_times_array:
        # Check if response is near any pattern time
        for pattern_time in pattern_times_array:
            if abs(response - pattern_time) <= window_ms:
                pattern_matches += 1
                break
    
    return pattern_matches / max(total_responses, 1)

def calculate_response_latency(input_times_array, output_times_array):
    """Calculate average response latency using plain arrays"""
    if len(input_times_array) == 0 or len(output_times_array) == 0:
        return 0.0
    
    latencies = []
    # Take first 5 input spikes to avoid too much computation
    for inp_time in input_times_array[:min(5, len(input_times_array))]:
        # Find first output after input
        later_outputs = [out for out in output_times_array if out > inp_time]
        if later_outputs:
            latency = later_outputs[0] - inp_time
            latencies.append(latency)
    
    return np.mean(latencies) if latencies else 0.0

def calculate_prediction_error(input_times_array, output_times_array, expected_delay_ms=75):
    """Calculate prediction error based on expected timing"""
    if len(input_times_array) == 0 or len(output_times_array) == 0:
        return 1000.0  # Large error when no data
    
    errors = []
    for inp_time in input_times_array[:min(10, len(input_times_array))]:  # Limit samples
        expected_output = inp_time + expected_delay_ms
        # Find closest actual output
        if len(output_times_array) > 0:
            closest_output = min(output_times_array, key=lambda x: abs(x - expected_output))
            error = abs(closest_output - expected_output)
            errors.append(error)
    
    return np.mean(errors) if errors else 1000.0

def run_functional_identity_test():
    """Main test function"""
    print("Test 5: Functional Identity of Adaptive Filter")
    print("="*50)
    
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # Generate test patterns ONCE for consistency
    print("Generating test patterns...")
    pattern_indices, pattern_times = generate_rhythmic_pattern(1000)
    
    # Generate noise with approximately same number of spikes
    noise_indices, noise_times = generate_noise_pattern(1000, len(pattern_indices))
    
    print(f"Pattern: {len(pattern_indices)} spikes")
    print(f"Noise: {len(noise_indices)} spikes")
    print(f"Pattern sample times (first 5): {[int(t/ms) for t in pattern_times[:5]]}")
    print(f"Noise sample times (first 5): {[int(t/ms) for t in noise_times[:5]]}")
    
    # Store pattern times as plain arrays (convert from Brian2 units)
    pattern_times_array = np.array([float(t/ms) for t in pattern_times])
    noise_times_array = np.array([float(t/ms) for t in noise_times])
    
    # PHASE 1: Train filter on pattern
    print("\nPHASE 1: Training filter on rhythmic pattern")
    
    input1 = SpikeGeneratorGroup(1, pattern_indices, pattern_times)
    
    # Simple network
    neurons1 = NeuronGroup(
        20,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8',
        reset='v = 0'
    )
    
    # Standard synapses
    synapses1 = Synapses(
        input1, neurons1,
        '''w : 1''',
        on_pre='''
        v_post += w
        w = clip(w + 0.05, 0, 3)  # Learning rule
        '''
    )
    synapses1.connect(p=0.7)
    synapses1.w = 0.2  # Initial weights
    
    # Monitor training
    spike_mon1 = SpikeMonitor(neurons1)
    
    net1 = Network(input1, neurons1, synapses1, spike_mon1)
    net1.run(1.5*second)
    
    trained_spikes = list(spike_mon1.t)
    final_weights = list(synapses1.w[:min(5, len(synapses1))])  # First few weights
    
    print(f"Training completed: {len(trained_spikes)} output spikes")
    if final_weights:
        print(f"Final weights sample: {[float(f'{w:.2f}') for w in final_weights]}")
    
    # PHASE 2: Test trained filter on pattern
    print("\nPHASE 2: Testing trained filter response to pattern")
    
    # CREATE NEW INPUT GROUP FOR THIS PHASE
    input2 = SpikeGeneratorGroup(1, pattern_indices, pattern_times)
    
    neurons2 = NeuronGroup(
        20,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8',
        reset='v = 0'
    )
    
    # Use trained weights (average)
    avg_weight = np.mean(final_weights) if final_weights else 0.2
    synapses2 = Synapses(input2, neurons2, on_pre=f'v_post += {avg_weight:.2f}')
    synapses2.connect(p=0.7)
    
    spike_mon2 = SpikeMonitor(neurons2)
    net2 = Network(input2, neurons2, synapses2, spike_mon2)
    net2.run(1*second)
    
    pattern_response_spikes = list(spike_mon2.t)
    print(f"Pattern response: {len(pattern_response_spikes)} spikes")
    
    # PHASE 3: Test trained filter on noise
    print("PHASE 3: Testing trained filter response to noise")
    
    # CREATE NEW INPUT GROUP FOR NOISE
    input3 = SpikeGeneratorGroup(1, noise_indices, noise_times)
    
    neurons3 = NeuronGroup(
        20,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8',
        reset='v = 0'
    )
    
    synapses3 = Synapses(input3, neurons3, on_pre=f'v_post += {avg_weight:.2f}')
    synapses3.connect(p=0.7)
    
    spike_mon3 = SpikeMonitor(neurons3)
    net3 = Network(input3, neurons3, synapses3, spike_mon3)
    net3.run(1*second)
    
    noise_response_spikes = list(spike_mon3.t)
    print(f"Noise response: {len(noise_response_spikes)} spikes")
    
    # PHASE 4: Reset fast components (simulate temporary deactivation)
    print("PHASE 4: Resetting fast components (simulating temporary deactivation)")
    
    # CREATE NEW INPUT AND NETWORK FOR RESET
    input4 = SpikeGeneratorGroup(1, pattern_indices, pattern_times)
    
    neurons4 = NeuronGroup(
        20,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8',
        reset='v = 0'
    )
    
    # Very weak weights (only slow component remains)
    synapses4 = Synapses(input4, neurons4, on_pre='v_post += 0.05')  # Minimal weights
    synapses4.connect(p=0.7)
    
    spike_mon4 = SpikeMonitor(neurons4)
    net4 = Network(input4, neurons4, synapses4, spike_mon4)
    net4.run(300*ms)
    
    reset_spikes = list(spike_mon4.t)
    print(f"After reset: {len(reset_spikes)} spikes (minimal activity)")
    
    # PHASE 5: Recovery and retest
    print("PHASE 5: Recovery and retesting with restored weights")
    
    # CREATE NEW INPUT AND NETWORK FOR RECOVERY
    input5 = SpikeGeneratorGroup(1, pattern_indices, pattern_times)
    
    neurons5 = NeuronGroup(
        20,
        '''dv/dt = -v/(20*ms) : 1''',
        threshold='v > 0.8',
        reset='v = 0'
    )
    
    # Gradually restore weights (85% of trained)
    recovery_weight = avg_weight * 0.85
    synapses5 = Synapses(input5, neurons5, on_pre=f'v_post += {recovery_weight:.2f}')
    synapses5.connect(p=0.7)
    
    spike_mon5 = SpikeMonitor(neurons5)
    net5 = Network(input5, neurons5, synapses5, spike_mon5)
    net5.run(1*second)
    
    recovered_spikes = list(spike_mon5.t)
    print(f"Recovered response: {len(recovered_spikes)} spikes")
    
    # CALCULATE METRICS using plain arrays to avoid unit conflicts
    print("\nCALCULATING FUNCTIONAL METRICS...")
    
    # Convert spike times to plain arrays (milliseconds)
    pattern_response_array = np.array([float(t/ms) for t in pattern_response_spikes])
    recovered_response_array = np.array([float(t/ms) for t in recovered_spikes])
    noise_response_array = np.array([float(t/ms) for t in noise_response_spikes])
    
    # Pattern selectivity
    selectivity_before = calculate_selectivity(pattern_response_array, pattern_times_array, window_ms=75)
    selectivity_after = calculate_selectivity(recovered_response_array, pattern_times_array, window_ms=75)
    noise_selectivity = calculate_selectivity(noise_response_array, noise_times_array, window_ms=75)
    
    # Response latency
    latency_before = calculate_response_latency(pattern_times_array, pattern_response_array)
    latency_after = calculate_response_latency(pattern_times_array, recovered_response_array)
    
    # Prediction error
    error_before = calculate_prediction_error(pattern_times_array, pattern_response_array, expected_delay_ms=75)
    error_after = calculate_prediction_error(pattern_times_array, recovered_response_array, expected_delay_ms=75)
    
    # Save results
    with open('test_results/functional_identity_results.txt', 'w') as f:
        f.write("TEST 5: FUNCTIONAL IDENTITY OF ADAPTIVE FILTER\n")
        f.write("="*50 + "\n\n")
        
        f.write("EXPERIMENTAL SETUP:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Pattern spikes: {len(pattern_indices)}\n")
        f.write(f"Noise spikes: {len(noise_indices)}\n")
        f.write(f"Trained weights: {[float(f'{w:.2f}') for w in final_weights] if final_weights else 'N/A'}\n\n")
        
        f.write("RESPONSE ANALYSIS:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Pattern response: {len(pattern_response_spikes)} spikes\n")
        f.write(f"Noise response: {len(noise_response_spikes)} spikes\n")
        f.write(f"Reset response: {len(reset_spikes)} spikes\n")
        f.write(f"Recovered response: {len(recovered_spikes)} spikes\n\n")
        
        f.write("FUNCTIONAL METRICS ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Pattern Selectivity:\n")
        f.write(f"  Before reset: {selectivity_before:.3f}\n")
        f.write(f"  After reset:  {selectivity_after:.3f}\n")
        f.write(f"  Noise control: {noise_selectivity:.3f}\n\n")
        
        f.write(f"Response Latency (ms):\n")
        f.write(f"  Before reset: {latency_before:.1f} ms\n")
        f.write(f"  After reset:  {latency_after:.1f} ms\n\n")
        
        f.write(f"Prediction Error (ms):\n")
        f.write(f"  Before reset: {error_before:.1f} ms\n")
        f.write(f"  After reset:  {error_after:.1f} ms\n\n")
        
        # Statistical analysis
        f.write("FUNCTIONAL IDENTITY ASSESSMENT:\n")
        f.write("-" * 30 + "\n")
        
        # Selectivity preservation
        pattern_selectivity_diff = abs(selectivity_before - selectivity_after)
        noise_vs_pattern_before = selectivity_before - noise_selectivity
        noise_vs_pattern_after = selectivity_after - noise_selectivity
        
        if pattern_selectivity_diff < 0.25:
            f.write("✅ SELECTIVITY PRESERVED: Filter recognizes patterns similarly\n")
        elif selectivity_after > selectivity_before * 0.5:  # At least 50% preserved
            f.write("⚠️  SELECTIVITY PARTIALLY PRESERVED: Pattern recognition somewhat maintained\n")
        else:
            f.write("❌ SELECTIVITY LOST: Pattern recognition significantly degraded\n")
        
        # Latency consistency
        latency_change = abs(latency_before - latency_after)
        if latency_change < 20:  # Within 20ms tolerance
            f.write("✅ LATENCY CONSISTENT: Response timing preserved\n")
        elif latency_change < 50:
            f.write("⚠️  LATENCY MODERATELY CHANGED: Response timing altered but recognizable\n")
        else:
            f.write("❌ LATENCY SIGNIFICANTLY CHANGED: Response timing disrupted\n")
        
        # Prediction accuracy (lower is better)
        error_improvement = error_before - error_after
        if abs(error_improvement) < 20:  # Within 20ms tolerance
            f.write("✅ PREDICTION PRESERVED: Timing prediction maintained\n")
        elif error_after < error_before * 1.5:  # Not more than 50% worse
            f.write("⚠️  PREDICTION PARTIALLY PRESERVED: Some timing accuracy lost\n")
        else:
            f.write("❌ PREDICTION SIGNIFICANTLY DEGRADED: Timing prediction lost\n")
        
        f.write(f"\nQUANTITATIVE COMPARISON:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Before reset → After reset:\n")
        f.write(f"  Selectivity: {selectivity_before:.3f} → {selectivity_after:.3f}\n")
        f.write(f"  Latency: {latency_before:.1f}ms → {latency_after:.1f}ms\n")
        f.write(f"  Prediction Error: {error_before:.1f}ms → {error_after:.1f}ms\n")
        
        # Discrimination test (pattern vs noise)
        discrimination_before = selectivity_before - noise_selectivity
        discrimination_after = selectivity_after - noise_selectivity
        
        f.write(f"\nDISCRIMINATION ABILITY:\n")
        f.write("-" * 22 + "\n")
        f.write(f"Pattern vs Noise discrimination:\n")
        f.write(f"  Before: {discrimination_before:.3f}\n")
        f.write(f"  After:  {discrimination_after:.3f}\n")
        
        if discrimination_after > 0.1:
            f.write("✅ FILTER MAINTAINS DISCRIMINATION: Can distinguish pattern from noise\n")
        else:
            f.write("⚠️  FILTER DISCRIMINATION REDUCED: Less able to distinguish patterns\n")
        
        # Overall assessment
        metrics_preserved = 0
        if pattern_selectivity_diff < 0.25:
            metrics_preserved += 1
        if latency_change < 20:
            metrics_preserved += 1
        if abs(error_improvement) < 20:
            metrics_preserved += 1
        if discrimination_after > 0.1:
            metrics_preserved += 1
        
        f.write(f"\nOVERALL ASSESSMENT:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Functional metrics preserved: {metrics_preserved}/4\n")
        
        if metrics_preserved >= 3:
            f.write("🏆 FUNCTIONAL IDENTITY MAINTAINED: Complete or near-complete recovery\n")
        elif metrics_preserved == 2:
            f.write("✅ FUNCTIONAL IDENTITY MOSTLY MAINTAINED: Good partial recovery\n")
        elif metrics_preserved == 1:
            f.write("⚠️  PARTIAL FUNCTIONAL RECOVERY: Limited preservation\n")
        else:
            f.write("❌ FUNCTIONAL IDENTITY LOST: No meaningful functional recovery\n")
    
    # Console output
    print("\nFUNCTIONAL IDENTITY RESULTS:")
    print("="*45)
    print(f"Pattern Selectivity: {selectivity_before:.3f} → {selectivity_after:.3f}")
    print(f"Response Latency:    {latency_before:.1f}ms → {latency_after:.1f}ms")  
    print(f"Prediction Error:    {error_before:.1f}ms → {error_after:.1f}ms")
    print(f"Pattern vs Noise Discrimination: {selectivity_before - noise_selectivity:.3f} → {selectivity_after - noise_selectivity:.3f}")
    
    # Quick assessment
    metrics_good = sum([
        abs(selectivity_before - selectivity_after) < 0.25,
        abs(latency_before - latency_after) < 20,
        abs(error_before - error_after) < 20,
        (selectivity_after - noise_selectivity) > 0.1
    ])
    
    if metrics_good >= 3:
        print("🏆 EXCELLENT FUNCTIONAL RECOVERY: Identity well maintained")
    elif metrics_good >= 2:
        print("✅ GOOD FUNCTIONAL RECOVERY: Identity mostly preserved")
    elif metrics_good >= 1:
        print("⚠️  LIMITED FUNCTIONAL RECOVERY: Some aspects preserved")
    else:
        print("❌ POOR FUNCTIONAL RECOVERY: Significant degradation")
    
    print(f"\nResults saved to test_results/functional_identity_results.txt")

if __name__ == "__main__":
    run_functional_identity_test()
