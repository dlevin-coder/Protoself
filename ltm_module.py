# ltm_module.py
from brian2 import *

def create_ltm_network(n_neurons=64):
    """
    Медленно изменяемые синапсы с STDP
    """
    ltm_eqs = '''
    dv/dt = (-v + I_input)/tau : 1
    dI_input/dt = -I_input/tau_input : 1
    tau : second
    tau_input : second
    '''
    
    ltm_neurons = NeuronGroup(n_neurons, ltm_eqs, threshold='v>1', 
                             reset='v=0', method='euler')
    
    # Параметры
    ltm_neurons.tau = 100*ms  # Медленные
    ltm_neurons.tau_input = 50*ms
    ltm_neurons.v = 'rand() * 0.3'
    
    # Простые фиксированные связи (STDP добавим позже)
    ltm_synapses = Synapses(ltm_neurons, ltm_neurons, 'w : 1', on_pre='v_post += w')
    ltm_synapses.connect(condition='i!=j', p=0.2)
    ltm_synapses.w = '0.1*rand()'
    
    return ltm_neurons, ltm_synapses
