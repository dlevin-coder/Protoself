# self_object_module.py
from brian2 import *

def create_summary_state(n_summary=8):
    """
    Компактное представление состояния системы
    """
    summary_eqs = '''
    dv/dt = (-v + I_obs + I_ltm)/tau : 1
    dI_obs/dt = -I_obs/tau_obs : 1
    dI_ltm/dt = -I_ltm/tau_ltm : 1  
    tau : second
    tau_obs : second
    tau_ltm : second
    '''
    
    summary_neurons = NeuronGroup(n_summary, summary_eqs, 
                                 threshold='v>1', reset='v=0', method='euler')
    
    # Очень медленная динамика
    summary_neurons.tau = 200*ms
    summary_neurons.tau_obs = 150*ms
    summary_neurons.tau_ltm = 300*ms
    summary_neurons.v = 'rand() * 0.2'
    
    return summary_neurons

def connect_summary_to_observer(summary, observer):
    """
    Summary state подается обратно в наблюдатель
    """
    summary_to_obs = Synapses(summary, observer, 
                             'w : 1', on_pre='I_input += w')
    summary_to_obs.connect(p=0.3)
    summary_to_obs.w = '0.2*rand()'
    
    return summary_to_obs
