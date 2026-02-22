# observer_module.py
from brian2 import *

def create_observer_network(n_input_channels=16, n_hidden=32):
    """
    Рекуррентная SNN для интеграции во времени
    """
    # Hidden recurrent layer
    eqs_neuron = '''
    dv/dt = (-v + I_input + I_rec)/tau : 1
    dI_input/dt = -I_input/tau_input : 1
    dI_rec/dt = -I_rec/tau_rec : 1
    tau : second
    tau_input : second
    tau_rec : second
    '''
    
    try:
        hidden = NeuronGroup(n_hidden, eqs_neuron, threshold='v>1', reset='v=0', 
                            method='euler')
        
        # Параметры
        hidden.tau = 20*ms
        hidden.tau_input = 5*ms
        hidden.tau_rec = 50*ms
        hidden.v = 'rand() * 0.1'
        
        # Рекуррентные связи
        rec_syn = Synapses(hidden, hidden, 'w : 1', on_pre='I_rec += w')
        rec_syn.connect(condition='i!=j', p=0.3)  # Разреженные связи без самосвязей
        rec_syn.w = '0.2*rand()'  # Слабые рекуррентные веса
        
        # Входные синапсы (будут подключены позже)
        input_syn = Synapses(None, hidden, 'w : 1', on_pre='I_input += w')
        
        print(f"Создан наблюдатель: {n_hidden} нейронов")
        return hidden, input_syn, rec_syn
        
    except Exception as e:
        print(f"Ошибка создания наблюдателя: {e}")
        # Создаем минимальный рабочий вариант
        simple_eqs = '''
        dv/dt = (-v + I_ext)/tau : 1
        dI_ext/dt = -I_ext/tau_input : 1
        tau : second
        '''
        
        simple_hidden = NeuronGroup(max(1, n_hidden//2), simple_eqs, 
                                   threshold='v>1', reset='v=0', method='euler')
        simple_hidden.tau = 20*ms
        simple_hidden.v = 'rand() * 0.1'
        
        dummy_syn = Synapses(simple_hidden, simple_hidden, 'w : 1', on_pre='v_post += w')
        dummy_input_syn = Synapses(None, simple_hidden, 'w : 1', on_pre='v_post += w')
        
        return simple_hidden, dummy_input_syn, dummy_syn
