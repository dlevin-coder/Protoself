# predictive_module.py
from brian2 import *

def create_predictive_network(observer_neurons, delay=50*ms):
    """
    Копия наблюдателя с задержанными обратными связями
    """
    # Создаем копию наблюдателя с задержкой
    predictor_eqs = '''
    dv/dt = (-v + I_pred)/tau : 1
    dI_pred/dt = -I_pred/tau_input : 1
    tau : second
    tau_input : second
    '''
    
    # Популяция предикторов
    n_neurons = len(observer_neurons) if hasattr(observer_neurons, '__len__') else observer_neurons.N
    predictor = NeuronGroup(n_neurons, predictor_eqs, 
                           threshold='v>0.8', reset='v=0', method='euler')
    
    # Параметры
    predictor.tau = 25*ms
    predictor.tau_input = 10*ms
    predictor.v = 'rand() * 0.2'
    
    return predictor

def create_error_detection(observer, predictor):
    """
    Механизм детекции ошибки между наблюдателем и предиктором
    """
    # Создаем нейроны для вычисления ошибки
    error_eqs = '''
    dv/dt = (-v + error_signal)/tau : 1
    error_signal : 1
    tau : second
    '''
    
    error_neurons = NeuronGroup(len(observer), error_eqs, 
                               threshold='v>0.5', reset='v=0', method='euler')
    error_neurons.tau = 15*ms
    
    return error_neurons
