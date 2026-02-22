#include <stdlib.h>
#include "objects.h"
#include <csignal>
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"

#include "code_objects/neurongroup_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_post_codeobject.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "code_objects/before_run_synapses_post_push_spikes.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>




void set_from_command_line(const std::vector<std::string> args)
{
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}

void _int_handler(int signal_num) {
	if (Network::_globally_running && !Network::_globally_stopped) {
		Network::_globally_stopped = true;
	} else {
		std::signal(signal_num, SIG_DFL);
		std::raise(signal_num);
	}
}

int main(int argc, char **argv)
{
	std::signal(SIGINT, _int_handler);
	std::random_device _rd;
	std::vector<std::string> args(argv + 1, argv + argc);
	if (args.size() >=2 && args[0] == "--results_dir")
	{
		brian::results_dir = args[1];
		#ifdef DEBUG
		std::cout << "Setting results dir to '" << brian::results_dir << "'" << std::endl;
		#endif
		args.erase(args.begin(), args.begin()+2);
	}
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _dynamic_array_spikegeneratorgroup_spike_number.resize(9);
        
                        
                        for(int i=0; i<_dynamic_array_spikegeneratorgroup_spike_number.size(); i++)
                        {
                            _dynamic_array_spikegeneratorgroup_spike_number[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_number[i];
                        }
                        
        _dynamic_array_spikegeneratorgroup_neuron_index.resize(9);
        
                        
                        for(int i=0; i<_dynamic_array_spikegeneratorgroup_neuron_index.size(); i++)
                        {
                            _dynamic_array_spikegeneratorgroup_neuron_index[i] = _static_array__dynamic_array_spikegeneratorgroup_neuron_index[i];
                        }
                        
        _dynamic_array_spikegeneratorgroup_spike_time.resize(9);
        
                        
                        for(int i=0; i<_dynamic_array_spikegeneratorgroup_spike_time.size(); i++)
                        {
                            _dynamic_array_spikegeneratorgroup_spike_time[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_time[i];
                        }
                        
        _dynamic_array_spikegeneratorgroup__timebins.resize(9);
        _array_spikegeneratorgroup__lastindex[0] = 0;
        _array_spikegeneratorgroup_period[0] = 0.0;
        _run_synapses_synapses_create_generator_codeobject();
        _run_synapses_group_variable_set_conditional_codeobject();
        
                        
                        for(int i=0; i<_num__array_statemonitor__indices; i++)
                        {
                            _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _array_spikegeneratorgroup__lastindex[0] = 0;
        
                        
                        for(int i=0; i<_dynamic_array_spikegeneratorgroup__timebins.size(); i++)
                        {
                            _dynamic_array_spikegeneratorgroup__timebins[i] = _static_array__dynamic_array_spikegeneratorgroup__timebins[i];
                        }
                        
        _array_spikegeneratorgroup__period_bins[0] = 0.0;
        _before_run_synapses_pre_push_spikes();
        _before_run_synapses_post_push_spikes();
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikegeneratorgroup_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_post_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_statemonitor_codeobject);
        set_from_command_line(args);
        magicnetwork.run(10.0, NULL, 10.0);
        _after_run_neurongroup_spike_thresholder_codeobject();
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_post_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}