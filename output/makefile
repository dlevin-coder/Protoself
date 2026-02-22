PROGRAM = main

SRCS = brianlib/spikequeue.cpp code_objects/after_run_neurongroup_spike_thresholder_codeobject.cpp code_objects/before_run_synapses_post_push_spikes.cpp code_objects/before_run_synapses_pre_push_spikes.cpp code_objects/neurongroup_spike_resetter_codeobject.cpp code_objects/neurongroup_spike_thresholder_codeobject.cpp code_objects/neurongroup_stateupdater_codeobject.cpp code_objects/spikegeneratorgroup_codeobject.cpp code_objects/spikemonitor_codeobject.cpp code_objects/statemonitor_codeobject.cpp code_objects/synapses_group_variable_set_conditional_codeobject.cpp code_objects/synapses_post_codeobject.cpp code_objects/synapses_post_push_spikes.cpp code_objects/synapses_pre_codeobject.cpp code_objects/synapses_pre_push_spikes.cpp code_objects/synapses_synapses_create_generator_codeobject.cpp main.cpp network.cpp objects.cpp run.cpp synapses_classes.cpp
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/dynamic_array.h brianlib/spikequeue.h code_objects/after_run_neurongroup_spike_thresholder_codeobject.h code_objects/before_run_synapses_post_push_spikes.h code_objects/before_run_synapses_pre_push_spikes.h code_objects/neurongroup_spike_resetter_codeobject.h code_objects/neurongroup_spike_thresholder_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/spikegeneratorgroup_codeobject.h code_objects/spikemonitor_codeobject.h code_objects/statemonitor_codeobject.h code_objects/synapses_group_variable_set_conditional_codeobject.h code_objects/synapses_post_codeobject.h code_objects/synapses_post_push_spikes.h code_objects/synapses_pre_codeobject.h code_objects/synapses_pre_push_spikes.h code_objects/synapses_synapses_create_generator_codeobject.h network.h objects.h run.h synapses_classes.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = -I"/home/dlevin/brian2/brian2_env/include" -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -L"/home/dlevin/brian2/brian2_env/lib" -Wl,--enable-new-dtags,-rpath,"/home/dlevin/brian2/brian2_env/lib" 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CXX) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > make.deps

ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@