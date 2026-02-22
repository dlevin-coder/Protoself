

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<chrono>
#include<random>
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg

// For multhreading, we need one generator for each thread.
std::vector< RandomGenerator > _random_generators;

std::ostream& operator<<(std::ostream& out, const RandomGenerator& rng)
{
    return out << rng.gen;
}

std::istream& operator>>(std::istream& in, RandomGenerator& rng)
{
    return in >> rng.gen;
}

//////////////// networks /////////////////
Network magicnetwork;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
    size_t var_size;
    size_t data_size;
    // C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
    // non-dynamic arrays
    if (name == "neurongroup._spikespace") {
        var_size = 4;
        data_size = 4*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.v") {
        var_size = 3;
        data_size = 3*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_v, data_size, s_value);
        }
        return;
    }
    if (name == "spikegeneratorgroup._spikespace") {
        var_size = 6;
        data_size = 6*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_spikegeneratorgroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_spikegeneratorgroup__spikespace, data_size, s_value);
        }
        return;
    }
    // dynamic arrays (1d)
    if (name == "synapses.delay") {
        var_size = _dynamic_array_synapses_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.delay") {
        var_size = _dynamic_array_synapses_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.lastupdate") {
        var_size = _dynamic_array_synapses_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.post") {
        var_size = _dynamic_array_synapses_post.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_post[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_post[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.pre") {
        var_size = _dynamic_array_synapses_pre.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_pre[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_pre[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.w") {
        var_size = _dynamic_array_synapses_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_w[0], data_size, s_value);
        }
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 4;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 3;
double * _array_neurongroup_v;
const int _num__array_neurongroup_v = 3;
int32_t * _array_spikegeneratorgroup__lastindex;
const int _num__array_spikegeneratorgroup__lastindex = 1;
int32_t * _array_spikegeneratorgroup__period_bins;
const int _num__array_spikegeneratorgroup__period_bins = 1;
int32_t * _array_spikegeneratorgroup__spikespace;
const int _num__array_spikegeneratorgroup__spikespace = 6;
int32_t * _array_spikegeneratorgroup_i;
const int _num__array_spikegeneratorgroup_i = 5;
double * _array_spikegeneratorgroup_period;
const int _num__array_spikegeneratorgroup_period = 1;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 3;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 3;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_statemonitor__indices;
const int _num__array_statemonitor__indices = 100;
int32_t * _array_statemonitor_N;
const int _num__array_statemonitor_N = 1;
double * _array_statemonitor_w;
const int _num__array_statemonitor_w = (0, 100);
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_spikegeneratorgroup__timebins;
std::vector<int32_t> _dynamic_array_spikegeneratorgroup_neuron_index;
std::vector<int32_t> _dynamic_array_spikegeneratorgroup_spike_number;
std::vector<double> _dynamic_array_spikegeneratorgroup_spike_time;
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<double> _dynamic_array_statemonitor_t;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<double> _dynamic_array_synapses_delay_1;
std::vector<double> _dynamic_array_synapses_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
std::vector<double> _dynamic_array_synapses_post;
std::vector<double> _dynamic_array_synapses_pre;
std::vector<double> _dynamic_array_synapses_w;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_statemonitor_w;

/////////////// static arrays /////////////
int32_t * _static_array__array_statemonitor__indices;
const int _num__static_array__array_statemonitor__indices = 100;
int32_t * _static_array__dynamic_array_spikegeneratorgroup__timebins;
const int _num__static_array__dynamic_array_spikegeneratorgroup__timebins = 9;
int64_t * _static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int _num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 9;
int64_t * _static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 9;
double * _static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 9;

//////////////// synapses /////////////////
// synapses
SynapticPathway synapses_post(
    _dynamic_array_synapses__synaptic_post,
    0, 3);
SynapticPathway synapses_pre(
    _dynamic_array_synapses__synaptic_pre,
    0, 5);

//////////////// clocks ///////////////////
// attributes will be set in run.cpp
Clock defaultclock;

// Profiling information for each code object
}

void _init_arrays()
{
    using namespace brian;

    // Arrays initialized to 0
    _array_defaultclock_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

    _array_defaultclock_t = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

    _array_defaultclock_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

    _array_neurongroup__spikespace = new int32_t[4];
    
    for(int i=0; i<4; i++) _array_neurongroup__spikespace[i] = 0;

    _array_neurongroup_i = new int32_t[3];
    
    for(int i=0; i<3; i++) _array_neurongroup_i[i] = 0;

    _array_neurongroup_v = new double[3];
    
    for(int i=0; i<3; i++) _array_neurongroup_v[i] = 0;

    _array_spikegeneratorgroup__lastindex = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;

    _array_spikegeneratorgroup__period_bins = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikegeneratorgroup__period_bins[i] = 0;

    _array_spikegeneratorgroup__spikespace = new int32_t[6];
    
    for(int i=0; i<6; i++) _array_spikegeneratorgroup__spikespace[i] = 0;

    _array_spikegeneratorgroup_i = new int32_t[5];
    
    for(int i=0; i<5; i++) _array_spikegeneratorgroup_i[i] = 0;

    _array_spikegeneratorgroup_period = new double[1];
    
    for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;

    _array_spikemonitor__source_idx = new int32_t[3];
    
    for(int i=0; i<3; i++) _array_spikemonitor__source_idx[i] = 0;

    _array_spikemonitor_count = new int32_t[3];
    
    for(int i=0; i<3; i++) _array_spikemonitor_count[i] = 0;

    _array_spikemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

    _array_statemonitor__indices = new int32_t[100];
    
    for(int i=0; i<100; i++) _array_statemonitor__indices[i] = 0;

    _array_statemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;

    _array_synapses_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_N[i] = 0;

    _dynamic_array_spikegeneratorgroup__timebins.resize(9);
    
    for(int i=0; i<9; i++) _dynamic_array_spikegeneratorgroup__timebins[i] = 0;


    // Arrays initialized to an "arange"
    _array_neurongroup_i = new int32_t[3];
    
    for(int i=0; i<3; i++) _array_neurongroup_i[i] = 0 + i;

    _array_spikegeneratorgroup_i = new int32_t[5];
    
    for(int i=0; i<5; i++) _array_spikegeneratorgroup_i[i] = 0 + i;

    _array_spikemonitor__source_idx = new int32_t[3];
    
    for(int i=0; i<3; i++) _array_spikemonitor__source_idx[i] = 0 + i;


    // static arrays
    _static_array__array_statemonitor__indices = new int32_t[100];
    _static_array__dynamic_array_spikegeneratorgroup__timebins = new int32_t[9];
    _static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[9];
    _static_array__dynamic_array_spikegeneratorgroup_spike_number = new int64_t[9];
    _static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[9];

    // Random number generator states
    std::random_device rd;
    for (int i=0; i<1; i++)
        _random_generators.push_back(RandomGenerator());
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_statemonitor__indices;
    f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor__indices.is_open())
    {
        f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 100*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
    }
    ifstream f_static_array__dynamic_array_spikegeneratorgroup__timebins;
    f_static_array__dynamic_array_spikegeneratorgroup__timebins.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup__timebins", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup__timebins.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup__timebins), 9*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
    f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 9*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 9*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 9*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    ofstream outfile__array_neurongroup__spikespace;
    outfile__array_neurongroup__spikespace.open(results_dir + "_array_neurongroup__spikespace_3522821529", ios::binary | ios::out);
    if(outfile__array_neurongroup__spikespace.is_open())
    {
        outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 4*sizeof(_array_neurongroup__spikespace[0]));
        outfile__array_neurongroup__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
    }
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open(results_dir + "_array_neurongroup_i_2649026944", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 3*sizeof(_array_neurongroup_i[0]));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    ofstream outfile__array_neurongroup_v;
    outfile__array_neurongroup_v.open(results_dir + "_array_neurongroup_v_283966581", ios::binary | ios::out);
    if(outfile__array_neurongroup_v.is_open())
    {
        outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 3*sizeof(_array_neurongroup_v[0]));
        outfile__array_neurongroup_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_v." << endl;
    }
    ofstream outfile__array_spikegeneratorgroup__lastindex;
    outfile__array_spikegeneratorgroup__lastindex.open(results_dir + "_array_spikegeneratorgroup__lastindex_987837788", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__lastindex.is_open())
    {
        outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(_array_spikegeneratorgroup__lastindex[0]));
        outfile__array_spikegeneratorgroup__lastindex.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
    }
    ofstream outfile__array_spikegeneratorgroup__period_bins;
    outfile__array_spikegeneratorgroup__period_bins.open(results_dir + "_array_spikegeneratorgroup__period_bins_4200411184", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__period_bins.is_open())
    {
        outfile__array_spikegeneratorgroup__period_bins.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__period_bins), 1*sizeof(_array_spikegeneratorgroup__period_bins[0]));
        outfile__array_spikegeneratorgroup__period_bins.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__period_bins." << endl;
    }
    ofstream outfile__array_spikegeneratorgroup__spikespace;
    outfile__array_spikegeneratorgroup__spikespace.open(results_dir + "_array_spikegeneratorgroup__spikespace_37288933", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__spikespace.is_open())
    {
        outfile__array_spikegeneratorgroup__spikespace.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__spikespace), 6*sizeof(_array_spikegeneratorgroup__spikespace[0]));
        outfile__array_spikegeneratorgroup__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__spikespace." << endl;
    }
    ofstream outfile__array_spikegeneratorgroup_i;
    outfile__array_spikegeneratorgroup_i.open(results_dir + "_array_spikegeneratorgroup_i_1329498599", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_i.is_open())
    {
        outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 5*sizeof(_array_spikegeneratorgroup_i[0]));
        outfile__array_spikegeneratorgroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
    }
    ofstream outfile__array_spikegeneratorgroup_period;
    outfile__array_spikegeneratorgroup_period.open(results_dir + "_array_spikegeneratorgroup_period_3457314764", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_period.is_open())
    {
        outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(_array_spikegeneratorgroup_period[0]));
        outfile__array_spikegeneratorgroup_period.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
    }
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open(results_dir + "_array_spikemonitor__source_idx_1477951789", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 3*sizeof(_array_spikemonitor__source_idx[0]));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open(results_dir + "_array_spikemonitor_count_598337445", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 3*sizeof(_array_spikemonitor_count[0]));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open(results_dir + "_array_spikemonitor_N_225734567", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    ofstream outfile__array_statemonitor__indices;
    outfile__array_statemonitor__indices.open(results_dir + "_array_statemonitor__indices_2854283999", ios::binary | ios::out);
    if(outfile__array_statemonitor__indices.is_open())
    {
        outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 100*sizeof(_array_statemonitor__indices[0]));
        outfile__array_statemonitor__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
    }
    ofstream outfile__array_statemonitor_N;
    outfile__array_statemonitor_N.open(results_dir + "_array_statemonitor_N_4140778434", ios::binary | ios::out);
    if(outfile__array_statemonitor_N.is_open())
    {
        outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(_array_statemonitor_N[0]));
        outfile__array_statemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_N." << endl;
    }
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open(results_dir + "_array_synapses_N_483293785", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    ofstream outfile__dynamic_array_spikegeneratorgroup__timebins;
    outfile__dynamic_array_spikegeneratorgroup__timebins.open(results_dir + "_dynamic_array_spikegeneratorgroup__timebins_4247931087", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        if (! _dynamic_array_spikegeneratorgroup__timebins.empty() )
        {
            outfile__dynamic_array_spikegeneratorgroup__timebins.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup__timebins[0]), _dynamic_array_spikegeneratorgroup__timebins.size()*sizeof(_dynamic_array_spikegeneratorgroup__timebins[0]));
            outfile__dynamic_array_spikegeneratorgroup__timebins.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
    outfile__dynamic_array_spikegeneratorgroup_neuron_index.open(results_dir + "_dynamic_array_spikegeneratorgroup_neuron_index_2789266935", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        if (! _dynamic_array_spikegeneratorgroup_neuron_index.empty() )
        {
            outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_neuron_index[0]), _dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(_dynamic_array_spikegeneratorgroup_neuron_index[0]));
            outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
    outfile__dynamic_array_spikegeneratorgroup_spike_number.open(results_dir + "_dynamic_array_spikegeneratorgroup_spike_number_3584615111", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        if (! _dynamic_array_spikegeneratorgroup_spike_number.empty() )
        {
            outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_spike_number[0]), _dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(_dynamic_array_spikegeneratorgroup_spike_number[0]));
            outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
    outfile__dynamic_array_spikegeneratorgroup_spike_time.open(results_dir + "_dynamic_array_spikegeneratorgroup_spike_time_2775435892", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        if (! _dynamic_array_spikegeneratorgroup_spike_time.empty() )
        {
            outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(&_dynamic_array_spikegeneratorgroup_spike_time[0]), _dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(_dynamic_array_spikegeneratorgroup_spike_time[0]));
            outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open(results_dir + "_dynamic_array_spikemonitor_i_1976709050", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
            outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
            outfile__dynamic_array_spikemonitor_i.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open(results_dir + "_dynamic_array_spikemonitor_t_383009635", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
            outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
            outfile__dynamic_array_spikemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_statemonitor_t;
    outfile__dynamic_array_statemonitor_t.open(results_dir + "_dynamic_array_statemonitor_t_3983503110", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_t.is_open())
    {
        if (! _dynamic_array_statemonitor_t.empty() )
        {
            outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
            outfile__dynamic_array_statemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open(results_dir + "_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
            outfile__dynamic_array_synapses__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open(results_dir + "_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
            outfile__dynamic_array_synapses__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open(results_dir + "_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        if (! _dynamic_array_synapses_delay.empty() )
        {
            outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
            outfile__dynamic_array_synapses_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay_1;
    outfile__dynamic_array_synapses_delay_1.open(results_dir + "_dynamic_array_synapses_delay_1_3310102259", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay_1.is_open())
    {
        if (! _dynamic_array_synapses_delay_1.empty() )
        {
            outfile__dynamic_array_synapses_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay_1[0]), _dynamic_array_synapses_delay_1.size()*sizeof(_dynamic_array_synapses_delay_1[0]));
            outfile__dynamic_array_synapses_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_synapses_lastupdate;
    outfile__dynamic_array_synapses_lastupdate.open(results_dir + "_dynamic_array_synapses_lastupdate_3710850267", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_lastupdate.is_open())
    {
        if (! _dynamic_array_synapses_lastupdate.empty() )
        {
            outfile__dynamic_array_synapses_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_lastupdate[0]), _dynamic_array_synapses_lastupdate.size()*sizeof(_dynamic_array_synapses_lastupdate[0]));
            outfile__dynamic_array_synapses_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open(results_dir + "_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
            outfile__dynamic_array_synapses_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open(results_dir + "_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
            outfile__dynamic_array_synapses_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_post;
    outfile__dynamic_array_synapses_post.open(results_dir + "_dynamic_array_synapses_post_2494337290", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_post.is_open())
    {
        if (! _dynamic_array_synapses_post.empty() )
        {
            outfile__dynamic_array_synapses_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_post[0]), _dynamic_array_synapses_post.size()*sizeof(_dynamic_array_synapses_post[0]));
            outfile__dynamic_array_synapses_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_pre;
    outfile__dynamic_array_synapses_pre.open(results_dir + "_dynamic_array_synapses_pre_3430426518", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_pre.is_open())
    {
        if (! _dynamic_array_synapses_pre.empty() )
        {
            outfile__dynamic_array_synapses_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_pre[0]), _dynamic_array_synapses_pre.size()*sizeof(_dynamic_array_synapses_pre[0]));
            outfile__dynamic_array_synapses_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_w;
    outfile__dynamic_array_synapses_w.open(results_dir + "_dynamic_array_synapses_w_441891901", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_w.is_open())
    {
        if (! _dynamic_array_synapses_w.empty() )
        {
            outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(&_dynamic_array_synapses_w[0]), _dynamic_array_synapses_w.size()*sizeof(_dynamic_array_synapses_w[0]));
            outfile__dynamic_array_synapses_w.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
    }

    ofstream outfile__dynamic_array_statemonitor_w;
    outfile__dynamic_array_statemonitor_w.open(results_dir + "_dynamic_array_statemonitor_w_1952857788", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_w.is_open())
    {
        for (int n=0; n<_dynamic_array_statemonitor_w.n; n++)
        {
            if (! _dynamic_array_statemonitor_w(n).empty())
            {
                outfile__dynamic_array_statemonitor_w.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_w(n, 0)), _dynamic_array_statemonitor_w.m*sizeof(_dynamic_array_statemonitor_w(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_w." << endl;
    }

    // Write spike queue states to disk
    ofstream outfile_synapses_post;
    outfile_synapses_post.open(results_dir + "synapses_post_queue", ios::out);
    if (outfile_synapses_post.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_post << *synapses_post.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_post' for file" << std::endl;
    }
    ofstream outfile_synapses_pre;
    outfile_synapses_pre.open(results_dir + "synapses_pre_queue", ios::out);
    if (outfile_synapses_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_pre << *synapses_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_pre' for file" << std::endl;
    }

    // Write random generator state to disk
    ofstream random_generator_state;
    random_generator_state.open(results_dir + "random_generator_state", ios::out);
    if (random_generator_state.is_open()) {
        for (int i=0; i<1; i++)
            random_generator_state << _random_generators[i] << "\n";
    } else {
        std::cout << "Error writing random generator state to file." << std::endl;
    }

    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

void _dealloc_arrays()
{
    using namespace brian;


    // static arrays
    if(_static_array__array_statemonitor__indices!=0)
    {
        delete [] _static_array__array_statemonitor__indices;
        _static_array__array_statemonitor__indices = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup__timebins!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup__timebins;
        _static_array__dynamic_array_spikegeneratorgroup__timebins = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_neuron_index!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_neuron_index;
        _static_array__dynamic_array_spikegeneratorgroup_neuron_index = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_number!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_number;
        _static_array__dynamic_array_spikegeneratorgroup_spike_number = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_time!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_time;
        _static_array__dynamic_array_spikegeneratorgroup_spike_time = 0;
    }
}

