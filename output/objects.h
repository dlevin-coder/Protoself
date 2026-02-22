
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<chrono>
#include<random>
#include<vector>


namespace brian {

extern std::string results_dir;

class RandomGenerator {
    private:
        std::mt19937 gen;
        double stored_gauss;
        bool has_stored_gauss = false;
    public:
        RandomGenerator() {
            seed();
        }
        void seed() {
            std::random_device rd;
            gen.seed(rd());
            has_stored_gauss = false;
        }
        void seed(unsigned long seed) {
            gen.seed(seed);
            has_stored_gauss = false;
        }
        // Allow exporting/setting the internal state of the random generator
        friend std::ostream& operator<<(std::ostream& out, const RandomGenerator& rng);
        friend std::istream& operator>>(std::istream& in, RandomGenerator& rng);

        double rand() {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            const long a = gen() >> 5;
            const long b = gen() >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        double randn() {
            if (has_stored_gauss) {
                const double tmp = stored_gauss;
                has_stored_gauss = false;
                return tmp;
            }
            else {
                double f, x1, x2, r2;

                do {
                    x1 = 2.0*rand() - 1.0;
                    x2 = 2.0*rand() - 1.0;
                    r2 = x1*x1 + x2*x2;
                }
                while (r2 >= 1.0 || r2 == 0.0);

                /* Box-Muller transform */
                f = sqrt(-2.0*log(r2)/r2);
                /* Keep for next call */
                stored_gauss = f*x1;
                has_stored_gauss = true;
                return f*x2;
            }
        }
};

extern std::ostream& operator<<(std::ostream& out, const RandomGenerator& rng);
extern std::istream& operator>>(std::istream& in, RandomGenerator& rng);

// In OpenMP we need one state per thread
extern std::vector< RandomGenerator > _random_generators;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;



void set_variable_by_name(std::string, std::string);

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup__timebins;
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup_neuron_index;
extern std::vector<int32_t> _dynamic_array_spikegeneratorgroup_spike_number;
extern std::vector<double> _dynamic_array_spikegeneratorgroup_spike_time;
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_delay;
extern std::vector<double> _dynamic_array_synapses_delay_1;
extern std::vector<double> _dynamic_array_synapses_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_post;
extern std::vector<double> _dynamic_array_synapses_pre;
extern std::vector<double> _dynamic_array_synapses_w;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern int32_t *_array_spikegeneratorgroup__lastindex;
extern const int _num__array_spikegeneratorgroup__lastindex;
extern int32_t *_array_spikegeneratorgroup__period_bins;
extern const int _num__array_spikegeneratorgroup__period_bins;
extern int32_t *_array_spikegeneratorgroup__spikespace;
extern const int _num__array_spikegeneratorgroup__spikespace;
extern int32_t *_array_spikegeneratorgroup_i;
extern const int _num__array_spikegeneratorgroup_i;
extern double *_array_spikegeneratorgroup_period;
extern const int _num__array_spikegeneratorgroup_period;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t *_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;
extern int32_t *_array_statemonitor_N;
extern const int _num__array_statemonitor_N;
extern double *_array_statemonitor_w;
extern const int _num__array_statemonitor_w;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_statemonitor_w;

/////////////// static arrays /////////////
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;
extern int32_t *_static_array__dynamic_array_spikegeneratorgroup__timebins;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup__timebins;
extern int64_t *_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_neuron_index;
extern int64_t *_static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_number;
extern double *_static_array__dynamic_array_spikegeneratorgroup_spike_time;
extern const int _num__static_array__dynamic_array_spikegeneratorgroup_spike_time;

//////////////// synapses /////////////////
// synapses
extern SynapticPathway synapses_post;
extern SynapticPathway synapses_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


