#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<chrono>
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
        
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int32_t,int32_t> { typedef int32_t type; };
    template < > struct _higher_type<int32_t,int64_t> { typedef int64_t type; };
    template < > struct _higher_type<int32_t,float> { typedef float type; };
    template < > struct _higher_type<int32_t,double> { typedef double type; };
    template < > struct _higher_type<int32_t,long double> { typedef long double type; };
    template < > struct _higher_type<int64_t,int32_t> { typedef int64_t type; };
    template < > struct _higher_type<int64_t,int64_t> { typedef int64_t type; };
    template < > struct _higher_type<int64_t,float> { typedef float type; };
    template < > struct _higher_type<int64_t,double> { typedef double type; };
    template < > struct _higher_type<int64_t,long double> { typedef long double type; };
    template < > struct _higher_type<float,int32_t> { typedef float type; };
    template < > struct _higher_type<float,int64_t> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int32_t> { typedef double type; };
    template < > struct _higher_type<double,int64_t> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int32_t> { typedef long double type; };
    template < > struct _higher_type<long double,int64_t> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    // General template, used for floating point types
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {
        return x-y*floor(1.0*x/y);
    }
    // Specific implementations for integer types
    // (from Cython, see LICENSE file)
    template <>
    inline int32_t _brian_mod(int32_t x, int32_t y)
    {
        int32_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int32_t x, int64_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int64_t x, int32_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int64_t x, int64_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    // General implementation, used for floating point types
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    // Specific implementations for integer types
    // (from Cython, see LICENSE file)
    template <>
    inline int32_t _brian_floordiv<int32_t, int32_t>(int32_t a, int32_t b) {
        int32_t q = a / b;
        int32_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int32_t, int64_t>(int32_t a, int64_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int64_t, int>(int64_t a, int32_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int64_t, int64_t>(int64_t a, int64_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_spikegeneratorgroup_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const int64_t N = 5;
const size_t _num_lastindex = 1;
const size_t _num_period_bins = 1;
const size_t _num_spikespace = 6;
int32_t* const _array_spikegeneratorgroup__timebins = _dynamic_array_spikegeneratorgroup__timebins.empty()? 0 : &_dynamic_array_spikegeneratorgroup__timebins[0];
const size_t _num_timebins = _dynamic_array_spikegeneratorgroup__timebins.size();
int32_t* const _array_spikegeneratorgroup_neuron_index = _dynamic_array_spikegeneratorgroup_neuron_index.empty()? 0 : &_dynamic_array_spikegeneratorgroup_neuron_index[0];
const size_t _numneuron_index = _dynamic_array_spikegeneratorgroup_neuron_index.size();
const size_t _numt_in_timesteps = 1;
int32_t* const _array_spikegeneratorgroup_spike_number = _dynamic_array_spikegeneratorgroup_spike_number.empty()? 0 : &_dynamic_array_spikegeneratorgroup_spike_number[0];
const size_t _numspike_number = _dynamic_array_spikegeneratorgroup_spike_number.size();
    ///// POINTERS ////////////
        
    int32_t*   _ptr_array_spikegeneratorgroup__lastindex = _array_spikegeneratorgroup__lastindex;
    int32_t*   _ptr_array_spikegeneratorgroup__period_bins = _array_spikegeneratorgroup__period_bins;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup__spikespace = _array_spikegeneratorgroup__spikespace;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup__timebins = _array_spikegeneratorgroup__timebins;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup_neuron_index = _array_spikegeneratorgroup_neuron_index;
    int64_t*   _ptr_array_defaultclock_timestep = _array_defaultclock_timestep;
    int32_t* __restrict  _ptr_array_spikegeneratorgroup_spike_number = _array_spikegeneratorgroup_spike_number;



    const int32_t _the_period = _ptr_array_spikegeneratorgroup__period_bins[0];
    int32_t _timebin          = _ptr_array_defaultclock_timestep[0];
    const int32_t _n_spikes   = 0;

    if (_the_period > 0) {
        _timebin %= _the_period;
        // If there is a periodicity in the SpikeGenerator, we need to reset the
        // lastindex when the period has passed
        if (_ptr_array_spikegeneratorgroup__lastindex[0] > 0 && _ptr_array_spikegeneratorgroup__timebins[_ptr_array_spikegeneratorgroup__lastindex[0] - 1] >= _timebin)
            _ptr_array_spikegeneratorgroup__lastindex[0] = 0;
    }

    int32_t _cpp_numspikes = 0;

    for(size_t _idx=_ptr_array_spikegeneratorgroup__lastindex[0]; _idx < _num_timebins; _idx++)
    {
        if (_ptr_array_spikegeneratorgroup__timebins[_idx] > _timebin)
            break;

        _ptr_array_spikegeneratorgroup__spikespace[_cpp_numspikes++] = _ptr_array_spikegeneratorgroup_neuron_index[_idx];
    }

    _ptr_array_spikegeneratorgroup__spikespace[N] = _cpp_numspikes;

    _ptr_array_spikegeneratorgroup__lastindex[0] += _cpp_numspikes;



}


