#ifndef BRAIN_FUNCTIONS_HH
#define BRAIN_FUNCTIONS_HH
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <glob.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <map>

#define MAX_SOURCE_SIZE 4096
#define NEURONINITBAL 10000
#define SYNAPSINITBAL 1000

#ifndef DEBUG
#define DEBUG (0)
#endif

#ifndef VERBOSE
#define VERBOSE (0)
#endif

using namespace std;

typedef cl_mem Mat;
typedef cl_mem Col;

class opencl_brain_functions{
    cl_context      ctx;
    cl_device_id    device;
    cl_command_queue queue;
    cl_event        event;
    map<string,cl_kernel>       kernels;
    //cl_kernel       kernels[15];
    cl_int          err;
    Col             world;
    cl_float        *tmp;
    int             w,I,O;
    unsigned long   mem_allocated;
    void init_kernels(string = "");
    void init_kernel(string);


    public:
        opencl_brain_functions(int w = 4,int I = 1,int O = 1);
        ~opencl_brain_functions();
	
	void	info();

        cl_mem  gpu_malloc(cl_mem,size_t,float = 1.0);
        void    gpu_free(cl_mem);
        void    opencl_setv(Mat S,long *a,int start, int stop);
        void    opencl_getv(Mat S,long *a,int start, int stop);
        void    opencl_setv(Mat S,float *a,int start, int stop);
        void    opencl_getv(Mat S,float *a,int start, int stop);
        void    opencl_copy(Mat,Mat);
        void    setIO(int ni,int no){I=ni,O=no;}
        void    wait();

        void    opencl_synaps_bet(Mat,Mat,Mat,Mat);
        void    opencl_bet_sum(Col,Mat);
        void    opencl_set(Col,int,cl_float v);
        void    opencl_synaps_refresh(Mat,Mat,Mat,Col);
        void    opencl_neuron_refresh(Col,Col,Col);
        void    opencl_find_winning_synapses(Mat,Mat,Col,Col,Col,Mat);
        void    opencl_find_winning_neurons(Col,Mat,Mat,Mat);
        void    opencl_update_synaps_info(Col);
        void    opencl_synaps_learn(Mat,Mat,Mat,Mat,Mat,Mat,Mat,Col,Col);
        void    opencl_synaps_learn_negation(Mat SP1, Mat SP0);
        void    opencl_synaps_learn2(Mat,Mat,Col);
        void    opencl_pay(Mat,Mat,Col,Col,Col,Mat,Col,Mat,Col);
        void    opencl_npay(Col NBET0,Col NBET1,Col NBAL,Col NW,Col NBETSUM);
        float   opencl_get_bal(Col,Mat,Col,Col);
        float   opencl_get_bal_bad(Col,Col,Mat,Col,Col,Col,Col);
        void    opencl_pay_neuron(Col,Col,int,float);
        void    opencl_synaps_die(Mat,Mat,long s, long v, long l);
        void    opencl_fill(Mat,cl_float,int);
        cl_float    opencl_sum(Mat M, long l);

        string  print(Mat S, int w, int h,int l = 0, char * A = NULL);
        void    printW(Col T, Col F, Mat S, int w, Col I, char * A = NULL, bool n = 1);
        void    print(Mat S, int p);

};

#endif
