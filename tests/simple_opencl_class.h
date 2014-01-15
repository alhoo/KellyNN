#ifndef SIMPLE_OPENCL_CLASS_H
#define SIMPLE_OPENCL_CLASS_H
#include <CL/cl.h>
#include <iostream>
#include <cassert>

using namespace std;
void print_opencl_error(cl_int err);

class simple_opencl_class{
    cl_context      ctx;
    cl_device_id    device;
    cl_command_queue queue;
    cl_event        event;
    cl_int          err;
    cl_platform_id  platform;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    public:
        simple_opencl_class();
        int test();
        ~simple_opencl_class();
};

#endif
