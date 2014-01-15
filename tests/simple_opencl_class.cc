#include "simple_opencl_class.h"

using namespace std;


simple_opencl_class::simple_opencl_class(){
        clGetPlatformIDs(1, &platform, &ret_num_platforms);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &ret_num_devices);
    assert(ret_num_devices > 0);
        ctx = clCreateContext(NULL, ret_num_devices, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
        queue = clCreateCommandQueue(ctx, device, 0, &err);
    assert(err == CL_SUCCESS);
}

int simple_opencl_class::test(){
        cl_mem M;
        M = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 10 * sizeof(cl_float), NULL, &err);
    assert(err == CL_SUCCESS);

        float tmp[10];
        for(size_t i = 0; i < 10; ++i) tmp[i] = 10*i;
        err = clEnqueueWriteBuffer(queue, M, CL_TRUE, 0*sizeof(long), 10* sizeof(*tmp), tmp, 0, NULL, &event);
    assert(err == CL_SUCCESS);
        wait();

        for(size_t i = 0; i < 10; ++i) tmp[i] = 0;
        err = clEnqueueReadBuffer(queue, M, CL_TRUE, 0*sizeof(float), 10*sizeof(float), tmp, 0, NULL, &event);
    assert(err == CL_SUCCESS);
        wait();

        for(size_t i = 0; i < 10; ++i) if(tmp[i] != 10*i) return 0;
        return 1;
}
simple_opencl_class::~simple_opencl_class(){
        clReleaseContext(ctx);
}

void print_opencl_error(cl_int err){
    switch(err){
        //case CL_SUCCESS : cerr << "\t\t\tthe function was executed successfully." << endl; break;
        case CL_SUCCESS: break;
        case CL_INVALID_COMMAND_QUEUE : cerr << "\t\t\tcommand_queue is not a valid command-queue." << endl; break;
        case CL_INVALID_CONTEXT : cerr << "\t\t\tINVALID CONTEXT." << endl; break;
        case CL_INVALID_MEM_OBJECT : cerr << "\t\t\tbuffer is not a valid buffer object." << endl; break;
        case CL_INVALID_VALUE : cerr << "\t\t\tINVALID value." << endl; break;
        case CL_INVALID_EVENT_WAIT_LIST : cerr << "\t\t\tINVALID event WAIT LIST." << endl; break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET : cerr << "\t\t\tCL_MISALIGNED_SUB_BUFFER_OFFSET." << endl; break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST : cerr << "\t\t\tCL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST." << endl; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE : cerr << "\t\t\tCL_MEM_OBJECT_ALLOCATION_FAILURE." << endl; break;
        case CL_OUT_OF_RESOURCES : cerr << "\t\t\tCL_OUT_OF_RESOURCES" << endl; break;
        case CL_OUT_OF_HOST_MEMORY : cerr << "\t\t\tCL_OUT_OF_HOST_MEMORY." << endl; break;
        case CL_INVALID_KERNEL : cerr << "\t\t\tyou have an invalid kernel!" << endl; break;
        case CL_DEVICE_NOT_FOUND : cerr << "\t\t\terror CL_DEVICE_NOT_FOUND" << endl; break;
        case CL_DEVICE_NOT_AVAILABLE : cerr << "\t\t\terror CL_DEVICE_NOT_AVAILABLE" << endl; break;
        case CL_COMPILER_NOT_AVAILABLE : cerr << "\t\t\terror CL_COMPILER_NOT_AVAILABLE" << endl; break;
        case CL_PROFILING_INFO_NOT_AVAILABLE : cerr << "\t\t\terror CL_PROFILING_INFO_NOT_AVAILABLE" << endl; break;
        case CL_MEM_COPY_OVERLAP : cerr << "\t\t\terror CL_MEM_COPY_OVERLAP" << endl; break;
        case CL_IMAGE_FORMAT_MISMATCH : cerr << "\t\t\terror CL_IMAGE_FORMAT_MISMATCH" << endl; break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED : cerr << "\t\t\terror CL_IMAGE_FORMAT_NOT_SUPPORTED" << endl; break;
        case CL_BUILD_PROGRAM_FAILURE : cerr << "\t\t\terror CL_BUILD_PROGRAM_FAILURE" << endl; break;
        case CL_MAP_FAILURE : cerr << "\t\t\terror CL_MAP_FAILURE" << endl; break;
        case CL_INVALID_DEVICE_TYPE : cerr << "\t\t\terror CL_INVALID_DEVICE_TYPE" << endl; break;
        case CL_INVALID_PLATFORM : cerr << "\t\t\terror CL_INVALID_PLATFORM" << endl; break;
        case CL_INVALID_DEVICE : cerr << "\t\t\terror CL_INVALID_DEVICE" << endl; break;
        case CL_INVALID_QUEUE_PROPERTIES : cerr << "\t\t\terror CL_INVALID_QUEUE_PROPERTIES" << endl; break;
        case CL_INVALID_HOST_PTR : cerr << "\t\t\terror CL_INVALID_HOST_PTR" << endl; break;
        case CL_INVALID_IMAGE_SIZE : cerr << "\t\t\terror CL_INVALID_IMAGE_SIZE" << endl; break;
        case CL_INVALID_SAMPLER : cerr << "\t\t\terror CL_INVALID_SAMPLER" << endl; break;
        case CL_INVALID_BINARY : cerr << "\t\t\terror CL_INVALID_BINARY" << endl; break;
        case CL_INVALID_BUILD_OPTIONS : cerr << "\t\t\terror CL_INVALID_BUILD_OPTIONS" << endl; break;
        case CL_INVALID_PROGRAM : cerr << "\t\t\terror CL_INVALID_PROGRAM" << endl; break;
        case CL_INVALID_PROGRAM_EXECUTABLE : cerr << "\t\t\terror CL_INVALID_PROGRAM_EXECUTABLE" << endl; break;
        case CL_INVALID_KERNEL_NAME : cerr << "\t\t\terror CL_INVALID_KERNEL_NAME" << endl; break;
        case CL_INVALID_KERNEL_DEFINITION : cerr << "\t\t\terror CL_INVALID_KERNEL_DEFINITION" << endl; break;
        case CL_INVALID_ARG_INDEX : cerr << "\t\t\terror CL_INVALID_ARG_INDEX" << endl; break;
        case CL_INVALID_ARG_VALUE : cerr << "\t\t\terror CL_INVALID_ARG_VALUE" << endl; break;
        case CL_INVALID_ARG_SIZE : cerr << "\t\t\terror CL_INVALID_ARG_SIZE" << endl; break;
        case CL_INVALID_KERNEL_ARGS : cerr << "\t\t\terror CL_INVALID_KERNEL_ARGS" << endl; break;
        case CL_INVALID_WORK_DIMENSION : cerr << "\t\t\terror CL_INVALID_WORK_DIMENSION" << endl; break;
        case CL_INVALID_WORK_GROUP_SIZE : cerr << "\t\t\terror CL_INVALID_WORK_GROUP_SIZE" << endl; break;
        case CL_INVALID_WORK_ITEM_SIZE : cerr << "\t\t\terror CL_INVALID_WORK_ITEM_SIZE" << endl; break;
        case CL_INVALID_GLOBAL_OFFSET : cerr << "\t\t\terror CL_INVALID_GLOBAL_OFFSET" << endl; break;
        case CL_INVALID_EVENT : cerr << "\t\t\terror CL_INVALID_EVENT" << endl; break;
        case CL_INVALID_OPERATION : cerr << "\t\t\terror CL_INVALID_OPERATION" << endl; break;
        case CL_INVALID_GL_OBJECT : cerr << "\t\t\terror CL_INVALID_GL_OBJECT" << endl; break;
        case CL_INVALID_BUFFER_SIZE : cerr << "\t\t\terror CL_INVALID_BUFFER_SIZE" << endl; break;
        case CL_INVALID_MIP_LEVEL : cerr << "\t\t\terror CL_INVALID_MIP_LEVEL" << endl; break;
        case CL_INVALID_GLOBAL_WORK_SIZE : cerr << "\t\t\terror CL_INVALID_GLOBAL_WORK_SIZE" << endl; break;
        case CL_INVALID_PROPERTY : cerr << "\t\t\terror CL_INVALID_PROPERTY" << endl; break;
        //case CL_BUILD_NONE : cerr << "\t\t\terror CL_BUILD_NONE" << endl; break;
        //case CL_BUILD_ERROR : cerr << "\t\t\terror CL_BUILD_ERROR" << endl; break;
        //case CL_BUILD_IN_PROGRESS : cerr << "\t\t\terror CL_BUILD_IN_PROGRESS" << endl; break;
        default:
             cerr << "\t\t\tthere is an unknown error " << err << endl;
    }
}
