#include "brain_functions.h"


void print_opencl_error(cl_int err);

opencl_brain_functions::~opencl_brain_functions(){
    for(map<string,cl_kernel>::iterator it = kernels.begin(); it != kernels.end(); ++it)
        clReleaseKernel(it->second);
    info();
    wait();
    clReleaseContext(ctx);
}
opencl_brain_functions::opencl_brain_functions(int w,int I,int O):w(w),I(I),O(O){
    tmp = new cl_float[w*w];
    for(int i = 0; i < w*w; ++i) tmp[i]=1.0;
    cl_platform_id  platform;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    clGetPlatformIDs(1, &platform, &ret_num_platforms);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &ret_num_devices);
    assert(ret_num_devices > 0);
    ctx = clCreateContext(NULL, ret_num_devices, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    print_opencl_error(err);
    assert(err == CL_SUCCESS);
    world = gpu_malloc(world,128);
    long v[6] = {w,w*w,NEURONINITBAL,SYNAPSINITBAL,I,O};
    opencl_setv(world,&v[0],0,6);
    init_kernels();
}
cl_mem opencl_brain_functions::gpu_malloc(cl_mem M,size_t s,float a){
    M = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, s * sizeof(cl_float), NULL, &err);
    assert(err == CL_SUCCESS);
    for(size_t i = 0; i < s; ++i) tmp[i] = a;
    opencl_setv(M,tmp,0,s);
    mem_allocated += s * sizeof(cl_float);
    return M;
}
void opencl_brain_functions::gpu_free(cl_mem M){
    if(VERBOSE>1) cout << "\t\t\tgpu_free()" << endl;
    if(DEBUG) {int quit = 0; cin >> quit; assert(quit);};
    size_t s;
    clGetMemObjectInfo(M,CL_MEM_SIZE,sizeof(size_t),&s,NULL);
    clReleaseMemObject(M);
    mem_allocated -= s * sizeof(size_t);
}


void opencl_brain_functions::wait(){
    if (err != CL_SUCCESS) {
        print_opencl_error(err);
    }
    else {
        clWaitForEvents(1, &event);
    }
    assert(err == CL_SUCCESS);
}

float   opencl_brain_functions::opencl_get_bal(Col N,Mat S,Col NB1, Col NB0){
   
    float ret = opencl_sum(N  , w);
    ret      += opencl_sum(NB0, w);
    ret      += opencl_sum(NB1, w);
    ret      += opencl_sum(S  , w*w);
    return ret;
}

void    opencl_brain_functions::opencl_pay_neuron(Col BAL,Col NW,Mat SW,int i,float v){
    opencl_getv(BAL,tmp,i,i+1);
    tmp[0] += v;
    opencl_setv(BAL,tmp,i,i+1);
    tmp[0] = (v>0)*2.0 - 1.0;
    for(i = 0; i < I; ++i) tmp[i+1] = 1.0;
    for(; i < w; ++i) tmp[i+1] = 0;
    opencl_setv(NW,tmp,0,w);
    for(i = 0; i < w; ++i) tmp[0] = (v>0)*2.0 - 1.0;
    opencl_setv(SW,tmp,0,w);
}

void opencl_brain_functions::opencl_setv(Mat S,long *a,int start, int stop){
    assert(start < stop);
    err = clEnqueueWriteBuffer(queue, S, CL_TRUE, start*sizeof(long),
        (stop - start)* sizeof(*a), a, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_getv(Mat S,long *a,int start, int stop)
{
    assert(start < stop);
    err = clEnqueueReadBuffer(queue, S, CL_TRUE, start*sizeof(long),
        (stop - start)*sizeof(long), a, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_setv(Mat S,float *a,int start, int stop){
    assert(start < stop);
    err = clEnqueueWriteBuffer(queue, S, CL_TRUE, start*sizeof(float),
        (stop - start)*sizeof(float), a, 0, NULL, &event);
    print_opencl_error(err);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_getv(Mat S,float *a,int start, int stop)
{
    assert(start < stop);
    err = clEnqueueReadBuffer(queue, S, CL_TRUE, start*sizeof(float),
        (stop - start)*sizeof(float), a, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}





void    opencl_brain_functions::opencl_synaps_die(Mat SBAL, Mat SP1,long s, long ve, long l){
    if(l == 0) return;
    long v[3] = {s,ve,l};
    opencl_setv(world,&v[0],6,9);
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SBAL);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SP1);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = l + (1 - ve)*(w*(l - 1));
    size_t local_item_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    
    print_opencl_error(err);
    assert(err == CL_SUCCESS);
    wait();


}
void    opencl_brain_functions::opencl_update_synaps_info(Col M){
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&M);
    size_t global_item_size = 1;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
    
}
void opencl_brain_functions::opencl_synaps_bet(Mat SBET0,Mat SBAL,Mat SP0,Mat SP)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SBET0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SBAL);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&SP0);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&SP);
    clSetKernelArg(kernels.at(__func__), 4, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w*w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_bet_sum(Col NBET0,Mat SBET0)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&NBET0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SBET0);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_synaps_refresh(Mat SP0,Mat SP01,Mat SP00,Col NP)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SP0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SP01);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&SP00);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&NP);
    clSetKernelArg(kernels.at(__func__), 4, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w*w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    print_opencl_error(err);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_neuron_refresh(Col NBET0,Col NBET1,Col NP)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&NBET0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&NBET1);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&NP);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}

void opencl_brain_functions::opencl_find_winning_neurons(Col NW,Mat SW){
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]<" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&NW);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SW);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_set(Col NW,int n,cl_float v){
    err = clEnqueueWriteBuffer(queue, NW, CL_TRUE, 0,
        (n)* sizeof(v), &v, 0, NULL, &event);
    assert(err == CL_SUCCESS);
}
void opencl_brain_functions::opencl_find_winning_synapses(Mat SBET0,Mat SBET1,Col NBET0,Col NBET1,Col NW,Mat SW)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]>" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SBET0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SBET1);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&NBET0);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&NBET1);
    clSetKernelArg(kernels.at(__func__), 4, sizeof(cl_mem), (void *)&NW);
    clSetKernelArg(kernels.at(__func__), 5, sizeof(cl_mem), (void *)&SW);
    clSetKernelArg(kernels.at(__func__), 6, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w*w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_synaps_learn(Mat SP00,Mat SBET0,Mat SBET1,Mat SW,Col NBET0,Col NBET1,Col SINFO)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SP00);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SBET0);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&SBET1);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&SW);
    clSetKernelArg(kernels.at(__func__), 4, sizeof(cl_mem), (void *)&NBET0);
    clSetKernelArg(kernels.at(__func__), 5, sizeof(cl_mem), (void *)&NBET1);
    clSetKernelArg(kernels.at(__func__), 6, sizeof(cl_mem), (void *)&SINFO);
    clSetKernelArg(kernels.at(__func__), 7, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w*w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_synaps_learn2(Mat SP,Mat SW,Col SINFO)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SP);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SW);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&SINFO);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&world);
    size_t global_item_size = w*w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_pay(Mat SBET0,Mat SBET1,Col NBET0,Col NBET1,Mat SBAL,Col NBAL,Mat SW,Col NW)
{
    if(VERBOSE>1) cout << "kernel[" << __func__ << "]" << endl;
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SBET0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SBET1);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&NBET0);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&NBET1);
    clSetKernelArg(kernels.at(__func__), 4, sizeof(cl_mem), (void *)&SBAL);
    clSetKernelArg(kernels.at(__func__), 5, sizeof(cl_mem), (void *)&NBAL);
    clSetKernelArg(kernels.at(__func__), 6, sizeof(cl_mem), (void *)&SW);
    clSetKernelArg(kernels.at(__func__), 7, sizeof(cl_mem), (void *)&NW);
    clSetKernelArg(kernels.at(__func__), 8, sizeof(cl_mem), (void *)&world);
/*
    clSetKernelArg(kernels.at(__func__), 0, sizeof(cl_mem), (void *)&SBET0);
    clSetKernelArg(kernels.at(__func__), 1, sizeof(cl_mem), (void *)&SBET1);
    clSetKernelArg(kernels.at(__func__), 2, sizeof(cl_mem), (void *)&NWIN);
    clSetKernelArg(kernels.at(__func__), 3, sizeof(cl_mem), (void *)&NBET0);
    clSetKernelArg(kernels.at(__func__), 4, sizeof(cl_mem), (void *)&NBET1);
    clSetKernelArg(kernels.at(__func__), 5, sizeof(cl_mem), (void *)&SBAL);
    clSetKernelArg(kernels.at(__func__), 6, sizeof(cl_mem), (void *)&NBAL);
    clSetKernelArg(kernels.at(__func__), 7, sizeof(cl_mem), (void *)&SW);
    clSetKernelArg(kernels.at(__func__), 8, sizeof(cl_mem), (void *)&NW);
    clSetKernelArg(kernels.at(__func__), 9, sizeof(cl_mem), (void *)&world);
*/
    size_t global_item_size = w;
    size_t local_item_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernels.at(__func__), 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, &event);
    print_opencl_error(err);
    assert(err == CL_SUCCESS);
    wait();
}
void opencl_brain_functions::opencl_fill(Mat S,float a,int h)
{
    float b[h];
    for(int i = 0; i < h; ++i) b[i] = a;
    opencl_setv(S,b,0,h);
    wait();
}

cl_float   opencl_brain_functions::opencl_sum(Col M, long s)
{
    if(DEBUG) {int quit = 0; cin >> quit; assert(quit);};
    opencl_getv(M,tmp,0,s);
    float SUM = tmp[0];
    for(int i = 1; i < s; ++i) SUM += tmp[i];
    return SUM;
}


char mult[6]={'B','K','M','G','T','P'};
void humanprint(unsigned long s){
    int i =0;
    while(s > 1024 && i < 5){
        s/=1024;
        i++;
    }
    if(VERBOSE>1) cout << s << mult[i];
    if(i && VERBOSE>1) cout << "\t\t\tB";
}

void opencl_brain_functions::info(){
    humanprint(mem_allocated);
    if(VERBOSE>1) cout << endl;
}

void opencl_brain_functions::init_kernel(string filename){
    FILE *fp;
    char *source_str;
    size_t source_size;

    int err;

    fp = fopen(filename.c_str(), "r");
    if (!fp) {
        if(VERBOSE>1) cout  << "!";
        cerr << "\t\t\tFailed to load kernel " << filename << "." << endl;
        exit(1);
    }
    else {
        if(VERBOSE>1) cout << "\t\t\t.";
    }
    string kernelname = filename.substr(11,filename.length() - 14);
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    cl_program clPgrm = clCreateProgramWithSource(ctx, 1,
            (const char **)&source_str, (const size_t *)&source_size, &err);

    print_opencl_error(err);

    err = clBuildProgram( clPgrm, 1, &device, NULL, NULL, NULL );
    assert(err == CL_SUCCESS);

    print_opencl_error(err);

//    kernels[i] = clCreateKernel( clPgrm, kernelname.c_str(), &err);
    kernels[filename.substr(4,filename.length() - 7)] = clCreateKernel( clPgrm, kernelname.c_str(), &err);

    print_opencl_error(err);

    err = clReleaseProgram(clPgrm);

    print_opencl_error(err);
    assert(err == CL_SUCCESS);
}
inline vector<string> glob(const string& pat){
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i)
        ret.push_back(string(glob_result.gl_pathv[i]));
    globfree(&glob_result);
    return ret;
}

void opencl_brain_functions::init_kernels(string filename){
    int i = 0;
    if(filename.length()>0){
        ifstream inf(filename.c_str());
        assert(inf.good());
        while(inf.good()){
            string sourcefile;
            inf >> sourcefile;
            if(sourcefile.length() < 2) break;
            init_kernel(sourcefile);
            i++;
        }
    }
    else{
        vector<string> v = glob("kls/*.cl");
        for(vector<string>::iterator it = v.begin(); it != v.end(); ++it)
            init_kernel(*it);
    }
    assert(kernels.size()>7);
    if(VERBOSE>1) cout << "\t\t\tok" << endl;
}











void opencl_brain_functions::print(Mat S, int p){
    if(VERBOSE>1) cout << "\t\t\tprint(M)" << endl;
    if(DEBUG) {int quit = 0; cin >> quit; assert(quit);};
    if(VERBOSE>1) cout << (void *) S << "\t";
    float mtmp[p + 1];
    opencl_getv(S,mtmp,0, p + 1);
    cout << mtmp[p] << endl;
}
void opencl_brain_functions::print(Mat S, int w, int h,bool l){
    if(VERBOSE>1) cout << "\t\t\tprint(M)" << endl;
    if(DEBUG) {int quit = 0; cin >> quit; assert(quit);};
    if(VERBOSE>1) cout << (void *) S << "\t";
    if(l){
            long mtmp[w*h];
            opencl_getv(S,mtmp,0,w*h);
            for(int i = 0; i < h; ++i){
                if(VERBOSE>1) cout << mtmp[i*w];
                for(int j = 1; j < w; ++j){
                    if(VERBOSE>1) cout << "\t\t\t\t" << mtmp[i*w + j];
                }
                if(VERBOSE>1) cout << endl;
                if(i + 1 != h)
                    if(VERBOSE>1) cout << "\t\t\t\t\t";
            }
    }else{
        cl_float mtmp[w*h];
        opencl_getv(S,mtmp,0,w*h);
        for(int i = 0; i < h; ++i){
            if(VERBOSE>1) cout << mtmp[i*w];
            for(int j = 1; j < w; ++j){
                if(VERBOSE>1) cout << "\t\t\t\t" << mtmp[i*w + j];
            }
            if(VERBOSE>1) cout << endl;
            if(i + 1 != h)
                if(VERBOSE>1) cout << "\t\t\t\t\t";
        }
    }
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

