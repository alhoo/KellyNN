__kernel void synaps_die(
    __global float *SBAL,
    __global float *SP1,
    __global const long *w
){
    int i    	= get_global_id(0);
    if(i < w[0]){
                SBAL[i] = 0.0;
                SP1[i] = 0.0;
    }
}
