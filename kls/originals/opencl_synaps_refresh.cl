__kernel void synaps_refresh(
    __global float *SP0,
    __global const float *SP01,
    __global const float *SP00,
    __global const float *NP,
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[1]){
        SP0[i] = SP01[i]*(NP[i%w[0]]) + SP00[i]*(1 - NP[i%w[0]]);
    }
}
