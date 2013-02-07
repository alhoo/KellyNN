__kernel void neuron_refresh(
    __global const float *NBET1,
    __global const float *NBET0,
    __global float *NP,
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[0]){
        if(NBET1[i] > 0 || NBET0[i] > 0){
            NP[i]    = ((NBET1[i] + 1)/(NBET1[i] + NBET0[i] + 2));
        }
    }
}
