__kernel void synaps_learn2(
    __global float *SP, 
    __global const float *SW, 
    __global const float *n,
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[1]){
        SP[i]    = (n[0]*SP[i] + 1 + SW[i])/(2.0 + n[0]);
    }
}
