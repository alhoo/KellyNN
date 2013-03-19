
__kernel void synaps_learn_negation(
__global float *SP1,
__global float *SP0,
__global const long *w
){
    int i   = get_global_id(0);
    if(i < w[1]){
        SP0[i] = 1.0 - SP1[i];
        //SP[i] = (n[0]*SP[i] + SW[i]*sign(SBET1[i] - SBET0[i])*sign(NBET1[i%w[0]] - NBET0[i%w[0]]) + 1)/(n[0] + 2);
        //SP[i] = (n[0]*SP[i] + SW[i]*sign(SBET1[i] - SBET0[i] + 0.001)*sign(NBET1[i%w[0]] - NBET0[i%w[0]] + 0.001) + 1)/(n[0] + 2);
    }
}
