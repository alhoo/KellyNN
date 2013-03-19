__kernel void synaps_learn(
__global float *SP,
__global const float *SBET1,
__global const float *SBET0,
__global const float *SW,
__global const float *NBET1,
__global const float *NBET0,
__global const float *n,
__global const long *w
){
    int i   = get_global_id(0);
    if(i < w[1]){
        if(SW[i]*sign(SBET1[i] - SBET0[i])*sign(NBET1[i%w[0]] - NBET0[i%w[0]]) > 0)
            SP[i] = (n[0]*SP[i] + 1)/(n[0] + 1);
        else
            SP[i] = (n[0]*SP[i])/(n[0] + 1);
        //SP[i] = (n[0]*SP[i] + SW[i]*sign(SBET1[i] - SBET0[i])*sign(NBET1[i%w[0]] - NBET0[i%w[0]]) + 1)/(n[0] + 2);
        //SP[i] = (n[0]*SP[i] + SW[i]*sign(SBET1[i] - SBET0[i] + 0.001)*sign(NBET1[i%w[0]] - NBET0[i%w[0]] + 0.001) + 1)/(n[0] + 2);
    }
}
