__kernel void synaps_learn(
__global float *SP00,
__global float *SP01,
__global float *SP10,
__global float *SP11,
__global const float *SBET0,
__global const float *SBET1,
__global const float *SW,
__global const float *NP,
__global const float *n,
__global const long *w
){
    int i   = get_global_id(0);
    if(i < w[1]){
//        if(NBET0[i%w[0]] > NBET1[i%w[0]]){
        if(NP[i%w[0]] < 0.5){
            if(SW[i]*sign(SBET0[i] - SBET1[i]) > 0){
                SP00[i] = (n[0]*SP00[i] + 1)/(n[0] + 1);
                SP10[i] = (n[0]*SP10[i])/(n[0] + 1);
            }
            else
            {
                SP00[i] = (n[0]*SP00[i])/(n[0] + 1);
                SP10[i] = (n[0]*SP10[i] + 1)/(n[0] + 1);
            }
        }
        else{
            if(SW[i]*sign(SBET1[i] - SBET0[i]) < 0){
                SP01[i] = (n[0]*SP00[i] + 1)/(n[0] + 1);
                SP11[i] = (n[0]*SP10[i])/(n[0] + 1);
            }
            else
            {
                SP01[i] = (n[0]*SP01[i])/(n[0] + 1);
                SP11[i] = (n[0]*SP11[i] + 1)/(n[0] + 1);
            }
        }
/*        
        else{
                SP[i] = (n[0]*SP[i] + 1)/(n[0] + 2);
        }
*/
        //SP[i] = (n[0]*SP[i] + SW[i]*sign(SBET1[i] - SBET0[i])*sign(NBET1[i%w[0]] - NBET0[i%w[0]]) + 1)/(n[0] + 2);
        //SP[i] = (n[0]*SP[i] + SW[i]*sign(SBET1[i] - SBET0[i] + 0.001)*sign(NBET1[i%w[0]] - NBET0[i%w[0]] + 0.001) + 1)/(n[0] + 2);
    }
}
