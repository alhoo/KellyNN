/** Working **/

__kernel void find_winning_neurons(
    __global float *NW, 
    __global const float *SW, 
    __global const float *SBET0, 
    __global const float *SBET1, 
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[0] && i > 0){
        NW[i] = SW[i]*fabs(SBET1[i] - SBET0[i]) - 0.1;
        for(int j = 1; j < w[0]; ++j){
            NW[i] = NW[i] + SW[i + j*w[0]]*fabs(SBET1[i + j*w[0]] - SBET0[i + j*w[0]]);
//                NW[i] = NW[i] + SW[i + j*w[0]];
        }
        NW[i] = sign(NW[i]);
    }
}
