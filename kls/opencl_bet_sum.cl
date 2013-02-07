/** Working **/

__kernel void bet_sum(
    __global float *NBET, 
    __global const float *SBET, 
    __global const long *w
){
    int i   = get_global_id(0);
    if( i < w[0] ){
        int iw  = i*w[0];
        NBET[i] += SBET[iw];
        for(int j = 1; j < w[0]; ++j){
            NBET[i]    += SBET[iw + j];
        }
    }
}
