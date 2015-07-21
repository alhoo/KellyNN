/** Working **/

__kernel void bet_sum(
    __global float *NBET, 
    __global const float *SBET, 
    __global const long *w
){
    int i   = get_global_id(0);
    if( i < w[0] ){
        int iw  = i*w[0];
        for(int j = iw; j < iw + w[0]; ++j){
            NBET[i]    += SBET[j];
        }
    }
}
