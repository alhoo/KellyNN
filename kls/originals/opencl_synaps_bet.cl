__kernel void synaps_bet(
    __global float *SBET1,  
    __global const float *SBAL,
    __global const float *SP1,
    __global const float *SP,
    __global const long *w
){
    int i    	= get_global_id(0);
    if(i < w[1]){
        SBET1[i]    = SBAL[i]*SP1[i]/(30.0 - 28.0*sqrt(SP[i]) );
/**        SBET1[i]    = SBAL[i]*SP1[i]/(2.23405518909087 - 7.91829070671564*log(SP[i]+0.03) );**/
    }
}
