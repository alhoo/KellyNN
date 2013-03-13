__kernel void synaps_die(
    __global float *SBAL,
    __global float *SP1,
    __global const long *w
){
    int i    	= get_global_id(0) + w[6];
    if(i < w[1]){
        if(w[7] > 0){
            if(i<w[8]){
                SBAL[i] = 0.0;
                SP1[i] = 0.0;
            }   
        }
        else{
            if(i%w[0] == w[6]){
                SBAL[i] = 0.0;
                SP1[i] = 0.0;
            }   
        }        
/**        SBET1[i]    = SBAL[i]*SP1[i]/(2.23 - 7.91*log(SP[i]+0.03) );**/
    }
}
