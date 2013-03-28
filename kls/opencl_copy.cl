__kernel void copy(
    __global float *A, 
    __global const float *B, 
    __global const long *w
){
    int i   = get_global_id(0);
    if( i < w[1]){
        //S[i] = STMP[i];
        A[i] = B[i];
    }
}
