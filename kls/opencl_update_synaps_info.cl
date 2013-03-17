__kernel void update_synaps_info(
    __global float *A
){
    int i   = get_global_id(0);
    if(i < 1){
        A[0] = A[0]+1;
/**
        A[0] = 100;
**/
    }
}
