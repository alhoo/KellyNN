/**
This function is written according to the matrix brain tag-2.00.00
but split into two pieces to support partial matrix updates
**/
__kernel void npay(
    __global float * NBET0,
    __global float * NBET1,
    __global float * NBAL,
    __global float * NW,
    __global float * NBETSUM,
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[0]){

        NBET0[i]        = NBET0[i] + 0.001;
        NBET1[i]        = NBET1[i] + 0.001;
        NBETSUM[i]      = NBET0[i] + NBET1[i];

        NBAL[i]         = NBAL[i] + NBETSUM[i];
        //NBETSUM[i]      = NBETSUM[i] + (tanh((NBAL[i] - 2*w[2])*2/w[2]) + 1.0)*(NBAL[i])/(8);

        NW[i]           = NW[i]*sign(NBET1[i]-NBET0[i]);

    }
}
