/**
This function is written according to the matrix brain tag-2.00.00
but split into two pieces to support partial matrix updates
**/
__kernel void pay(
    __global float * SBET0,
    __global float * SBET1,
    __global const float * NBET0,
    __global const float * NBET1,
    __global const float * NBETSUM,
    __global float * SBAL,
    __global float * NBAL,
    __global float * SW,
    __global const float * NW,
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[0]){

        int iw   = i*w[0];

        for(int j = 0; j < w[0]; j++){
            //float C         = -(tanh((SBAL[iw + j] - 2*w[3])*2/w[3]) + 1.0)*(SBAL[iw + j]/(8));
            float C         = 0;
            if(NW[i] > 0)
                C           = C + NBETSUM[i]*SBET1[iw + j]/NBET1[i];
            else
                C           = C + NBETSUM[i]*SBET0[iw + j]/NBET0[i];

            NBAL[i]         = NBAL[i] - C;
            SBAL[iw + j]    = SBAL[iw + j] - SBET0[iw + j] - SBET1[iw + j] + C;

            SBET1[iw + j]   = 0;
            SBET0[iw + j]   = 0;
        }
    }
}
