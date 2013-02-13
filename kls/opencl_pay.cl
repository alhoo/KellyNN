/**
This function is written according to the matrix brain tag-2.00.00

**/
__kernel void pay(
    __global float * SBET0,
    __global float * SBET1,
    __global float * STMP,
    __global float * NBET0,
    __global float * NBET1,
    __global float * SBAL,
    __global float * NBAL,
    __global float * SW,
    __global float * NW,
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[0]){
        int iw   = i*w[0];
        float NBETSUM   = NBET0[i] + NBET1[i];
        float NWIN      = NBETSUM + NW[i]*fabs(NBET1[i] - NBET0[i]);
        for(int j = 0; j < w[0]; j++){
            float SBETSUM   = SBET0[iw + j] + SBET1[iw + j];
            float C         = (tanh(0.001*(SBAL[iw + j] - 3*w[3])) + 1)*(SBAL[iw + j]/8);
            STMP[iw + j]    = NBETSUM*(SBETSUM + SW[iw + j]*fabs(SBET1[iw + j] - SBET0[iw + j]))/(0.01 + NWIN);
            SBAL[iw + j]    = SBAL[iw + j] + STMP[iw + j] - SBETSUM - C;
            NBAL[i]         = NBAL[i] + C;
            SBET1[iw + j]   = 0;
            SBET0[iw + j]   = 0;
        }
        if(i > 1){
            NBETSUM  = (tanh(0.001*(NBAL[i] - 3*w[2])) + 1.0)*(NBAL[i])/(16.0);
            NBAL[i]  = NBAL[i] - 2*NBETSUM;
            NBET1[i] = NBETSUM;
            NBET0[i] = NBETSUM;
        }
    }
}
