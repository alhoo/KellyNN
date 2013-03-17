/**
This function is written according to the matrix brain tag-2.00.00

**/
__kernel void pay(
    __global float * SBET0,
    __global float * SBET1,
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



        NBET0[i]        = NBET0[i] + 0.001;
        NBET1[i]        = NBET1[i] + 0.001;
        float NBETSUM   = NBET0[i] + NBET1[i];

        NBAL[i]         = NBAL[i] + NBETSUM;
        NBETSUM         = NBETSUM + (tanh((NBAL[i] - 2*w[2])*2/w[2]) + 1.0)*(NBAL[i])/(8);

        float TMP       = NW[i]*sign(NBET1[i]-NBET0[i]);

        for(int j = 0; j < w[0]; j++){
            float C         = -(tanh((SBAL[iw + j] - 2*w[3])*2/w[3]) + 1.0)*(SBAL[iw + j]/(8));
            if(TMP > 0)
                C           = C + NBETSUM*SBET1[iw + j]/NBET1[i];
            else
                C           = C + NBETSUM*SBET0[iw + j]/NBET0[i];

            NBAL[i]         = NBAL[i] - C;
            SBAL[iw + j]    = SBAL[iw + j] - SBET0[iw + j] - SBET1[iw + j] + C;

            SBET1[iw + j]   = 0;
            SBET0[iw + j]   = 0;
        }
    }
}
