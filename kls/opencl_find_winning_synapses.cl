/**
A synaps is a winning synaps if it has enforced a winning 
neuron to the decision that made it win.
*/

__kernel void find_winning_synapses(
    __global const float *SBET0,
    __global const float *SBET1,
    __global const float *NBET0,
    __global const float *NBET1,
    __global const float *NW,
    __global float *SW,
    __global const long *w)
{
    int i   = get_global_id(0);
    if(i < w[1]){
        if(NW[i%w[0]]*sign(SBET0[i] - SBET1[i] + 0.00001)*sign(NBET0[i%w[0]] - NBET1[i%w[0]] + 0.00001) == 1.0){
            SW[i] = 1.0;
        }
        else{
            SW[i] = -1.0;
        }
/**
        SW[i] = SW[i] + NW[i%w[0]]*sign(SBET0[i] - SBET1[i] + 0.00001)*sign(NBET0[i%w[0]] - NBET1[i%w[0]] + 0.00001);
        SW[i] = SW[i] + tanh(NW[i%w[0]]*((SBET0[i] - SBET1[i])*(NBET0[i%w[0]] - NBET1[i%w[0]]) + 1)); 
**/
    }
}
