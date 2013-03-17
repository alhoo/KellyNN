/** Working **/

__kernel void find_winning_neurons(
    __global float *NW, 
    __global const float *SW, 
    __global const long *w
){
    int i   = get_global_id(0);
    if(i < w[0] && i > 0){
        if(i < 3)   /*FIXME hack*/
            NW[i] = 1.0;
        else {
            NW[i] = SW[i] - 0.1;
            for(int j = 1; j < w[0]; ++j){
                NW[i] = NW[i] + SW[i + j*w[0]];
            }
            NW[i] = sign(NW[i]);
        }
    }
}
