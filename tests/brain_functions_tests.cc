#include "brain_functions_tests.h"
#ifndef NBSIZE
#define NBSIZE (16)
#endif
#ifndef SBSIZE
#define SBSIZE (NBSIZE*NBSIZE)
#endif


namespace{


class OBFT : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        OBFT() {
            bf = new opencl_brain_functions(NBSIZE,8,4);
            FNBET0 = bf->gpu_malloc(FNBET0,NBSIZE,0);
            FNBET1 = bf->gpu_malloc(FNBET1,NBSIZE,0);
            FNBAL = bf->gpu_malloc(FNBAL,NBSIZE,0);
            FNW = bf->gpu_malloc(FNW,NBSIZE,0);
            FN5 = bf->gpu_malloc(FN5,NBSIZE,0);
            FN6 = bf->gpu_malloc(FN6,NBSIZE,0);
            FNP = bf->gpu_malloc(FNP,NBSIZE,0);

            TNBET0 = bf->gpu_malloc(TNBET0,NBSIZE,0);
            TNBET1 = bf->gpu_malloc(TNBET1,NBSIZE,0);
            TNBAL = bf->gpu_malloc(TNBAL,NBSIZE,0);
            TNW = bf->gpu_malloc(TNW,NBSIZE,0);
            TN5 = bf->gpu_malloc(TN5,NBSIZE,0);
            TN6 = bf->gpu_malloc(TN6,NBSIZE,0);
            TNP = bf->gpu_malloc(TNP,NBSIZE,0);

            SBET0 = bf->gpu_malloc(SBET0,SBSIZE,0);
            SBET1 = bf->gpu_malloc(SBET1,SBSIZE,0);
            STMP = bf->gpu_malloc(STMP,SBSIZE,0);
            SBAL = bf->gpu_malloc(SBAL,SBSIZE,0);
            SW = bf->gpu_malloc(SW,SBSIZE,0);
            S6 = bf->gpu_malloc(S6,SBSIZE,0);
            SP0 = bf->gpu_malloc(SP0,SBSIZE,0);
            SP01 = bf->gpu_malloc(SP01,SBSIZE,0);
            SP00 = bf->gpu_malloc(SP00,SBSIZE,0);
            SP1 = bf->gpu_malloc(SP1,SBSIZE,0);
            SP11 = bf->gpu_malloc(SP11,SBSIZE,0);
            SP10 = bf->gpu_malloc(SP10,SBSIZE,0);
            SINFO = bf->gpu_malloc(SINFO,SBSIZE,0);
            SP = bf->gpu_malloc(SP,SBSIZE,0);
            for(int i = 0; i < NBSIZE; ++i) a[i] = 0;
            for(int i = 0; i < SBSIZE; ++i) b[i] = 0;
            // You can do set-up work for each test here.
        }

        virtual ~OBFT() {
            bf->gpu_free(TNBET0);
            bf->gpu_free(TNBET1);
            bf->gpu_free(TNBAL);
            bf->gpu_free(TNW);
            bf->gpu_free(TN5);
            bf->gpu_free(TN6);
            bf->gpu_free(TNP);
            bf->gpu_free(FNBET0);
            bf->gpu_free(FNBET1);
            bf->gpu_free(FNBAL);
            bf->gpu_free(FNW);
            bf->gpu_free(FN5);
            bf->gpu_free(FN6);
            bf->gpu_free(FNP);
            bf->gpu_free(SBET0);
            bf->gpu_free(SBET1);
            bf->gpu_free(STMP);
            bf->gpu_free(SBAL);
            bf->gpu_free(SW);
            bf->gpu_free(S6);
            bf->gpu_free(SP0);
            bf->gpu_free(SP01);
            bf->gpu_free(SP00);
            bf->gpu_free(SP1);
            bf->gpu_free(SP11);
            bf->gpu_free(SP10);
            bf->gpu_free(SINFO);
            bf->gpu_free(SP);
        // You can do clean-up work that doesn't throw exceptions here.
        }

        // If the constructor and destructor are not enough for setting up
        // and cleaning up each test, you can define the following methods:

        virtual void SetUp() {
        // Code here will be called immediately after the constructor (right
        // before each test).
        }

        virtual void TearDown() {
            // Code here will be called immediately after each test (right
            // before the destructor).
        }

        // Objects declared here can be used by all tests in the test case for Foo.
        float a[NBSIZE];
        float b[SBSIZE];
        opencl_brain_functions *bf;
        Col TNBET0, TNBET1, TNBAL;
        Col TNW, TN5, TN6;
        Col TNP;
        Col FNBET0, FNBET1, FNBAL;
        Col FNW, FN5, FN6;
        Col FNP;
        Mat SBET0, SBET1, STMP;
        Mat SBAL, SW, S6;
        Mat SP0;
        Mat SP01;
        Mat SP00;
        Mat SP1;
        Mat SP11;
        Mat SP10;
        Mat SINFO;
        Mat SP;

};

TEST_F(OBFT, opencl_pay_0) {
    bf->opencl_setv(FNBET0,a,0,NBSIZE);
    bf->opencl_setv(FNBET1,a,0,NBSIZE);
    bf->opencl_setv(FNBAL,a,0,NBSIZE);
    bf->opencl_setv(FNW,a,0,NBSIZE);
    bf->opencl_setv(SBET0,b,0,SBSIZE);
    bf->opencl_setv(SBET1,b,0,SBSIZE);
    bf->opencl_setv(STMP,b,0,SBSIZE);
    bf->opencl_setv(SBAL,b,0,SBSIZE);
    bf->opencl_setv(SW,b,0,SBSIZE);
    bf->opencl_pay(SBET0,SBET1,FNBET0,FNBET1,SBAL,FNBAL,SW,FNW);
}
TEST_F(OBFT, opencl_pay_1) {
    for(int i = 0; i < NBSIZE; ++i) a[i] = 1;
    for(int i = 0; i < SBSIZE; ++i) b[i] = 1;
    bf->opencl_setv(FNBET0,a,0,NBSIZE);
    bf->opencl_setv(FNBET1,a,0,NBSIZE);
    bf->opencl_setv(FNBAL,a,0,NBSIZE);
    bf->opencl_setv(FNW,a,0,NBSIZE);
    bf->opencl_setv(SBET0,b,0,SBSIZE);
    bf->opencl_setv(SBET1,b,0,SBSIZE);
    bf->opencl_setv(STMP,b,0,SBSIZE);
    bf->opencl_setv(SBAL,b,0,SBSIZE);
    bf->opencl_setv(SW,b,0,SBSIZE);
    bf->opencl_pay(SBET0,SBET1,FNBET0,FNBET1,SBAL,FNBAL,SW,FNW);
}
TEST_F(OBFT, opencl_pay_large) {
    for(int i = 0; i < NBSIZE; ++i) a[i] = 10e10;
    for(int i = 0; i < SBSIZE; ++i) b[i] = 10e10;
    bf->opencl_setv(FNBET0,a,0,NBSIZE);
    bf->opencl_setv(FNBET1,a,0,NBSIZE);
    bf->opencl_setv(FNBAL,a,0,NBSIZE);
    bf->opencl_setv(FNW,a,0,NBSIZE);
    bf->opencl_setv(SBET0,b,0,SBSIZE);
    bf->opencl_setv(SBET1,b,0,SBSIZE);
    bf->opencl_setv(STMP,b,0,SBSIZE);
    bf->opencl_setv(SBAL,b,0,SBSIZE);
    bf->opencl_setv(SW,b,0,SBSIZE);
    bf->opencl_pay(SBET0,SBET1,FNBET0,FNBET1,SBAL,FNBAL,SW,FNW);
}

TEST_F(OBFT, opencl_pay_conservation_of_value) {
  /** INIT **/
    for(int i = 0; i < NBSIZE; ++i) a[i] = 1000.0;
    bf->opencl_setv(FNBAL,a,0,NBSIZE);
    
    for(int i = 0; i < NBSIZE; ++i) a[i] = 1000.0;
    bf->opencl_setv(TNBAL,a,0,NBSIZE);

    for(int i = 0; i < SBSIZE; ++i) b[i] = 0;
    for(int i = 0; i < SBSIZE; ++i) b[i] = 1000.0;
    bf->opencl_setv(SBAL,a,0,SBSIZE);

  /** ::I() **/
    float *A = new float[NBSIZE];
    for(int i = 0; i < NBSIZE; ++i) A[i] = 1;
    bf->opencl_setv(FNW,A,1,2);

    for(int i = 0; i < NBSIZE; ++i) a[i] = 0;
    a[1] = 0.8;
    a[2] = 0.2;
    bf->opencl_setv(FNP,a,0,NBSIZE);

  /** ::U() **/
    bf->opencl_synaps_refresh(SP0,SP01,SP00,FNP);
    bf->opencl_synaps_refresh(SP1,SP11,SP10,FNP);

    bf->opencl_synaps_bet(SBET1,SBAL,SP1,SP);
    bf->opencl_synaps_bet(SBET0,SBAL,SP0,SP);


    for(int i = 0; i < SBSIZE; ++i) b[i] = 0.0;
    b[2] = 10.0;
    bf->opencl_setv(SBET0,b,0,SBSIZE);
    b[2] = 0.0;
    b[1] = 10.0;
    bf->opencl_setv(SBET1,b,0,SBSIZE);
    b[1] = 0.0;
    
    bf->opencl_bet_sum(TNBET1,SBET1);
    bf->opencl_bet_sum(TNBET0,SBET0);

    for(int i = 0; i < SBSIZE; ++i) b[i] = 0;
    bf->opencl_setv(STMP,b,0,SBSIZE);
    float initbal = bf->opencl_get_bal(FNBAL,SBAL,FNBET0,FNBET1)
                + bf->opencl_get_bal(TNBAL,STMP,TNBET0,TNBET1);

    bf->opencl_neuron_refresh(TNBET1,TNBET0,TNP);

  /** ::R() **/
    a[0] = 1000.0;
    bf->opencl_setv(FNP,a,0,1);
    bf->opencl_pay_neuron(FNBAL,FNW,0,a[0]);

    for(int i = 0; i < 3; ++i){
        bf->opencl_find_winning_synapses(SBET0,SBET1,TNBET0,TNBET1,TNW,SW);
        bf->wait();
        bf->opencl_find_winning_neurons(TNW,SW,SBET0,SBET1);
        bf->wait();
    }
    bf->opencl_synaps_learn(SP00,SP01,SP10,SP11,SBET1,SBET0,SW,FNBET0,FNBET1,SINFO);

    bf->opencl_synaps_learn2(SP,SW,SINFO);

    bf->opencl_update_synaps_info(SINFO);
 
    bf->opencl_pay(SBET0,SBET1,TNBET0,TNBET1,SBAL,TNBAL,SW,TNW);
    
    bf->opencl_getv(SBAL,b,0,SBSIZE);
    bf->opencl_getv(FNBAL,a,0,NBSIZE);
    
//    float currentbal = bf->opencl_get_bal(FNBAL,TNBAL,SBAL,FNBET0,FNBET1,TNBET0,TNBET1);
    for(int i = 0; i < SBSIZE; ++i) b[i] = 0;
    bf->opencl_setv(STMP,b,0,SBSIZE);
    float currentbal = bf->opencl_get_bal(FNBAL,SBAL,FNBET0,FNBET1)
                + bf->opencl_get_bal(TNBAL,STMP,TNBET0,TNBET1);

//    EXPECT_FLOAT_EQ(initbal + 1000.0, currentbal);
    ASSERT_NEAR(initbal + 1000.0, currentbal, 1.0);
}
TEST_F(OBFT, setv_getv_test) {
    for(int i = 0; i < NBSIZE; ++i) a[i] = 0.45*i;
    EXPECT_FLOAT_EQ(a[NBSIZE/2], 0.45*(NBSIZE/2));

    bf->opencl_setv(FNBET0,a,0,NBSIZE);

    for(int i = 0; i < NBSIZE; ++i) a[i] = 0;
    EXPECT_FLOAT_EQ(a[NBSIZE/2], 0 );

    bf->opencl_getv(FNBET0,a,0,NBSIZE);

    for(int i = 0; i < NBSIZE; ++i) EXPECT_FLOAT_EQ(a[i], 0.45*(i));
}
TEST_F(OBFT, opencl_synaps_bet) {
    bf->opencl_synaps_bet(SBET0,SBET1,STMP,SBAL);
}
TEST_F(OBFT, opencl_bet_sum) {
    for(int i = 0; i < SBSIZE; ++i) b[i] = 1;
    bf->opencl_setv(SBET0,b,0,SBSIZE);
    bf->opencl_bet_sum(FNBET0,SBET0);
    bf->opencl_getv(FNBET0,a,0,NBSIZE);
    for(int i = 0; i < NBSIZE; ++i) EXPECT_FLOAT_EQ(a[i], NBSIZE);
}
TEST_F(OBFT, opencl_set) {
    int      i = 0;
    cl_float v = 5.0f;
    bf->opencl_set(FNBET0,i,v);
}
TEST_F(OBFT, opencl_synaps_refresh) {
    bf->opencl_synaps_refresh(SBET0,SBET1,STMP,FNBET0);
}
TEST_F(OBFT, opencl_neuron_refresh) {
    bf->opencl_neuron_refresh(FNBET0,FNBET1,FNBAL);
}
TEST_F(OBFT, opencl_find_winning_synapses) {
    bf->opencl_find_winning_synapses(SBET0,SBET1,FNBET0,FNBET1,FNBAL,SBET0);
}
TEST_F(OBFT, opencl_find_winning_neurons) {
    bf->opencl_find_winning_neurons(TNW,SW,SBET0,SBET1);
}
TEST_F(OBFT, opencl_update_synaps_info) {
    bf->opencl_update_synaps_info(FNBET0);
}
/*
TEST_F(OBFT, opencl_synaps_learn) {
    bf->opencl_synaps_learn(SBET0,SBET1,STMP,SBAL,FNBET0,FNBET1,FNBAL);
}
*/
TEST_F(OBFT, opencl_synaps_learn2) {
    bf->opencl_synaps_learn2(SBET0,SBET1,FNBET0);
}
/*
TEST_F(OBFT, opencl_pay_neuron) {
    int     i = 0;
    float   f = 10.0;
    bf->opencl_pay_neuron(FNBET0,FNBET1,SBET0,i,f);
}
*/
TEST_F(OBFT, opencl_synaps_die) {
    long    s = 0;
    long    v = 0;
    long    l = 0;
    bf->opencl_synaps_die(SBET0,SBET1,s, v, l);
}
TEST_F(OBFT, opencl_fill) {
    int         l = SBSIZE;
    cl_float    v = 5.0;
    bf->opencl_fill(SBET0,v,l);
    bf->opencl_getv(SBET0,b,0,SBSIZE);
    for(int i = 0; i < l; ++i) EXPECT_FLOAT_EQ(5.0f,b[i]);
}
void        printv(float *a , int c){
    cout << a[0];
    for(int i = 1; i < c; ++i) cout << ", " << a[i];
    cout << endl;
}
/*
TEST_F(OBFT, opencl_test_xor) {
    srand ( time(FNULL) );

    float failcost = -900.0f;

    float vprev = (rand() % 8 + 1)*0.1;
    for(int i = 0; i < NBSIZE; ++i) a[i] = 10000.0;
    bf->opencl_setv(FNBAL,a,0,NBSIZE);

    for(int i = 0; i < SBSIZE; ++i) b[i] = 1000.0;
    bf->opencl_setv(SBAL,a,0,SBSIZE);

    for(int i = 0; i < SBSIZE; ++i) b[i] = 0;
    bf->opencl_setv(STMP,b,0,SBSIZE);
    float initbal = bf->opencl_get_bal(FNBAL,SBAL,FNBET0,FNBET1)
                 +  bf->opencl_get_bal(TNBAL,STMP,TNBET0,TNBET1);

    for(int n = 0; n < 1000; ++n){
        float v = (rand() % 8 + 1)*0.1;
        for(int i = 0; i < NBSIZE - 4; ++i) a[i] = 0;
        a[1] = v;
//        a[2] = 1.0 - v;

        bf->opencl_setv(FNP,a,0,NBSIZE - 4);

        bf->opencl_synaps_refresh(SP0,SP01,SP00,FNP);
        bf->opencl_synaps_refresh(SP1,SP11,SP10,FNP);

        bf->opencl_synaps_bet(SBET1,SBAL,SP1,SP);
        bf->opencl_synaps_bet(SBET0,SBAL,SP0,SP);

        bf->opencl_bet_sum(FNBET1,SBET1);
        bf->opencl_bet_sum(FNBET0,SBET0);

        bf->opencl_neuron_refresh(FNBET1,FNBET0,FNP);

        bf->opencl_getv(FNP,a,0,NBSIZE);


        a[0] = failcost;
        a[0] += 40.0f*0.05f/(0.05f + fabs(a[8] - v/2 + 0.4));
        a[0] += 300.0f*0.05f/(0.05f + fabs(a[9] - v));
        a[0] += 4000.0f*0.05f/(0.05f + fabs(a[10] + vprev));
        a[0] += 1000.0f*0.05f/(0.05f + fabs(a[11] - v*v));

        bf->opencl_pay_neuron(FNBAL,FNW,SW,0,a[0]);

        for(int i = 0; i < 3; ++i){
            bf->opencl_find_winning_synapses(SBET0,SBET1,FNBET0,FNBET1,FNW,SW);
            bf->wait();
            bf->opencl_find_winning_neurons(FNW,SW);
            bf->wait();
        }



        bf->opencl_synaps_learn(SP11,SBET1,SBET0,SW,FNBET1,FNBET0,SINFO);
        bf->opencl_synaps_learn(SP10,SBET1,SBET0,SW,FNBET0,FNBET1,SINFO);
        bf->opencl_synaps_learn(SP01,SBET0,SBET1,SW,FNBET1,FNBET0,SINFO);
        bf->opencl_synaps_learn(SP00,SBET0,SBET1,SW,FNBET0,FNBET1,SINFO);

        bf->opencl_synaps_learn2(SP,SW,SINFO);

        bf->opencl_update_synaps_info(SINFO);
     
        bf->opencl_pay(SBET0,SBET1,FNBET0,FNBET1,SBAL,FNBAL,SW,FNW);
        
        vprev = v;
    }
    int nw = 0;
    for(int n = 0; n < 10; ++n){
        float v = (rand() % 8 + 1)*0.1;
        for(int i = 0; i < NBSIZE; ++i) a[i] = 0;
        a[1] = v;
//        a[2] = 1.0 - v;
        cout << v << " =>\t(";
        bf->opencl_setv(FNP,a,0,NBSIZE - 4);

        bf->opencl_synaps_refresh(SP0,SP01,SP00,FNP);
        bf->opencl_synaps_refresh(SP1,SP11,SP10,FNP);

        bf->opencl_synaps_bet(SBET1,SBAL,SP1,SP);
        bf->opencl_synaps_bet(SBET0,SBAL,SP0,SP);

        bf->opencl_bet_sum(FNBET1,SBET1);
        bf->opencl_bet_sum(FNBET0,SBET0);

        bf->opencl_neuron_refresh(FNBET1,FNBET0,FNP);

        bf->opencl_getv(FNP,a,0,NBSIZE);
        cout << a[8] <<  "," << a[9] <<  "," << a[10] <<  "," << a[11] << ")" << endl;
        cout << "\t" << fabs(a[8] - v/2 + 0.4) <<  "," << fabs(a[9] - v) <<  "," << fabs(a[10] + vprev) <<  "," << fabs(a[11] - v*v) << ")" << endl;
        a[0] = failcost;
        a[0] += 40.0f*0.05f/(0.05f + fabs(a[8] - v/2 + 0.4));
        a[0] += 300.0f*0.05f/(0.05f + fabs(a[9] - v));
        a[0] += 4000.0f*0.05f/(0.05f + fabs(a[10] + vprev));
        a[0] += 1000.0f*0.05f/(0.05f + fabs(a[11] - v*v));
        if(a[0]>0) nw++;
        bf->opencl_pay_neuron(FNBAL,FNW,SW,0,a[0]);

        for(int i = 0; i < 3; ++i){
            bf->opencl_find_winning_synapses(SBET0,SBET1,FNBET0,FNBET1,FNW,SW);
            bf->wait();
            bf->opencl_find_winning_neurons(FNW,SW);
            bf->wait();
        }
        bf->opencl_getv(FNW,a,0,NBSIZE);
        printv(a,NBSIZE);

        bf->opencl_synaps_learn(SP11,SBET1,SBET0,SW,FNBET1,FNBET0,SINFO);
        bf->opencl_synaps_learn(SP10,SBET1,SBET0,SW,FNBET0,FNBET1,SINFO);
        bf->opencl_synaps_learn(SP01,SBET0,SBET1,SW,FNBET1,FNBET0,SINFO);
        bf->opencl_synaps_learn(SP00,SBET0,SBET1,SW,FNBET0,FNBET1,SINFO);

        bf->opencl_synaps_learn2(SP,SW,SINFO);

        bf->opencl_update_synaps_info(SINFO);
     
        bf->opencl_pay(SBET0,SBET1,FNBET0,FNBET1,SBAL,FNBAL,SW,FNW);
        
        vprev = v;
    }

    for(int i = 0; i < SBSIZE; ++i) b[i] = 0;
    bf->opencl_setv(STMP,b,0,SBSIZE);
    float currentbal = bf->opencl_get_bal(FNBAL,SBAL,FNBET0,FNBET1)
                    +  bf->opencl_get_bal(TNBAL,STMP,TNBET0,TNBET1);
    EXPECT_GT(nw,7);
}
*/


}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
