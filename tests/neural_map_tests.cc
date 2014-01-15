#include "neural_map_tests.h"
#ifndef NBSIZE
#define NBSIZE (2)
#endif
#ifndef SBSIZE
#define SBSIZE (NBSIZE*NBSIZE)
#endif
#ifndef VERBOSE
#define VERBOSE (0)
#endif
#define NI (3)
#define NO (3)


namespace{

/*
class neuron_block_test : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        neuron_block_test():NM(8,4) {
            N = new NeuronBlock();
            // You can do set-up work for each test here.
        }

        virtual ~neuron_block_test() {
        // You can do clean-up work that doesn't throw exceptions here.
            delete N;
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
        neural_map NM;
        NeuronBlock *N;
};

TEST_F(neuron_block_test, testSize) {
    EXPECT_EQ((size_t)NBSIZE*7*sizeof(float), N->size());
}

TEST_F(neuron_block_test, setStateTest) {
    statetype s = 3;
    
    EXPECT_FLOAT_EQ(8.96e-08, N->setState(s));
}

TEST_F(neuron_block_test, payTest) {
    //N->pay(0,10.0f);
    EXPECT_EQ(1, 1);
}

class synaps_block_test : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        synaps_block_test():NM(8,4) {
            // You can do set-up work for each test here.
            N = new NeuronBlock();
            S = new SynapsBlock(N,NBSIZE,NBSIZE);
        }

        virtual ~synaps_block_test() {
        // You can do clean-up work that doesn't throw exceptions here.
            delete S;
            delete N;
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
        neural_map NM;
        SynapsBlock *S;
        NeuronBlock *N;
};

TEST_F(synaps_block_test, payTest) {
    
    S->update();
    EXPECT_EQ(1, 1);
}
TEST_F(synaps_block_test, updateTest) {
    
    S->update();
    EXPECT_EQ(1, 1);
}

class neurons_test : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        neurons_test():NM(8,4) {
            // You can do set-up work for each test here.
            N = new Neurons();
        }

        virtual ~neurons_test() {
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
        neural_map NM;
        Neurons *N;
};

class synapses_test : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        synapses_test():NM(8,4) {
            // You can do set-up work for each test here.
            N = new Neurons();
            S = new Synapses(N);
        }

        virtual ~synapses_test() {
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
        neural_map NM;
        Neurons *N;
        Synapses *S;
};
*/
class NMT : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        NMT():nm(1,NI,NO) {
            // You can do set-up work for each test here.
            for(int i = 0; i < NI; ++i) I[i] = 0;
            for(int i = 0; i < NO; ++i) O[i] = 0;
        }

        virtual ~NMT() {
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
        neural_map nm;
        float I[NI];
        float O[NO];
};

/*
    TEST_F(NMT, UpdateTest) {
        nm.U(1.0);
    }

    TEST_F(NMT, OutTest) {
        nm.O(O);
    }

    TEST_F(NMT, InTest) {
        nm.I(I);
    }

    TEST_F(NMT, SimpleNetCreationTest) {
        for(int i = 0; i < NBSIZE; ++i){
            nm.killN(i);
            for(int j = 0; j < NBSIZE; ++j)
                nm.killS(position(i,j));
        }
        //nm.stateN(0);
        //nm.stateN(1);
        //nm.stateS(position(1,0));
        nm.initN(0);
        nm.initN(1);
        nm.initS(position(1,0));
        nm.setIO(1,1);
        //nm.stateN(0);
        //nm.stateN(1);
        //nm.stateS(position(1,0));

        
        EXPECT_FLOAT_EQ(2*NEURONINITBAL + SYNAPSINITBAL, nm.Bal());
    }

*/
  TEST_F(NMT, conservation_of_value) {
        float I[3],O[3];
        nm.U(0.05,10);
        float initbal = nm.Bal();

        I[0] = 0.00000001;
        I[1] = 1.0 - 0.00000001;
        I[2] = 1.0 - 0.00000001;
        I[3] = 1.0 - 0.00000001;

        nm.I(I);
        ASSERT_NEAR(initbal, nm.Bal(), 1.0);
        nm.U(0.05,10);
        ASSERT_NEAR(initbal, nm.Bal(), 1.0);
        nm.R(10);
        ASSERT_NEAR(initbal + 10.0, nm.Bal(), 1.0);
  }
  TEST_F(NMT, NeuronSharedWin) {
      float I[3],O[3];
      I[0] = 0.00000001;
      I[1] = 1.0 - 0.00000001;
      I[2] = 1.0 - 0.00000001;
      I[3] = 1.0 - 0.00000001;
//      cerr << "INPUT" << endl << "--------------" << endl;
      nm.I(I);
//      nm.print();
//      cerr << "UPDATE" << endl << "--------------" << endl;
      nm.U(0.05,10);
//      nm.print();
//      cerr << "VALUE" << endl << "--------------" << endl;
      nm.R(10);
//      nm.print();
//      cerr << "DONE" << endl << "--------------" << endl;
      float VAL[256];
      nm.getX('B',VAL);
      uint32_t bc = nm.getbc();
      cout << setprecision(5);
      for(size_t x = 0; x < bc*NBSIZE; ++x)
          cout << setw(11) << VAL[x];
      cout << endl << "-----------------------------------------------------" << endl;
      for(size_t x = 0; x < bc*NBSIZE; ++x){
          for(size_t y = 0; y < bc*NBSIZE; ++y)
              cout << setw(11) << VAL[bc*NBSIZE + x + y*bc*NBSIZE];
          cout << endl;
      } 

  }
  TEST_F(NMT, NeuronLooseFunctionTest) {
        float I[2],O[2];
        for(int i = 0; i < 15; ++i){
            I[0] = 0.1;
            I[1] = (i%2)*0.8 + 0.1;
            nm.I(I);
            nm.U(0.05);
            nm.O(O);
            float pay = -15;
            if((O[0]-0.5)*((i%2)*1.0 - 0.5) < 0)
                pay += 10;
            if((O[1]-0.5)*(0.5 - (i%2)*1.0) < 0)
                pay += 10;
            nm.R(pay);
        }

        I[0] = 0.1;
        I[1] = 0.9;
        nm.I(I);
        nm.U(0.05);
        nm.O(O);
        EXPECT_LT((O[0]-0.5)*(1.0 - 0.5),0);
        EXPECT_LT((O[1]-0.5)*(0.5 - 1.0),0);

        I[1] = 0.1;
        nm.I(I);
        nm.U(0.05);
        nm.O(O);
        EXPECT_LT((O[0]-0.5)*(-0.5),0);
        EXPECT_LT((O[1]-0.5)*(0.5),0);

  }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
