#include "neural_map_tests.h"
#include <vector>
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

using namespace std;

namespace{

    void printBlock(vector<float> Av){
      int i = 0;
      for (auto t: Av){
          cout << t;
          if(i++%3==2) cout << endl;
          else cout << " ";
      }
    }
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
  TEST_F(NMT, construct_init_map) {
        string map = nm.write_map();
        string expected_map = "0111\n0000\n0101\n0101\n";
        ASSERT_EQ(expected_map,map);
        //ASSERT_STREQ(expected_map.c_str(),map.c_str());
  }
  TEST_F(NMT, conservation_of_value) {
        float I[3],O[3];
        nm.U(-1);
        float initbal = nm.Bal();

        I[0] = 0.00000001;
        I[1] = 1.0 - 0.00000001;
        I[2] = 1.0 - 0.00000001;
   //     I[3] = 1.0 - 0.00000001;

        nm.I(I);
        ASSERT_NEAR(initbal, nm.Bal(), 1.0);
        nm.U(-1);
        ASSERT_NEAR(initbal, nm.Bal(), 1.0);
        nm.R(1);
        initbal = nm.Bal();
        nm.R(1);
        ASSERT_NEAR(initbal + 1.0, nm.Bal(), 1.0);
        nm.R(10);
        ASSERT_NEAR(initbal + 11.0, nm.Bal(), 1.0);
        nm.R(-11);
        ASSERT_NEAR(initbal, nm.Bal(), 1.0);
  }
  TEST_F(NMT, NeuronSharedWin) {
      float I[3],O[3];
      I[0] = 0.00000001;
      I[1] = 1.0 - 0.00000001;
      I[2] = 1.0 - 0.00000001;
      I[3] = 1.0 - 0.00000001;
      nm.I(I);
      nm.U(-1);
      nm.R(10);
      nm.print();

  }
  TEST_F(NMT, NeuronBlockBetTests){
      int from = 3, to = 0;
      float A[3*3];
      int i = 0;
      int change_i = 0;

      I[0] = 0.1;
      I[1] = I[0];
      I[2] = I[0];
      
      vector<float> Av;

      nm.N.setWin(3*from, 1);
      nm.N.set(3*from, 1, I);

      Av = nm.readSBlock(nm.S.at(position(from, to))->SP11);
      Av[change_i] = 0.99;
      nm.writeSBlock(nm.S.at(position(from, to))->SP10, Av);
      nm.writeSBlock(nm.S.at(position(from, to))->SP01, Av);
      Av[change_i] = 0.01;
      nm.writeSBlock(nm.S.at(position(from, to))->SP11, Av);
      nm.writeSBlock(nm.S.at(position(from, to))->SP00, Av);

      nm.S.at(position(from, to))->update();
      assert(nm.S.at(position(from, to))->To == nm.N.at(to));
      assert(nm.S.at(position(from, to))->From == nm.N.at(from));

      nm.N.get(3*to, 1, O);
      EXPECT_GT(O[0], 0.6);
  }
  TEST_F(NMT, NeuronBlockFindWinnerTests){
      int from = 3, to = 0;
      float A[3*3];
      int i = 0;
      int change_i = 0;

      I[0] = 0.1;
      I[1] = I[0];
      I[2] = I[0];
      
      vector<float> Av;

      nm.N.setWin(3*from, 1);
      nm.N.set(3*from, 1, I);

      Av = nm.readSBlock(nm.S.at(position(from, to))->SP11);
      Av[change_i] = 0.99;
      nm.writeSBlock(nm.S.at(position(from, to))->SP10, Av);
      nm.writeSBlock(nm.S.at(position(from, to))->SP01, Av);
      Av[change_i] = 0.01;
      nm.writeSBlock(nm.S.at(position(from, to))->SP11, Av);
      nm.writeSBlock(nm.S.at(position(from, to))->SP00, Av);

      nm.S.at(position(from, to))->update();
      assert(nm.S.at(position(from, to))->To == nm.N.at(to));
      assert(nm.S.at(position(from, to))->From == nm.N.at(from));

      nm.payNB(to, 100);
      nm.N.at(to)->R();
      nm.S.at(position(from, to))->R();
      nm.N.at(from)->R();
      
      Av = nm.readSBlock(nm.S.at(position(from, to))->SW);
      EXPECT_GT(Av[change_i], 0);
  }
  TEST_F(NMT, NeuronBlockLearnTests){
      int from = 3, to = 0;
      for(int i = 0; i < 100; ++i){
          I[0] = (i%2)*0.8 + 0.1;
          I[1] = I[0];
          I[2] = 1.0 - I[0];

          nm.N.setWin(3*from, 3);
          nm.N.set(3*from, 3, I);

          nm.S.at(position(from, to))->update();
          assert(nm.S.at(position(from, to))->To == nm.N.at(to));
          assert(nm.S.at(position(from, to))->From == nm.N.at(from));

          nm.N.get(3*to, 1, O);

          float pay = (((0.1 + 0.8*(i%2)) - O[0])*300);

          nm.print();
          nm.payNB(to, pay);
          nm.print();

          nm.N.at(to)->R();
          nm.S.at(position(from, to))->R();
          nm.print();
          nm.N.at(from)->R();
          nm.print();
          nm.N.at(to)->zero_nbets();
      }
      vector<float> Av;
      //cout << "p(out == 1 | in == 1)" << endl;
      Av = nm.readSBlock(nm.S.at(position(from, to))->SP11);
      //printBlock(Av);
      ASSERT_GT(Av[0], 0.9);
      //cout << "p(out == 1 | in == 0)" << endl;
      Av = nm.readSBlock(nm.S.at(position(from, to))->SP10);
      //printBlock(Av);
      ASSERT_LT(Av[0], 0.1);
      //cout << "p(out == 0 | in == 1)" << endl;
      Av = nm.readSBlock(nm.S.at(position(from, to))->SP01);
      //printBlock(Av);
      ASSERT_LT(Av[0], 0.1);
      //cout << "p(out == 0 | in == 0)" << endl;
      Av = nm.readSBlock(nm.S.at(position(from, to))->SP00);
      //printBlock(Av);
      ASSERT_GT(Av[0], 0.9);
  }
  TEST_F(NMT, BetSumOrientationTest){
      vector<float> Av;
      int from = 3, to = 0;
      int i = 0;

      Av = nm.readSBlock(nm.S.at(position(from, to))->SBET1);
      for(auto t : Av){
         t=0;
      }
      Av[0] = 5;
      Av[1] = 5;
      nm.writeSBlock(nm.S.at(position(from, to))->SBET1, Av);
      Av = nm.readSBlock(nm.S.at(position(from, to))->SBET1);
//      printBlock(Av);
      nm.BetSum(nm.S.at(position(from, to))->To->BET1,
                nm.S.at(position(from, to))->SBET1);
      Av = nm.readNBlock(nm.S.at(position(from, to))->To->BET1);
      i = 0;
//      printBlock(Av);
      nm.printBet1();
      ASSERT_EQ(Av[0], 10);
      ASSERT_EQ(Av[1], 0);
      ASSERT_EQ(Av[1], 0);
  }
  TEST_F(NMT, LearnOneToOneMapping){

      float I[3],O[3];
      for(int i = 0; i < 100; ++i){
          I[0] = (i%2)*0.8 + 0.1;
          I[1] = I[0];
          I[2] = I[0];
          nm.I(I);
          nm.U(-1);
          nm.O(O);
          float pay = ((i%2)*300 - 250);
          nm.R(pay);
//          cout << i%2 << " " << nm.getNP(0) << " " << pay << endl;
      }

      for(int i = 0; i < 2; ++i){
        I[0] = 0.1;
        I[1] = (i%2)*0.8 + 0.1;
        nm.I(I);
        nm.U(-1);
        nm.O(O);
        float pay = ((i%2)*300-250);
        if(pay > 0)
            ASSERT_GT(nm.getNP(0), 0.5);
        else
            ASSERT_LT(nm.getNP(0), 0.5);
//        EXPECT_LT(((pay>0)*2 - 1)*nm.getNP(0),0);
      }

  }
  /*
  TEST_F(NMT, NeuronLearnPleasureFromInput){

      float I[2],O[2];
      for(int i = 0; i < 100; ++i){
          I[0] = 0.1;
          I[1] = (i%2)*0.8 + 0.1;
          nm.I(I);
          nm.U(-1);
          nm.O(O);
          float pay = ((i%2)*20-15);
          nm.R(pay);
      }

      for(int i = 0; i < 2; ++i){
        I[0] = 0.1;
        I[1] = (i%2)*0.8 + 0.1;
        nm.I(I);
        nm.U(-1);
        nm.O(O);
        nm.printP();
        float pay = ((i%2)*20-15);
        EXPECT_LT(((pay>0)*2 - 1)*nm.getNP(0),0);
      }

  }
  */
  /*
  TEST_F(NMT, NeuronLooseFunctionTest) {
      float I[2],O[2];
      for(int i = 0; i < 100; ++i){
          I[0] = 0.1;
          I[1] = (i%2)*0.8 + 0.1;
          nm.I(I);
          nm.U(-1);
          nm.O(O);
          float pay = -15;
          if((O[0]-0.5)*((i%2)*1.0 - 0.5) < 0)
              pay += 10;
          if((O[1]-0.5)*(0.5 - (i%2)*1.0) < 0)
              pay += 10;
          nm.R(pay);
      }

      for(int i = 0; i < 2; ++i){
        I[0] = 0.1;
        I[1] = (i%2)*0.8 + 0.1;
        nm.I(I);
        nm.U(-1);
        nm.O(O);
        EXPECT_LT((O[0]-0.5)*((i%2)*1.0 - 0.5),0);
        EXPECT_LT((O[1]-0.5)*(0.5 - (i%2)*1.0),0);
      }
  }
  */
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
