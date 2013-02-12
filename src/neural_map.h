#ifndef NEURALMAP_HH
#define NEURALMAP_HH
#include <iostream>
#include <map>
#include <vector>
#include <CL/cl.h>
#include "brain_functions.h"
#include "timer.hh"

#define NBSIZE (32)
#define SBSIZE (NBSIZE*NBSIZE)
#define ITERS 5

using namespace std;
typedef float seconds;
typedef float neur;
typedef pair<size_t,size_t> position; 

ostream &operator<<(ostream& os, position p);

float costOfTime(seconds time);
seconds timeOfIncome(float income_snt);
seconds timeOfCost(float income_snt);

//G/s = M/ms = k/µs = B/ns|€/G = n€/B
float static stateCosts[4]     		= {0.05,1.6 ,8.75 ,133}; // n€/B
float static stateStayCosts[4]     	= {.001,.05 ,.277 ,4.2}; // n€/B/s*(10^-6)
float static stateCapacities[4]		= {2048,256 ,32   ,3  }; // GB
float static stateWriteCost[4] 		= {0.0005,0.016,0 ,0  }; // n€/B
float static stateBandwidth[4] 		= {0.1 ,0.5 ,64   ,320}; // B/ns
float static statePowerDown[4] 		= {1   ,1   ,0    ,0  }; // bool

float static stateChangeInit[4*4] 	= {                       // B/ns
0   ,4100000,4000000,4050000,
4100000,0   ,100000 ,150000 ,
4000000,100000 ,0   ,5000   ,
4050000,100000 ,5000,0
};
float static stateChangeSpeed[4*4] 	= {                       // B/ns
10e9,0.1 ,0.1,0.1,
0.1 ,10e9,0.5,0.5,
0.1 ,0.5 ,10e9,16,
0.1 ,0.5 ,16 ,10e9
};

string static stateName[4] 		= {"hdd", "sdd", "ram", "gpu"};

enum {HDD,SDD,RAM,GPU};

typedef size_t statetype;

typedef unsigned int uint;
typedef cl_mem Mat;
typedef cl_mem Col;

class NeuronBlock{
    public:
        NeuronBlock();
        ~NeuronBlock();
        size_t size();
        void pay(int, float);
        float setState(statetype s);
        Col  P,TMP,BAL,BET0,BET1,W,L; //float8
    private:
        statetype state;
};
class SynapsBlock{
    Mat SINFO,SP0,SP1,STMP,SP00,SP01,SP10,SP11,SP,SBAL,SBET0,SBET1; //float16
    NeuronBlock *N;
    size_t x,y;
    statetype state;
    size_t size();
    void refresh();
    float setState(uint8_t s);
    public:
        void R(float);
        Mat SW;
        SynapsBlock(NeuronBlock *N,size_t x,size_t y);
//        SynapsBlock();
        ~SynapsBlock();
        //void R();
        void kill(long s,long l=1, long v=NBSIZE);
        void print();
        void printS();
        void printW();
        seconds update();
        seconds cost();
        seconds expected();
};

typedef map<size_t,NeuronBlock *> nmap;
class    Neurons{
    nmap N;
    public:
        Neurons();
        NeuronBlock *at(size_t i);
        void addBlock();
        void printN();
        void get(size_t start,int l,float *A);
        void set(size_t start,int l,float *A);
};

typedef map<position, SynapsBlock *> smap;
class    Synapses{
    smap SB;
    Neurons  *N;
    public:
        Synapses(Neurons *N);
        SynapsBlock *at(position);
        void addBlock(SynapsBlock *S,position p);
        void update();
        position maxbid();
};


class neural_map{
    size_t ni, no;
    Neurons  N;
    Synapses S;
    void get_neural_states(size_t,size_t,float *);
    void set_neural_states(size_t,size_t,float *);
    position get_highest_bid();
    seconds update_synaps_map(seconds);
    public:
        neural_map(size_t ni, size_t no);
        ~neural_map();
        void R(float a);
        void I(float * I);
        void O(float * O);
        void U(seconds t);
};

#endif
