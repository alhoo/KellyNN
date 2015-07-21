#include "neural_map.h"

ofstream logfile("latest.log");
#define LOG logfile
//#define LOG logfile << "INFO: " << __FILE__ << ":" << __LINE__ << " " << __func__ << "() : "
//#define LOG cout

opencl_brain_functions *bf;
void log_nm(float *NB, int bc);

ostream &operator<<(ostream& os, position p){
    os << p.first << "," << p.second;
    return os;
}

inline bool Synapses::has(position p){
  return SB.find(p) != SB.end();
}

void NeuronBlock::getP(float *CPU_P){
  bf->opencl_getv(P,CPU_P,0,NBSIZE);
}
float neural_map::getNP(size_t p){
  float P[NBSIZE];
  size_t bp = p/NBSIZE;
  N.at(bp)->getP(P);
  size_t pb = p - bp*NBSIZE;
  return P[pb];
}
neural_map::neural_map(size_t np, size_t ni, size_t no, size_t nc)
    :np(np),ni(ni),no(no),nc(nc),N(),S(&N)
{

    LOG << "Initializing neural map" << endl;

    nbp=(np - 1)/NBSIZE + 1;
    nbi=(ni - 1)/NBSIZE + 1;
    nbo=(no - 1)/NBSIZE + 1;
    nbc=(nc - 1)/NBSIZE + 1;
    bf = new opencl_brain_functions(NBSIZE,ni,no);
    size_t i;
    for(i = 0; i < nbp + nbi + nbo + nbc; ++i){
        N.at(i);
    }
    for(; i < nbp; ++i){
        //Pleasure center doesn't form connections to anything.
        //Pleasure has an effect on the center?
        /*
        for(size_t j = nbp + nbi + nbo; j < nbp + nbi + nbo + nbc; ++j){
            S.at(position(j,i));
        }
        */
    }
    for(i = nbp + nbi; i < nbp + nbi + nbo; ++i){
        //Actions are connected to pleasure center
        //From action To pleasure
        //i == action
        //j == pleasure
        for(size_t j = 0; j < nbp; ++j){
            S.at(position(i,j));
        }
    }
    for(i = nbp + nbi + nbo; i < nbp + nbi + nbo + nbc; ++i){
        //Internal neurons are fully connected
        //Center has an effect on the pleasure?
        for(size_t j = 0; j < nbp; ++j){
            S.at(position(i,j));
        }
        //Center has an effect on output and its self
        for(size_t j = nbp + nbi; j < nbp + nbi + nbo + nbc; ++j){
            S.at(position(i,j));
        }
    }
    for(i = nbp; i < nbp + nbi; ++i){
        //Input is fully connected
        for(size_t j = 0; j < nbp; ++j){
            S.at(position(i,j));
        }
        for(size_t j = nbp + nbi; j < nbp + nbi + nbo + nbc; ++j){
            S.at(position(i,j));
        }
    }
    bf->info();
    //print_map();
}
size_t neural_map::get_block_count(char blocktype){
  switch (blocktype){
    case 'c':
      return nbc;
    case 'i':
      return nbi;
    case 'o':
      return nbo;
    case 'p':
      return nbp;
    default:
      return 0;
  }
  return 0;
}
string neural_map::write_map(){
  ostringstream oss;
    for(size_t to = 0; to < nbp + nbi + nbo + nbc; ++to){
      for(size_t from = 0; from < nbp + nbi + nbo + nbc; ++from){
      char connected = '0';
      //bf->opencl_getv(S.at(position(from, to))->SBET1,TMP,0,SBSIZE);
      if(S.has(position(from, to))){
        connected = '1';
      }
      oss << connected;
    }
    oss << endl;
  }
  return oss.str();
}
void neural_map::print_map(){
  cout << write_map();
}
neural_map::~neural_map()
{
    LOG << "" << endl;
    delete bf;
}

void neural_map::payNB(int nb, float a){

    bf->opencl_pay_neuron(N.at(nb)->BAL,N.at(nb)->W,0,a);
}
void neural_map::R(float a)
{
    LOG << "Got " << a << "$ for my effort" << endl;
   // set_neural_states(0,1,&a);
//    S.at(position(0,0))->print();
    payNB(0, a);
    //bf->opencl_pay_neuron(N.at(0)->BAL,N.at(0)->W,0,a);
    for(int i = 0; i < nbp + nbi + nbo + nbc; ++i){
      N.at(i)->R();
      for(int j = nbp; j < nbp + nbi + nbo + nbc; ++j){
        if(S.has(position(j,i))) {
          assert(S.at(position(j, i))->getTo() == N.at(i));
          S.at(position(j, i))->R();
        }
      }
      N.at(i)->zero_nbets();
    }
    print();
}

void neural_map::I(float * I)
{
    LOG << "Setting inputs from a vector" << endl;
    set_neural_win(0,ni);
    set_neural_states(0,ni,I);

    LOG << "Input values: ";
    float TMP[NBSIZE];
    bf->opencl_getv(N.at(nbp)->P,TMP,0,NBSIZE);
    for(int x = 0; x < NBSIZE; ++x) 
        LOG << setw(8) << TMP[x];
    LOG << endl;

    print();
}
void neural_map::setIO(size_t niN,size_t noN){
    LOG << "Setting number of in- and outputs" << endl;
    ni = niN; no = noN;
    bf->setIO(niN,noN);
}
void neural_map::O(float * O)
{
    LOG << "Setting outputs to a vector" << endl;
    get_neural_states(0,no,O);
    LOG << "Output values: ";
    float TMP[NBSIZE];
    bf->opencl_getv(N.at(nbp + nbi)->P,TMP,0,NBSIZE);
    for(int x = 0; x < NBSIZE; ++x) 
        LOG << setw(8) << TMP[x];
    LOG << endl;
    print();
}

void neural_map::U(seconds t)
{
    LOG << "Updating neuralmap (for t = " << t << "s)" << endl;
    if(t > 0){
      double a = now() + t;
      while(a - now() > 0){
          update_synaps_map(a - now());
      }
    }
    else{
      //Full I->C->P->O map update
      //nbp(nbp),nbi(nbi),nbo(nbo),nbc(nbc)
      for(int i =nbp;i < nbp+nbi; ++i){
        for(int j = 0; j < nbp + nbi + nbo + nbc; ++j){
          //j = Connection To
          //i = Connection From
          if(S.has(position(j,i))) S.at(position(j,i))->update();
        }
      }
      for(int i =nbp+nbi+nbo;i < nbp+nbi+nbo+nbc; ++i){
        for(int j = 0; j < nbp + nbi + nbo + nbc; ++j){
          if(S.has(position(j,i))) S.at(position(j,i))->update();
        }
      }
      for(int i = 0;i < nbp; ++i){
        for(int j = 0; j < nbp + nbi + nbo + nbc; ++j){
          if(S.has(position(j,i))) S.at(position(j,i))->update();
        }
      }
      for(int i =nbp+nbi;i < nbp+nbi+nbo; ++i){
        for(int j = 0; j < nbp + nbi + nbo + nbc; ++j){
          if(S.has(position(j,i))) S.at(position(j,i))->update();
        }
      }
    }
    print();
}
void neural_map::get_neural_states(size_t start, size_t l, float *A)
{
    LOG << "Getting neural states to a variable" << endl;
    start = start + (nbp + nbi)*NBSIZE;
    N.get(start,l,A);
}

void neural_map::set_neural_win(size_t start, size_t l)
{
    LOG << "Setting neurons as winners" << endl;
    start = start + (nbp)*NBSIZE;
    N.setWin(start,l);
}
void neural_map::set_neural_states(size_t start, size_t l, float *A)
{
    LOG << "Setting neural states from a variable" << endl;
    start = start + (nbp)*NBSIZE;
    LOG << "setting neurons from " << start << " to " << start + l << endl;
    N.set(start,l,A);
}

position neural_map::get_highest_bid()
{
    LOG << "Getting the position of the highest bids" << endl;
//    S.update();
    return S.maxbid();
}
bool operator==(position &a, position &b){
    if(a.first != b.first) return false;
    if(a.second != b.second) return false;
    return true;
}
seconds neural_map::update_synaps_map(seconds limit)
{
    LOG << "Updating synapsmap" << endl;
    position p = get_highest_bid();
    if(p == position(0,3)) return limit;
    else return S.at(p)->update();

    if(timeOfCost(S.at(p)->cost()) < limit) return S.at(p)->update();
    else return limit;
}


/*

SYNAPSBLOCK

*/


SynapsBlock::SynapsBlock(NeuronBlock *From,NeuronBlock *To,size_t i,size_t j)
    :To(To),From(From),x(i),y(j),state(GPU)
{
    LOG << "Initializing a synapsblock" << endl;
    SINFO = bf->gpu_malloc(SINFO,SBSIZE,1);
    SP0 = bf->gpu_malloc(SP0,SBSIZE,0.49);
    SP1 = bf->gpu_malloc(SP1,SBSIZE,0.51);
    STMP = bf->gpu_malloc(STMP,SBSIZE,0);
    SP00 = bf->gpu_malloc(SP00,SBSIZE,0.49);
    SP01 = bf->gpu_malloc(SP01,SBSIZE,0.51);
    SP10 = bf->gpu_malloc(SP10,SBSIZE,0.51);
    SP11 = bf->gpu_malloc(SP11,SBSIZE,0.49);
    SP = bf->gpu_malloc(SP,SBSIZE,0.5);
    SBAL = bf->gpu_malloc(SBAL,SBSIZE,SYNAPSINITBAL);
    SBET0 = bf->gpu_malloc(SBET0,SBSIZE,0);
    SBET1 = bf->gpu_malloc(SBET1,SBSIZE,0);
    SW = bf->gpu_malloc(SW,SBSIZE,0);
}
SynapsBlock::~SynapsBlock(){
    LOG << "Freeing a synapsblock" << endl;
    bf->gpu_free(SINFO);
    bf->gpu_free(SP0);
    bf->gpu_free(SP1);
    bf->gpu_free(STMP);
    bf->gpu_free(SP00);
    bf->gpu_free(SP01);
    bf->gpu_free(SP10);
    bf->gpu_free(SP11);
    bf->gpu_free(SP);
    bf->gpu_free(SBAL);
    bf->gpu_free(SBET0);
    bf->gpu_free(SBET1);
    bf->gpu_free(SW);
    state = 0;
}

/**
This function is the same as in tag-2.00.00 function Brain::refresh()
**/
seconds SynapsBlock::update()
{
//    seconds d = timeOfCost(cost());
    LOG << "Updating synapsblock" << endl;

    if(state != GPU)
      if(DEBUG>1) cerr << "\t\timplement moving to GPU" << endl;
    assert(state == GPU);

    bf->opencl_synaps_refresh(SP0,SP01,SP00,From->P);
    bf->opencl_synaps_refresh(SP1,SP11,SP10,From->P);

    bf->opencl_synaps_bet(SBET1,SBAL,SP1,SP);
    bf->opencl_synaps_bet(SBET0,SBAL,SP0,SP);

    bf->opencl_bet_sum(To->BET1,SBET1);
    bf->opencl_bet_sum(To->BET0,SBET0);

    bf->opencl_neuron_refresh(To->BET1,To->BET0,To->P);
    return 1;
}
void neural_map::BetSum(Col N, Mat S){
    bf->opencl_bet_sum(N,S);
}

void NeuronBlock::R(){
    bf->opencl_npay(BET0,BET1,BAL,W,TMP);
}
void NeuronBlock::zero_nbets(){
    bf->opencl_fill(BET0,0.0,NBSIZE);
    bf->opencl_fill(BET1,0.0,NBSIZE);
}
void SynapsBlock::R(){
    LOG << "Adding value to synapsblock" << endl;
    for(int i = 0; i< ITERS + 1; ++i){
        bf->opencl_find_winning_synapses(SBET0,SBET1,To->BET0,
                To->BET1,To->W,SW);
        bf->wait();
        bf->opencl_find_winning_neurons(To->W,SW,SBET0,SBET1);
    }
    bf->opencl_synaps_learn(SP00,SP01,SP10,SP11,SBET0,SBET1,
                SW,From->P,SINFO);

    bf->opencl_synaps_learn2(SP,SW,SINFO);

    bf->opencl_update_synaps_info(SINFO);

    bf->opencl_pay(SBET0,SBET1,To->BET0,To->BET1,To->TMP,SBAL,To->BAL,SW,To->W);
}

size_t SynapsBlock::size(){
	return SBSIZE*12*sizeof(float);
}
neur SynapsBlock::cost()
{
    LOG << "Checking synaps cost" << endl;
    neur ret = stateStayCosts[state]*size()*10e-15 + 
          costOfTime((size()/stateBandwidth[state] + 
          size()/stateChangeSpeed[4*state + GPU])*10e-9);
    return ret;
}

Synapses::Synapses(Neurons *N):N(N)
{
    LOG << "Initializing synapses" << endl;
    state=0;
}
void Synapses::addBlock(SynapsBlock *S,position p){
    LOG << "Adding a synapsblock" << endl;
    SB[p] = S;
}
NeuronBlock *Neurons::at(size_t p)
{
	if(N.find(p) == N.end())
    {
        N.insert(pair<size_t,NeuronBlock*>(p,new NeuronBlock()));
    }
	return N.at(p);
}
SynapsBlock *Synapses::at(position p)
{
	if(SB.find(p) == SB.end())
    {
        LOG << "Synapsblock not found" << endl;
        SB.insert(pair<position,SynapsBlock*>(p,new SynapsBlock(N->at(p.first),
            N->at(p.second),p.first,p.second)));
    }
	return SB.at(p);
}
seconds SynapsBlock::expected()
{
    return 10 - cost();
}
void SynapsBlock::kill(long s, long vertical, long l){
    bf->opencl_synaps_die(SBAL,SP1,s,vertical,l);
}

/*

NEURONBLOCK

*/


NeuronBlock::NeuronBlock():state(GPU)
{
    LOG << "Initializing a neuronblock" << endl;
    P = bf->gpu_malloc(P,NBSIZE,0.5);
    TMP = bf->gpu_malloc(TMP,NBSIZE);
    BAL = bf->gpu_malloc(BAL,NBSIZE,NEURONINITBAL);
    float PleasureCenterInitBal = NEURONINITBAL;
//    float PleasureCenterInitBal = 100.0f*NEURONINITBAL;
//FIXME separate this from the neuronblock
    bf->opencl_setv(BAL,&PleasureCenterInitBal,0,1);
    BET0 = bf->gpu_malloc(BET0,NBSIZE,0);
    BET1 = bf->gpu_malloc(BET1,NBSIZE,0);
    W = bf->gpu_malloc(W,NBSIZE,0);
    L = bf->gpu_malloc(L,NBSIZE,0);
    PI = bf->gpu_malloc(PI,NBSIZE,0);
}
NeuronBlock::~NeuronBlock()
{
    LOG << "Removing a neuronblock" << endl;
    bf->gpu_free(P);
    bf->gpu_free(TMP);
    bf->gpu_free(BAL);
    bf->gpu_free(BET0);
    bf->gpu_free(BET1);
    bf->gpu_free(W);
    bf->gpu_free(L);
}
size_t NeuronBlock::size(){
    return NBSIZE*7*sizeof(float);
}

float NeuronBlock::setState(statetype s){
    float cost = size()/stateChangeSpeed[4*state + s];
    return cost;
}
/*

SYNAPSES

*/


void Synapses::update()
{

    LOG << "Updating synapses (depreciated)" << endl;
    cerr << "FUNCTION NOT USED: " << __func__ << endl;

}

position Synapses::maxbid()
{
    LOG << "Getting maxpid" << endl;
//    if(VERBOSE){int val = 1; cin >> val; assert(val > 0);}
    //FIXME: do this right
    if(state == 0){ ++state; return position(3,1); }
    if(state == 1){ ++state; return position(2,3); }
    if(state == 2){ ++state; return position(2,1); }
    if(state == 3){ ++state; return position(3,2); }
    if(state == 4){ ++state; return position(0,1); }
    if(state == 5){ ++state; return position(0,2); }
    if(state == 6){ state = 0; return position(0,3); }
    if(state>6) state=0;
    if(state<0) state=0;


    smap::iterator min = SB.begin(); 
    float minexp = -10e10;
    for(smap::iterator it = SB.begin(); it != SB.end(); ++it){
        float nowexp = it->second->expected();
        if(nowexp > minexp) { minexp = nowexp; min = it; }
    }
    return min->first;
}

/*

NEURONS

*/


void Neurons::addBlock(){
    LOG << "Adding a new neuronblock" << endl;
        N[N.size()] = new NeuronBlock();
}
    Neurons::Neurons()
{
}

int min(int a,int b){ if(a < b) return a; return b; }

void Neurons::get(size_t start,int l,float *A)
{
    LOG << "Getting neuron probability values" << endl;
    int i = start/NBSIZE;
    int j = start%NBSIZE;
    assert(j == start - NBSIZE*i);
    bf->opencl_getv(N.at(i)->P,A,j,min(j+l,NBSIZE));

    l -= (NBSIZE - j);
    A += (NBSIZE - j);
    for(++i; l > 0; ++i){
        bf->opencl_getv(N.at(i)->P,A,0,min(l,NBSIZE));
        l -= NBSIZE;
        A += NBSIZE;
    }
}

void Neurons::setWin(size_t start,int l)
{
    LOG << "Setting winning neurons" << endl;
    float *A = new float[l];
    for(int i = 0; i < l; ++i) A[i] = 1;
    int i = start/NBSIZE;
    int j = start%NBSIZE;
    assert(j == start - NBSIZE*i);
    bf->opencl_setv(N.at(i)->W,A,j,min(j+l,NBSIZE));
    l -= (NBSIZE - j);
    A += (NBSIZE - j);
    for(++i; l > 0; ++i){
        bf->opencl_setv(N.at(i)->W,A,0,min(l,NBSIZE));
        l -= NBSIZE;
        A += NBSIZE;
    }
}
void Neurons::set(size_t start,int l,float *A)
{
    LOG << "Setting neuron probability values" << endl;
    int i = start/NBSIZE;
    int j = start%NBSIZE;
    LOG << "Setting probability on block " << i << " from " << j << " to " << min(j+l,NBSIZE) << endl;
    assert(j == start - NBSIZE*i);
    bf->opencl_setv(N.at(i)->P,A,j,min(j+l,NBSIZE));

    l -= (NBSIZE - j);
    A += (NBSIZE - j);
    for(++i; l > 0; ++i){
        LOG << "Setting probability on block " << i << " from " << 0 << " to " << min(l,NBSIZE) << endl;
        bf->opencl_setv(N.at(i)->P,A,0,min(l,NBSIZE));
        l -= NBSIZE;
        A += NBSIZE;
    }
}




/*

HELPERS

*/



/**
Operating costs: 4377.920€ = year <=> 72.034 s/snt
**/
float costOfTime(seconds time){ 
    return ((8.76*(0.07*0.6) + 4.01)*1000)*(time)/(365*24*3600);
}
/*
The time it should take (at max) to gather the income for a positive outcome
*/
seconds timeOfIncome(float income_neur){ 
    return (365*24*3600)*income_neur*10e-7/(((8.76*(0.07*0.6) + 4.01)*100000));
}
seconds timeOfCost(float income_snt){ 
    return timeOfIncome(income_snt);
}

void log_nm(float *NB, int bc){
    LOG << setprecision(4);
    for(int x = 0; x < bc*NBSIZE; ++x)
        LOG << setw(8) << NB[x];
    LOG << endl;
    for(int x = 0; x < bc*NBSIZE*bc*NBSIZE; ++x){
            LOG << setw(8) << NB[bc*NBSIZE + x];
            if(x % (bc*NBSIZE) == bc*NBSIZE - 1) LOG << endl;
    }
}
void neural_map::print(){
    printW();
    printP00();
    printP01();
    printP();
    printBal();
    printBet1();
    printBet0();
    }
vector<float> neural_map::readNBlock(Mat S){
    float TMP[NBSIZE];
    vector<float> ret;
    bf->opencl_getv(S, TMP, 0, NBSIZE);
    for(int i = 0; i < NBSIZE; ++i)
        ret.push_back(TMP[i]);
    return ret;
}
void neural_map::writeNBlock(Mat S, vector<float> A){
    float TMP[NBSIZE];
    int i = 0;
    for(auto t : A){
        TMP[i] = t;
        ++i;
    }
    bf->opencl_setv(S, TMP, 0, NBSIZE);
}
vector<float> neural_map::readSBlock(Mat S){
    float TMP[SBSIZE];
    vector<float> ret;
    bf->opencl_getv(S, TMP, 0, SBSIZE);
    for(int i = 0; i < SBSIZE; ++i)
        ret.push_back(TMP[i]);
    return ret;
}
void neural_map::writeSBlock(Mat S, vector<float> A){
    float TMP[SBSIZE];
    int i = 0;
    for(auto t : A){
        TMP[i] = t;
        ++i;
    }
    bf->opencl_setv(S, TMP, 0, SBSIZE);
}
void neural_map::printBet1(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int from = 0; from < bc; ++from){
        bf->opencl_getv(N.at(from)->BET1,&NB[from*NBSIZE],0,NBSIZE);
        for(int to = 0; to < bc; ++to){
            int pos = bc*NBSIZE + to*bc*NBSIZE*NBSIZE + from*NBSIZE;
            if(S.has(position(from, to))){
              bf->opencl_getv(S.at(position(from, to))->SBET1,TMP,0,SBSIZE);
              for(int x = 0; x < SBSIZE; ++x){
                NB[pos + x%NBSIZE + (x/NBSIZE)*bc*NBSIZE] = TMP[x];
              }
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}
void neural_map::printW(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int i = 0; i < bc; ++i){
        bf->opencl_getv(N.at(i)->W,&NB[i*NBSIZE],0,NBSIZE);
        for(int j = 0; j < bc; ++j){
            int pos = bc*NBSIZE + j*bc*NBSIZE*NBSIZE + i*NBSIZE;
            if(S.has(position(i,j))){
            bf->opencl_getv(S.at(position(i,j))->SW,TMP,0,SBSIZE);
            for(int x = 0; x < NBSIZE; ++x)
                for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = TMP[x + y*NBSIZE];
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}
void neural_map::printP00(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int i = 0; i < bc; ++i){
        bf->opencl_getv(N.at(i)->P,&NB[i*NBSIZE],0,NBSIZE);
        //i == toNeuron
        //j == fromNeuron
        for(int j = 0; j < bc; ++j){
            int pos = bc*NBSIZE + j*bc*NBSIZE*NBSIZE + i*NBSIZE;
            if(S.has(position(i,j))){
                bf->opencl_getv(S.at(position(i,j))->SP00,TMP,0,SBSIZE);
                for(int x = 0; x < NBSIZE; ++x)
                    for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = TMP[x + y*NBSIZE];
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                    NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}
void neural_map::printP01(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int i = 0; i < bc; ++i){
        bf->opencl_getv(N.at(i)->P,&NB[i*NBSIZE],0,NBSIZE);
        for(int j = 0; j < bc; ++j){
            int pos = bc*NBSIZE + j*bc*NBSIZE*NBSIZE + i*NBSIZE;
            if(S.has(position(i,j))){
                bf->opencl_getv(S.at(position(i,j))->SP01,TMP,0,SBSIZE);
                for(int x = 0; x < NBSIZE; ++x)
                    for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = TMP[x + y*NBSIZE];
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                    NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}
void neural_map::printBet0(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int i = 0; i < bc; ++i){
        bf->opencl_getv(N.at(i)->BET0,&NB[i*NBSIZE],0,NBSIZE);
        for(int j = 0; j < bc; ++j){
            int pos = bc*NBSIZE + j*bc*NBSIZE*NBSIZE + i*NBSIZE;
            if(S.has(position(i,j))){
              bf->opencl_getv(S.at(position(i,j))->SBET0,TMP,0,SBSIZE);
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = TMP[x + y*NBSIZE];
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}
void neural_map::printBal(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int i = 0; i < bc; ++i){
        bf->opencl_getv(N.at(i)->BAL,&NB[i*NBSIZE],0,NBSIZE);
        for(int j = 0; j < bc; ++j){
            int pos = bc*NBSIZE + j*bc*NBSIZE*NBSIZE + i*NBSIZE;
            if(S.has(position(i,j))){
                  bf->opencl_getv(S.at(position(i,j))->SBAL,TMP,0,SBSIZE);
                  for(int x = 0; x < NBSIZE; ++x)
                      for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = TMP[x + y*NBSIZE];
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                      NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}
void neural_map::printP(){
    LOG << "              ----  " << __func__ << " ----"  << endl;
    unsigned int bc = (nbp + nbi + nbo + nbc);
    float NB[(bc*NBSIZE)*(bc*NBSIZE + 1)];
    float TMP[(SBSIZE)];

    for(int i = 0; i < bc; ++i){
        bf->opencl_getv(N.at(i)->P,&NB[i*NBSIZE],0,NBSIZE);
        for(int j = 0; j < bc; ++j){
            int pos = bc*NBSIZE + j*bc*NBSIZE*NBSIZE + i*NBSIZE;
            if(S.has(position(i,j))){
                bf->opencl_getv(S.at(position(i,j))->SP,TMP,0,SBSIZE);
                for(int x = 0; x < NBSIZE; ++x)
                    for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x*bc*NBSIZE + y] = TMP[x + y*NBSIZE];
            }
            else{
              for(int x = 0; x < NBSIZE; ++x)
                  for(int y = 0; y < NBSIZE; ++y)
                        NB[pos + x + y*bc*NBSIZE] = 0;
            }
        }
    }
    log_nm(NB,bc);
}

/*
Initialize neurons and synapses at position
*/
void neural_map::initN(size_t p)
{
    /*
    P = bf->gpu_malloc(P,NBSIZE,0.5);
    TMP = bf->gpu_malloc(TMP,NBSIZE);
    BAL = bf->gpu_malloc(BAL,NBSIZE,NEURONINITBAL);
    float PleasureCenterInitBal = 100.0f*NEURONINITBAL;
//FIXME separate this from the neuronblock
    bf->opencl_setv(BAL,&PleasureCenterInitBal,0,1);
    BET0 = bf->gpu_malloc(BET0,NBSIZE,0);
    BET1 = bf->gpu_malloc(BET1,NBSIZE,0);
    W = bf->gpu_malloc(W,NBSIZE,0);
    L = bf->gpu_malloc(L,NBSIZE,0);
    if(VERBOSE) cout << "\t\tNB(x) created, state = " << state << endl;
    */
    NeuronBlock * MyNB = N.at(p/NBSIZE);
    MyNB->init(p);
        //Col  P,TMP,BAL,BET0,BET1,W,L; //float8
}
void NeuronBlock::init(int p){

    float InitBal = (1.0f*NEURONINITBAL);
    bf->opencl_setv(BAL,&InitBal,p,p+1);

}


void neural_map::initS(position p)
{
    SynapsBlock * MySB = S.at(position(p.first/NBSIZE,p.second/NBSIZE));
    MySB->init(p);
}
void SynapsBlock::init(position p){
    float InitBal = (1.0f*SYNAPSINITBAL);
    bf->opencl_setv(SBAL,&InitBal,p.first*NBSIZE + 
                p.second,p.first*NBSIZE + p.second + 1);

}

/*
Get the state of neurons and synapses at position
*/
void neural_map::stateN(size_t p)
{
    
    NeuronBlock * MyNB = N.at(p/NBSIZE);
//    MyNB->printN(p);

}

void neural_map::stateS(position p)
{
    SynapsBlock * MySB = S.at(position(p.first/NBSIZE,p.second/NBSIZE));
//    MySB->printS(p);

}

/*
Kill neurons and synapses at position
*/
void neural_map::killN(size_t p)
{
    
    NeuronBlock * MyNB = N.at(p/NBSIZE);
    MyNB->kill(p);

}

void neural_map::killS(position p)
{
    if(S.has(position(p.first/NBSIZE,p.second/NBSIZE))){
      SynapsBlock * MySB = S.at(position(p.first/NBSIZE,p.second/NBSIZE));
      MySB->kill(p);
    }

}

void NeuronBlock::kill(int p){

    
    float DeadBal = (0.0f);
    bf->opencl_setv(BAL,&DeadBal,p,p + 1);

}

void SynapsBlock::kill(position p){

    float DeadBal = (0.0f);
    bf->opencl_setv(SBAL,&DeadBal,p.first*NBSIZE + 
            p.second,p.first*NBSIZE + p.second + 1);

}

float neural_map::Bal(){
    return S.Bal() + N.Bal();
}
float Synapses::Bal(){
    float ret = 0;
    for(smap::iterator it = SB.begin(); it!=SB.end(); ++it)
        ret += it->second->Bal();
    return ret;
}
float Neurons::Bal(){
    float ret = 0;
    for(nmap::iterator it = N.begin(); it!=N.end(); ++it)
        ret += it->second->Bal();
    return ret;
}
float SynapsBlock::Bal(){
    float ret = bf->opencl_sum(SBAL,SBSIZE);
//    ret += bf->opencl_sum(SBET0,SBSIZE);
//    ret += bf->opencl_sum(SBET1,SBSIZE);
    return ret;
}
float NeuronBlock::Bal(){
    float ret = bf->opencl_sum(BAL,NBSIZE);
//    ret += bf->opencl_sum(BET0,NBSIZE);
//    ret += bf->opencl_sum(BET1,NBSIZE);
    return ret;
}
