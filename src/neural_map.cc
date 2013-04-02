#include "neural_map.h"

ofstream logfile("latest.log");
#define LOG logfile << "INFO: " << __FILE__ << ":" << __LINE__ << " " << __func__ << "() : "

opencl_brain_functions *bf;

ostream &operator<<(ostream& os, position p){
    os << p.first << "," << p.second;
    return os;
}

neural_map::neural_map(size_t np, size_t ni, size_t no, size_t nc)
    :np(np),ni(ni),no(no),nc(nc),N(),S(&N)
{

    LOG << "Initializing neural map" << endl;

    nbp=np/NBSIZE + 1;
    nbi=ni/NBSIZE + 1;
    nbo=no/NBSIZE + 1;
    nbc=nc/NBSIZE + 1;
    bf = new opencl_brain_functions(NBSIZE,ni,no);
    size_t i = 0;
    for(; i < nbp; ++i){
        N.at(i);
        //Pleasure center doesn't form connections.
    }
    for(i = nbp + nbi; i < nbp + nbi + nbo; ++i){
        N.at(i);
        //Actions are connected to pleasure center
        for(size_t j = 0; j < nbp; ++j){
            S.at(position(j,i));
        }
    }
    for(i = nbp + nbi + nbo; i < nbp + nbi + nbo + nbc; ++i){
        N.at(i);
        //Internal neurons are fully connected
        for(size_t j = 0; j < nbp; ++j){
            S.at(position(i,j));
        }
        for(size_t j = nbp + nbi; j < nbp + nbi + nbo + nbc; ++j){
            S.at(position(i,j));
        }
    }
    for(i = nbp; i < nbp + nbi; ++i){
        N.at(i);
        //Input is fully connected
        for(size_t j = 0; j < nbp; ++j){
            S.at(position(i,j));
        }
        for(size_t j = nbp + nbi; j < nbp + nbi + nbo + nbc; ++j){
            S.at(position(j,i));
        }
    }
    bf->info();
}

neural_map::~neural_map()
{
    LOG << "" << endl;
    delete bf;
}

void neural_map::R(float a)
{
    LOG << "Got " << a << "$ for my effort" << endl;
   // set_neural_states(0,1,&a);
//    S.at(position(0,0))->print();
    bf->opencl_pay_neuron(N.at(0)->BAL,N.at(0)->W,0,a);
    S.at(position(0,1))->R();
    S.at(position(0,2))->R();
    S.at(position(0,3))->R();
    S.at(position(3,2))->R();
    S.at(position(2,1))->R();
    S.at(position(2,3))->R();
    S.at(position(3,1))->R();
    printW();
}

void neural_map::I(float * I)
{
    LOG << "Setting inputs from a vector" << endl;
    set_neural_win(1,ni);
    set_neural_states(1,ni,I);
}
void neural_map::setIO(size_t niN,size_t noN){
    LOG << "Setting number of in- and outputs" << endl;
    ni = niN; no = noN;
    bf->setIO(niN,noN);
}
void neural_map::O(float * O)
{
    LOG << "Setting outputs to a vector" << endl;
    get_neural_states(1 + ni,no,O);
}

void neural_map::U(seconds t)
{
    LOG << "Updating neuralmap" << endl;
    double a = now() + t;
    while(a - now() > 0){
        update_synaps_map(a - now());
    }
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
    N.setWin(start,l);
}
void neural_map::set_neural_states(size_t start, size_t l, float *A)
{
    LOG << "Setting neural states from a variable" << endl;
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


SynapsBlock::SynapsBlock(NeuronBlock *To,NeuronBlock *From,size_t i,size_t j)
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
    printS();
}
SynapsBlock::~SynapsBlock(){
    LOG << "Freeing a synapsblock" << endl;
    printS();
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
    printS();

    if(state != GPU)
	if(DEBUG>1) cerr << "\t\timplement moving to GPU" << endl;

    bf->opencl_synaps_refresh(SP0,SP01,SP00,From->P);
    bf->opencl_synaps_refresh(SP1,SP11,SP10,From->P);

    bf->opencl_synaps_bet(SBET1,SBAL,SP1,SP);
    bf->opencl_synaps_bet(SBET0,SBAL,SP0,SP);

//    print();

    bf->opencl_bet_sum(To->BET1,SBET1);
    bf->opencl_bet_sum(To->BET0,SBET0);

    bf->opencl_neuron_refresh(To->BET1,To->BET0,To->P);
    printS();
    return 1;
}

void SynapsBlock::R(){
    LOG << "Adding value to synapsblock" << endl;
    printS();
    for(int i = 0; i< ITERS + 1; ++i){
        bf->opencl_find_winning_synapses(SBET0,SBET1,To->BET0,
                To->BET1,To->W,SW);
        bf->wait();
        bf->opencl_find_winning_neurons(To->W,SW,SBET0,SBET1);
    }
    bf->opencl_synaps_learn(SP00,SP01,SP10,SP11,SBET0,SBET1,
                SW,From->BET0,From->BET1,SINFO);

    bf->opencl_synaps_learn2(SP,SW,SINFO);

    bf->opencl_update_synaps_info(SINFO);

    bf->opencl_pay(SBET0,SBET1,To->BET0,To->BET1,SBAL,To->BAL,SW,To->W);
    printS();
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
    printN();
}
NeuronBlock::~NeuronBlock()
{
    LOG << "Removing a neuronblock" << endl;
    printN();
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
    assert(j == start - NBSIZE*i);
    bf->opencl_setv(N.at(i)->P,A,j,min(j+l,NBSIZE));

    l -= (NBSIZE - j);
    A += (NBSIZE - j);
    for(++i; l > 0; ++i){
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

void NeuronBlock::printN(){
    LOG << "NeuronBlock::N()" << endl;
    char A[NBSIZE];
    bf->print(P,NBSIZE,1,0,A);
    LOG << "N.P: " << A << endl;
    bf->print(BAL,NBSIZE,1,0,A);
    LOG << "N->BAL: " << A << endl;
    bf->print(BET1,NBSIZE,1,0,A);
    LOG << "N->BET1: " << A << endl;
    bf->print(BET0,NBSIZE,1,0,A);
    LOG << "N->BET0: " << A << endl;
}

void NeuronBlock::printN(int p){
    LOG << "NeuronBlock::N(" << p << ")" << endl;
    LOG << "N.P ";
    bf->print(P,p);
    LOG << "N->BAL ";
    bf->print(BAL,p);
    LOG << "N->BET0 ";
    bf->print(BET0,p);
    LOG << "N->BET1 ";
    bf->print(BET1,p);
}

void Neurons::printN(){
    LOG << "NeuronBlock::N()" << endl;
    LOG << "N.P" << endl;
    bf->print(N[0]->P,NBSIZE,1);
    LOG << "N->BAL" << endl;
    bf->print(N[0]->BAL,NBSIZE,1);
    LOG << "N->BET1" << endl;
    bf->print(N[0]->BET1,NBSIZE,1);
    LOG << "N->BET0" << endl;
    bf->print(N[0]->BET0,NBSIZE,1);
}
void neural_map::printW(){
    int Unisize = (NBSIZE + 1);
    int Width   = 4*Unisize + 1;
    int Height  = 4*Unisize;
    char tmp[Unisize*(Unisize + 1)];
    char map[Width*Height];
    //map[0] = ' ';
    LOG << "Neural_map" << endl;
    for(int y = 0; y < 4; ++y){
        for(int x = 0; x < 4; ++x){
            S.at(position(y,x))->printW(tmp,x*y == 0);
            for(int i = 0; i < Unisize; ++i)
                for(int j = 0; j < Unisize; ++j)
                {
            map[Width*Unisize*y + i*Width + j + x*Unisize]
                        = tmp[i*(Unisize + 1) + j];
                }
        }
    //    map[Width*Unisize*y + Width - 1] = '\n';
    }
//    for(int i = 0 ; i < Width*Height ; ++i) map[i] = '.';
    for(int y = 0; y < 4*Unisize; ++y)
        map[Width*y + Width - 1] = '\n';
    map[Width*Height - 1] = '\0';
    LOG << map << endl;
}
void SynapsBlock::printW(){
    LOG << "SB(" << x << "," << y << ")" << endl;
    bf->printW(To->W,From->W,SW,NBSIZE,SINFO);
}
void SynapsBlock::printW(char *A,bool n){
    bf->printW(To->W,From->W,SW,NBSIZE,SINFO,A,n);
}
void SynapsBlock::printS(position p){
    LOG << "\tSynapsBlock::S("<<p<<")" << endl;
    LOG << "\t\tSP ";
    bf->print(SP,p.first*NBSIZE + p.second);
    LOG << "\t\tSBAL ";
    bf->print(SBAL,p.first*NBSIZE + p.second);
}
void SynapsBlock::printS(){
    LOG << "\tSynapsBlock::S()" << endl;
    char A[SBSIZE];
    bf->print(SP,NBSIZE,NBSIZE,0,A);
    LOG << "\t\tSP" << endl;
    LOG << A << endl;
}
void SynapsBlock::print(){
    cout << "\tSynapsBlock("<< bf->opencl_sum(From->BAL,NBSIZE) << " + " 
         << bf->opencl_sum(SBAL,NBSIZE*NBSIZE) << " = " 
         << bf->opencl_sum(From->BAL,NBSIZE)+bf->opencl_sum(SBAL,NBSIZE*NBSIZE) 
         << ")" << endl;
    cout << "\t\tS.P10" << endl;
    bf->print(SP10,NBSIZE,NBSIZE);
    cout << "\t\tS.P11" << endl;
    bf->print(SP11,NBSIZE,NBSIZE);
    cout << "\t\tS.W" << endl;
    bf->print(SW,NBSIZE,NBSIZE);
    cout << "\t\tS.P"   << endl;
    bf->print(SP,NBSIZE,NBSIZE);
    cout << "\t\tS.BAL" << endl;
    bf->print(SBAL,NBSIZE,NBSIZE);
    cout << "\t\tSBET0" << endl;
    bf->print(SBET0,NBSIZE,NBSIZE);
    cout << "\t\tSBET1" << endl;
    bf->print(SBET1,NBSIZE,NBSIZE);
    cout << "\t\tN.BET0" << endl;
    bf->print(To->BET0,NBSIZE,1);
    cout << "\t\tN.BET1" << endl;
    bf->print(To->BET1,NBSIZE,1);
    cout << "\t\tN.BAL" << endl;
    bf->print(From->BAL,NBSIZE,1);
    cout << "\t\tN.W"   << endl;
    bf->print(From->W,NBSIZE,1);
    cout << "\t\tN.P"   << endl;
    bf->print(From->P,NBSIZE,1);
    cout << "\t\tn"     << endl;
    bf->print(SINFO,1,1);
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
    printN();

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
    MyNB->printN(p);

}

void neural_map::stateS(position p)
{
    SynapsBlock * MySB = S.at(position(p.first/NBSIZE,p.second/NBSIZE));
    MySB->printS(p);

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

    SynapsBlock * MySB = S.at(position(p.first/NBSIZE,p.second/NBSIZE));
    MySB->kill(p);

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
