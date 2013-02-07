#include "neural_map.h"

opencl_brain_functions *bf;

ostream &operator<<(ostream& os, position p){
    os << p.first << "," << p.second;
    return os;
}

neural_map::neural_map(size_t ni, size_t no):ni(ni),no(no),N(),S(&N)
{
    bf = new opencl_brain_functions(NBSIZE,ni,no);
    for(int i = 0; i < (ni+no+1)/NBSIZE + 1; ++i){
        //N.addBlock();
        N.at(i);
        for(int j = 0; j < (ni+no+1)/NBSIZE + 1; ++j){
            S.at(position(i,j));
            //S.addBlock(new SynapsBlock(N.at(i),i,j),position(i,j));
/*
            if(i*NBSIZE < ni){
                S.at(position(i,j))->kill(0,1,(ni%NBSIZE)*NBSIZE);
            }
            if((i - 1)*NBSIZE < ni){
                S.at(position(i,j))->kill(0,1,SBSIZE);
            }
            if((j + 1)*NBSIZE > ni && (j)*NBSIZE < ni)
                S.at(position(i,j))->kill(ni - j*NBSIZE,0,NBSIZE*(NBSIZE - 1));
*/
        }
    }
    bf->info();
}

neural_map::~neural_map()
{

}

void neural_map::R(float a)
{
    set_neural_states(0,1,&a);
    bf->opencl_pay_neuron(N.at(0)->BAL,N.at(0)->W,S.at(position(0,0))->SW,0,a);
    S.at(position(0,0))->R(a);
}

void neural_map::I(float * I)
{
    set_neural_states(1,ni,I);
}

void neural_map::O(float * O)
{
    get_neural_states(1 + ni,no,O);
}

void neural_map::U(seconds t)
{
    cout << "neural_map::U(" << t << ")" << endl;
    double a = now() + t;
    cout << "\tTime remaining: " << a - now() << "s" << endl;
    while(a - now() > 0){
        update_synaps_map(a - now());
        cout << "\tTime remaining: " << a - now() << "s" << endl;
    }
    cout << "neural_map::U(" << t << ")::done" << endl;
}
void neural_map::get_neural_states(size_t start, size_t l, float *A)
{
    N.get(start,l,A);
}

void neural_map::set_neural_states(size_t start, size_t l, float *A)
{
    N.set(start,l,A);
}

position neural_map::get_highest_bid()
{
    S.update();
    return S.maxbid();
}

seconds neural_map::update_synaps_map(seconds limit)
{
    position p = get_highest_bid();
    if(timeOfCost(S.at(p)->cost()) < limit) return S.at(p)->update();
    else return limit;
}


/////////////// SYNAPSBLOCK \\\\\\\\\\\\\\\\\

SynapsBlock::SynapsBlock(NeuronBlock *N,size_t i,size_t j):N(N),x(i),y(j),state(GPU)
{
    SINFO = bf->gpu_malloc(SINFO,SBSIZE,16);
    SP0 = bf->gpu_malloc(SP0,SBSIZE,0.41);
    SP1 = bf->gpu_malloc(SP1,SBSIZE,0.59);
    STMP = bf->gpu_malloc(STMP,SBSIZE,0);
    SP00 = bf->gpu_malloc(SP00,SBSIZE,0.41);
    SP01 = bf->gpu_malloc(SP01,SBSIZE,0.59);
    SP10 = bf->gpu_malloc(SP10,SBSIZE,0.59);
    SP11 = bf->gpu_malloc(SP11,SBSIZE,0.41);
    SP = bf->gpu_malloc(SP,SBSIZE,0.5);
    SBAL = bf->gpu_malloc(SBAL,SBSIZE,SYNAPSINITBAL);
    SBET0 = bf->gpu_malloc(SBET0,SBSIZE,0);
    SBET1 = bf->gpu_malloc(SBET1,SBSIZE,0);
    SW = bf->gpu_malloc(SW,SBSIZE,0);
    cout << "\t\tSB(" << position(x,y) << ") created, state = " << state << endl;
}
/*
    SynapsBlock::SynapsBlock():state(GPU)
{
    SINFO = bf->gpu_malloc(SINFO,SBSIZE,16);
    SP0 = bf->gpu_malloc(SP0,SBSIZE,0.41);
    SP1 = bf->gpu_malloc(SP1,SBSIZE,0.59);
    STMP = bf->gpu_malloc(STMP,SBSIZE,0);
    SP00 = bf->gpu_malloc(SP00,SBSIZE,0.41);
    SP01 = bf->gpu_malloc(SP01,SBSIZE,0.59);
    SP10 = bf->gpu_malloc(SP10,SBSIZE,0.59);
    SP11 = bf->gpu_malloc(SP11,SBSIZE,0.41);
    SP = bf->gpu_malloc(SP,SBSIZE,0.5);
    SBAL = bf->gpu_malloc(SBAL,SBSIZE,SYNAPSINITBAL);
    SBET0 = bf->gpu_malloc(SBET0,SBSIZE,0);
    SBET1 = bf->gpu_malloc(SBET1,SBSIZE,0);
    SW = bf->gpu_malloc(SW,SBSIZE,0);
    cout << "\t\tSB(" << position(x,y) << ") created, state = " << state << endl;
}
*/
SynapsBlock::~SynapsBlock(){
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
    cout << "\t\tSB(" << position(x,y) << ") removed, state = " << state << endl;
}
seconds SynapsBlock::update()
{
//    seconds d = timeOfCost(cost());
//    cout << "\t\tSB[" << position(x,y) << "]::U.inState("<<state<<") = " << d << endl;

    if(state != GPU)
	cerr << "\t\timplement moving to GPU" << endl;

    bf->opencl_synaps_refresh(SP0,SP01,SP00,N->P);
    bf->opencl_synaps_refresh(SP1,SP11,SP10,N->P);

    bf->opencl_synaps_bet(SBET1,SBAL,SP1,SP);
    bf->opencl_synaps_bet(SBET0,SBAL,SP0,SP);

    //printS();

    bf->opencl_bet_sum(N->BET1,SBET1);
    bf->opencl_bet_sum(N->BET0,SBET0);

    bf->opencl_neuron_refresh(N->BET1,N->BET0,N->P);

    return 1;
}

void SynapsBlock::R(float v){
    //printW();
    for(int i = 0; i< ITERS*(x==y) + 1; ++i){
        bf->opencl_find_winning_synapses(SBET0,SBET1,N->BET0,N->BET1,N->W,SW);
        //printW();
        bf->wait();
        bf->opencl_find_winning_neurons(N->W,SW);
        //printW();
    }

    bf->opencl_synaps_learn(SP11,SBET1,SBET0,SW,N->BET1,N->BET0,SINFO);
    bf->opencl_synaps_learn(SP10,SBET1,SBET0,SW,N->BET0,N->BET1,SINFO);
    bf->opencl_synaps_learn(SP01,SBET0,SBET1,SW,N->BET1,N->BET0,SINFO);
    bf->opencl_synaps_learn(SP00,SBET0,SBET1,SW,N->BET0,N->BET1,SINFO);

    bf->opencl_synaps_learn2(SP,SW,SINFO);

    bf->opencl_update_synaps_info(SINFO);

//    print();
    bf->opencl_pay(SBET0,SBET1,STMP,N->BET0,N->BET1,SBAL,N->BAL,N->TMP,SW,N->W);
//    printS();
}

size_t SynapsBlock::size(){
	return SBSIZE*12*sizeof(float);
}
neur SynapsBlock::cost()
{
    neur ret = stateStayCosts[state]*size()*10e-15 + costOfTime((size()/stateBandwidth[state] + size()/stateChangeSpeed[4*state + GPU])*10e-9);
    cout << "\t\tSB[" << position(x,y) << "]::c.instate("<<state<<")::cost = " << ret << "n€" << endl;
    cout << "\t\tSB[" << position(x,y) << "]::c.instate("<<state<<")::timeOfCost = " << timeOfCost(ret) << "s" << endl;
    return ret;
}

Synapses::Synapses(Neurons *N):N(N)
{
    cout << "\t\tS()" << endl;
}
void Synapses::addBlock(SynapsBlock *S,position p){
    SB[p] = S;
}
NeuronBlock *Neurons::at(size_t p)
{
	if(N.find(p) == N.end())
    {
        cerr << "\t\tadding a new neuron block ( N[" << p << "] )" << endl;
        N.insert(pair<size_t,NeuronBlock*>(p,new NeuronBlock()));
    }
	return N.at(p);
}
SynapsBlock *Synapses::at(position p)
{
	if(SB.find(p) == SB.end())
    {
        cerr << "\t\tadding a new synaps block ( SB[" << p << "] )" << endl;
        SB.insert(pair<position,SynapsBlock*>(p,new SynapsBlock(N->at(p.first),p.first,p.second)));
    }
	return SB.at(p);
}
seconds SynapsBlock::expected()
{
//    cout << "\t\tSB::e() = ?? What is the expected win of a block?" << endl;    
    return 10 - cost();
}
void SynapsBlock::kill(long s, long vertical, long l){
    bf->opencl_synaps_die(SBAL,SP1,s,vertical,l);
}

//////////// NEURONBLOCK \\\\\\\\\\\\

NeuronBlock::NeuronBlock():state(GPU)
{
    //cout << "\t\tNeuronBlock()::init" << endl;
    P = bf->gpu_malloc(P,NBSIZE,0.5);
    TMP = bf->gpu_malloc(TMP,NBSIZE);
    BAL = bf->gpu_malloc(BAL,NBSIZE,NEURONINITBAL);
    BET0 = bf->gpu_malloc(BET0,NBSIZE,0);
    BET1 = bf->gpu_malloc(BET1,NBSIZE,0);
    W = bf->gpu_malloc(W,NBSIZE,0);
    L = bf->gpu_malloc(L,NBSIZE,0);
    cout << "\t\tNB(x) created, state = " << state << endl;
}
NeuronBlock::~NeuronBlock()
{
    //cout << "\t\tNeuronBlock()::delete" << endl;
    bf->gpu_free(P);
    bf->gpu_free(TMP);
    bf->gpu_free(BAL);
    bf->gpu_free(BET0);
    bf->gpu_free(BET1);
    bf->gpu_free(W);
    bf->gpu_free(L);
    cout << "\t\tNB(x) removed, state = " << state << endl;
}
size_t NeuronBlock::size(){
    return NBSIZE;
}

float NeuronBlock::setState(statetype s){
    float cost = size()/stateChangeSpeed[4*state + s];
    cerr << "\t\tImplement hdd- and sdd-writing" << endl;
    return cost;
}
/////////// SYNAPSES \\\\\\\\\\\\

void Synapses::update()
{

    cout << "\tS::U()" << endl;
    cout << "\tdone" << endl;

}

position Synapses::maxbid()
{
//    if(VERBOSE){int val = 1; cin >> val; assert(val > 0);}
    cout << "\tS::maxbid() should get the SB that predicts to be most useful for the brain (maxbid() = argmax(P - C))" << endl;
    smap::iterator min = SB.begin(); 
    float minexp = -10e10;
    for(smap::iterator it = SB.begin(); it != SB.end(); ++it){
        float nowexp = it->second->expected();
        if(nowexp > minexp) { minexp = nowexp; min = it; }
    }
    cout << "\tdone" << endl;
    return min->first;
}

/////////// NEURONS \\\\\\\\\\\\

void Neurons::addBlock(){
        N[N.size()] = new NeuronBlock();
}
    Neurons::Neurons()
{
    cout << "\tNeurons()" << endl;
}

int min(int a,int b){ if(a < b) return a; return b; }

void Neurons::get(size_t start,int l,float *A)
{
    int i = start/NBSIZE;
    int j = start - NBSIZE*i;
//    cout << "\tNeurons[" << i << "]::P.get( " << j <<","<< min(j+l,NBSIZE) << ","<<(void *) A<<" )" << endl;
    bf->opencl_getv(N.at(i)->P,A,j,min(j+l,NBSIZE));

    l -= (NBSIZE - j);
    A += (NBSIZE - j);
    for(++i; l > 0; ++i){
//        cout << "\tNeurons[" << i << "]::P.get( " << 0 <<","<< min(j+l,NBSIZE) << ","<<(void *) A<<" )" << endl;
        bf->opencl_getv(N.at(i)->P,A,0,min(l,NBSIZE));
        l -= NBSIZE;
        A += NBSIZE;
    }
}

void Neurons::set(size_t start,int l,float *A)
{
    int i = start/NBSIZE;
    int j = start - NBSIZE*i;
//    cout << "\tNeurons[" << i << "]::P.set( " << j <<","<< min(j+l,NBSIZE) << ","<<(void *) A<<" )" << endl;
    bf->opencl_setv(N.at(i)->P,A,j,min(j+l,NBSIZE));

    l -= (NBSIZE - j);
    A += (NBSIZE - j);
    for(++i; l > 0; ++i){
//        cout << "\tNeurons[" << i << "]::P.set( " << 0 <<","<< min(j+l,NBSIZE) << ","<<(void *) A<<" )" << endl;
        bf->opencl_setv(N.at(i)->P,A,0,min(l,NBSIZE));
        l -= NBSIZE;
        A += NBSIZE;
    }
}




//////////////HELPERS\\\\\\\\\\\\\



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

void Neurons::printN(){
    cout << "\tSynapsBlock::N()" << endl;
    cout << "\t\tN.P" << endl;
    bf->print(N[0]->P,NBSIZE,1);
    cout << "\t\tN->BAL" << endl;
    bf->print(N[0]->BAL,NBSIZE,1);
}
void SynapsBlock::printW(){
    cout << "\tSynapsBlock::W()" << endl;
    cout << "\t\tS.W" << endl;
    bf->print(SW,NBSIZE,NBSIZE);
    cout << "\t\tN->W" << endl;
    bf->print(N->W,NBSIZE,1);

    cout << "\t\tn : ";
    bf->print(SINFO,1,1);
}
void SynapsBlock::printS(){
    cout << "\tSynapsBlock::S()" << endl;
    cout << "\t\tSBAL" << endl;
    bf->print(SBAL,NBSIZE,NBSIZE);
}
void SynapsBlock::print(){
    cout << "\tSynapsBlock("<< bf->opencl_sum(N->BAL,NBSIZE) << " + " << bf->opencl_sum(SBAL,NBSIZE*NBSIZE) << " = " << bf->opencl_sum(N->BAL,NBSIZE) + bf->opencl_sum(SBAL,NBSIZE*NBSIZE) << ")" << endl;
    cout << "\t\tS.P00" << endl;
    bf->print(SP00,NBSIZE,NBSIZE);
    cout << "\t\tS.P01" << endl;
    bf->print(SP01,NBSIZE,NBSIZE);
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
    bf->print(N->BET0,NBSIZE,1);
    cout << "\t\tN.BET1" << endl;
    bf->print(N->BET1,NBSIZE,1);
    cout << "\t\tN.BAL" << endl;
    bf->print(N->BAL,NBSIZE,1);
    cout << "\t\tN.W"   << endl;
    bf->print(N->W,NBSIZE,1);
    cout << "\t\tN.P"   << endl;
    bf->print(N->P,NBSIZE,1);
    cout << "\t\tn"     << endl;
    bf->print(SINFO,1,1);
}

