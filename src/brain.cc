#include "brain.h"


Brain::Brain	 	(IO &io, emotion &e):io(io),e(e),N(io.np(),io.ni(),io.no(),io.nc())
{

}

void Brain::in  	()
{
    cout << "B::in("<<io.I()<<")" << endl;
    N.I(io.I());
    N.R(e.R());
}

void Brain::out 	()
{
    cout << "B::out("<<io.O()<<")" << endl;
    N.O(io.O());

}

void Brain::update	(nanoseconds t)
{
    cout << "B::update()" << endl;
    N.U(t);
}

Brain::~Brain		()
{

}


