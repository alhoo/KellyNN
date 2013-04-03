#include "brain.h"


Brain::Brain	 	(IO &io, emotion &e):io(io),e(e),N(1,io.ni(),io.no())
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


