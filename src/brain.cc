#include "brain.h"

void Brain::R   (float a)
{
    N.R(a);
}


Brain::Brain	 	(IO io, emotion e):io(io),e(e),N(io.ni(),io.no())
{

}

void Brain::in  	()
{
    cout << "B::in()" << endl;
    N.I(io.I());
    N.R(e.R());
    //R(e.R());
}

void Brain::out 	()
{
    cout << "B::out()" << endl;
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


