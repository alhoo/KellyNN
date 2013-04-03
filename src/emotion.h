#ifndef BRAINE_HH
#define BRAINE_HH
#include <iostream>


using namespace std;
class emotion{
    IO io;
    public:
        float r;
        emotion	 	(IO &io):io(io){};
        float R		(){return r;};
        void in		(){};
        void update	(){};
        ~emotion	(){};
};



#endif
