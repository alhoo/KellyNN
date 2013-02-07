#ifndef BRAINE_HH
#define BRAINE_HH
#include <iostream>


using namespace std;
class emotion{
    IO io;
    public:
        emotion	 	(IO &io):io(io){};
	float R		(){};
	void in		(){};
        void update	(){};
        ~emotion	(){};
};



#endif
