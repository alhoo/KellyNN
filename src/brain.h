#ifndef BRAIN_HH
#define BRAIN_HH
#include <iostream>
#include <map>
#include "robotio.h"
#include "emotion.h"
#include "neural_map.h"

using namespace std;
typedef float nanoseconds;

class Brain{
    neural_map N;
    IO io;
    emotion e;
    public:
        Brain	 	(IO io, emotion e);
        void in  	();
        void out 	();
        void update	(nanoseconds t);
        ~Brain		();
};

#endif
