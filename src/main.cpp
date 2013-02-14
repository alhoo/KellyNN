#include <iostream>
#include "brain.h"
#include "emotion.h"
#include "robotio.h"

int main(){
//There is a connection to the world (IO)
	IO io;
//There is a source of 'emotions' or metric of 'happiness'.
//This can be simple as 'happiness' = 'percentage of frames during the last 5 seconds with a smiling face in front of the camera');
	emotion em(io);
//The brain or mind is a system that learns to understand the connection between the world and happiness.
//This means that there is a relationship between inputs, outputs and a one dimensional metric describing happiness or something similar.
//The brain acts as to maximize the happiness defined by the emotion.
	Brain b(io,em);
	unsigned long long alive = 1;
//The brain is constrained by a limit to how fast it can calculate, process and learn.
//The state of the world changes every 40ms ( 40ms = 1/25s = 40 * 10^-3 = 40*10^6*10^-9s = 40000000 ns ). This is the time given to the brain per cycle.
	while(alive){
		++alive;
		io.update();
		em.in();
		em.update();
		b.in();
		b.update(0.03);
		//b.update(40000000.0);
		b.out();
        if(alive > 10) break;
	}
	return 0;
}
