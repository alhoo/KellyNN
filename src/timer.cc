#include "timer.hh"

timespec ts;

timespec totimespec(double s){
        timespec ret;
        ret.tv_sec = (int)s;
        ret.tv_nsec = (s - ret.tv_sec)*1e9;
        return ret;
}

double now(){
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec + (ts.tv_nsec)*1e-9;
}
