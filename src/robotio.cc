#include "robotio.h"

IO::IO(){
    I_ = new float[256];
    O_ = new float[64];
    cout << "I = " << (void*)I_ << endl;
    cout << "O = " << (void*)O_ << endl;
}
IO::~IO(){
//delete I_;
//delete O_;
}
