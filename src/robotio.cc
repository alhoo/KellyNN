#include "robotio.h"

IO::IO(){
    I_ = new float[3];
    O_ = new float[3];
    cout << "I = " << (void*)I_ << endl;
    cout << "O = " << (void*)O_ << endl;
}
IO::~IO(){
//delete I_;
//delete O_;
}
