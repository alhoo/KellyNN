#ifndef BRAINIO_HH
#define BRAINIO_HH
#include <iostream>

using namespace std;

class IO{
    float *I_,*O_;
    public:
        IO	 	();
        ~IO	 	();
        void update	(){};
        size_t ni	(){return 8;};
        size_t no	(){return 6;};
        float *I	(){
//    cout << "I = " << (void*)I_ << endl;
            return I_;
        };
        float *O	(){
//    cout << "O = " << (void*)O_ << endl;
            return O_;
        };
};


#endif
