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
        size_t np	(){return 1;};
        size_t ni	(){return 3;};
        size_t no	(){return 3;};
        size_t nc	(){return 1;};
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
