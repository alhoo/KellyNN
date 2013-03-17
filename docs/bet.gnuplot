
==Neuron betting behaviour as a function of balance==
INITBAL=1000
plot [0:6*INITBAL] (tanh(0.001*(x - 3*INITBAL)) + 1.0)*(x)/(16.0),x/8;

==Synaps betting behaviour as a function of balance==
INITBAL
plot [0:6*INITBAL] (tanh(0.001*(x - 3*INITBAL)) + 1)*x/8,x/4

