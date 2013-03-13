SOURCES=src/brain.cc src/brain_functions.cc src/main.cc src/neural_map.cc src/timer.cc src/robotio.cc
CC=g++
LIBS=-lOpenCL -lrt
IOLIB=-lRobotIO
EMLIB=-lEmotions
FLAGS=

all: bin/test

bin/debug: $(SOURCES)
	$(CC) $^ $(LIBS) $(FLAGS) -g -DDEBUG -DVERBOSE -o $@

bin/test: $(SOURCES)
	$(CC) $^ $(LIBS) $(FLAGS) -g -DVERBOSE -o $@

bin/brain: $(SOURCES)
	$(CC) $^ $(LIBS) $(FLAGS) -DVERBOSE=0 -O2 -o $@
    
bin/connectedBrain: $(SOURCES)
	$(CC) $^ $(LIBS) $(IOLIB) $(EMLIB) $(FLAGS) -DVERBOSE=0 -O2 -o $@
