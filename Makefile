SOURCES=src/brain.cc src/brain_functions.cc src/main.cc src/neural_map.cc src/timer.cc src/robotio.cc
CC=g++
LIBS=-lOpenCL -lrt
FLAGS=-DDEBUG=0

all: bin/test

bin/debug: $(SOURCES)
	$(CC) $^ $(LIBS) -g -DDEBUG -DVERBOSE -o $@

bin/test: $(SOURCES)
	$(CC) $^ $(LIBS) $(FLAGS) -g -DVERBOSE -o $@

bin/brain: $(SOURCES)
	$(CC) $^ $(LIBS) $(FLAGS) -DVERBOSE=0 -O2 -o $@
