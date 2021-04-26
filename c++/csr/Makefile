EXE = driver

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

CFLAG = -Wall -O3 -g

all: $(EXE)

%.o: %.cpp %.hpp
	icc -c $< -o $@ $(CFLAG)

%.o: %.cpp
	icc -c $< -o $@ $(CFLAG)

driver: $(OBJS)
	icc $^ -o $@ $(CFLAG)

clean:
	rm -f driver *.o

