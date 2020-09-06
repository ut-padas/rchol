
all: driver


driver: test_laplace_matrix.cpp
	g++ -Wall -O3 $< -o $@

run: driver
	./driver

clean:
	rm -f driver

