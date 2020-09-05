all: python_interface.cpp python_interface.so

python_interface.cpp: python_interface.pyx
	cython python_interface.pyx --cplus

python_interface.so: python_interface.cpp
	g++ python_interface.cpp -I/usr/include/python3.6m -I/usr/include/python3.6m/numpy -fPIC -shared -o python_interface.so -O3

clean:
	rm -f python_interface.cpp
	rm -f python_interface.so
