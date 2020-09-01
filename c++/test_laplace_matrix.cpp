#include <iostream>
#include <string>
#include "laplace_3d.hpp"

template <typename T>
void print(const T &x, std::string name) {
  std::cout<<name<<":"<<std::endl;
  for (int i=0; i<x.size(); i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl;
}

int main(int argc, char *argv[]) {
  int n = 3;
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[i+1]);
  }
  std::vector<int> rowPtr;
  std::vector<int> colIdx;
  std::vector<float> val;
  laplace_3d(n, rowPtr, colIdx, val);

  //print(rowPtr, "rowPtr");
  //print(colIdx, "colIdx");
  //print(val, "val");
  return 0;
}

