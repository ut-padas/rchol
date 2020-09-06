#include <iostream>
#include <string>
#include <cstring>
#include "laplace_3d.hpp"

template <typename T>
void print(const T &x, std::string name) {
  std::cout<<name<<":"<<std::endl;
  for (size_t i=0; i<x.size(); i++)
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
  
  std::vector<int> rowPtrU;
  std::vector<int> colIdxU;
  std::vector<float> valU;
  triu_csr(rowPtr, colIdx, val, rowPtrU, colIdxU, valU);
  print(rowPtrU, "rowPtrU");
  print(colIdxU, "colIdxU");
  print(valU, "valU");
  
  return 0;
}

