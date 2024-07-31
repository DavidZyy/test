#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
using namespace std;

int main(){
  int a = 4;
  int b = 5;
  const int * const p0 = &a;
  const int * p1 = &a;
  int * const p2 = &a;
  p1 = &b;
  *p2 = 5;
}