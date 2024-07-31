#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
using namespace std;


int main(){
  uint64_t a = 0xfedcba9876543210;
  uint32_t temp0 = a;
  uint16_t temp1 = a;
  uint8_t temp2 = a;

  printf("%lx\n", temp0);
  printf("%lx\n", temp1);
  printf("%lx\n", temp2);

  uint64_t  b;
  void *addr = &b;

  b = 0;
  *(uint64_t *)addr = a;
  printf("%lx\n", *(uint64_t *)addr);

  b = 0;
  *(uint32_t *)addr = a;
  printf("%lx\n", *(uint64_t *)addr);


  b = 0;
  *(uint16_t *)addr = a;
  printf("%lx\n", *(uint64_t *)addr);

  b = 0;
  *(uint8_t *)addr = a;
  printf("%lx\n", *(uint64_t *)addr);
}