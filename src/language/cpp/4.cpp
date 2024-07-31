#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
using namespace std;


/* logical right shift and arithmetic right shift test */
/* conclusion: unsigned use logical shift, signed use arithmetic shift */
int main() {
  uint64_t a = -1;
  printf("a: %lx\n", a);
  uint64_t tempa1 = a >> 1;
  printf("tempa1: %lx\n", tempa1);
  uint64_t tempa2;
  tempa2 = (int64_t) a >> 1;
  printf("tempa2: %lx\n", tempa2);


  int64_t b = -1;
  printf("b: %lx\n", b);
  b = b >> 1;
  printf("b: %lx\n", b);
}