#include <stdint.h>
int64_t fun1(int64_t a, int64_t b) { return a < b; }
uint64_t fun2(uint64_t a, uint64_t b) { return a < b; }

int main(){
  int64_t a = fun1(0xffffffff, 1);
  uint64_t b = fun1(0xffffffff, 1);
  return a+b;
}
