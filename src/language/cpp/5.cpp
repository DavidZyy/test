#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
using namespace std;


/* truncate 64 bit to 32 bit(int32_t or uint32_t), 
than right shift it separately. To see it whether executes 
arithmetic shift right or logical shift right. */
int main(){
  uint64_t a = 0xfedcba9876543210;
  uint64_t b = 0xfedcba98f6543210;
  printf("%x\n", (int32_t)a);
  printf("%x\n", (uint32_t)a);

  printf("%x\n", (int32_t)b);
  printf("%x\n", (uint32_t)b);

  printf("%x\n", (int32_t)a >> 1);
  printf("%x\n", (uint32_t)a >> 1);

  printf("%x\n", (int32_t)b >> 1);
  printf("%x\n", (uint32_t)b >> 1);
}