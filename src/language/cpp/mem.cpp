#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
using namespace std;

uint32_t mem[10] = {0x12345678, 0x87654321, 0x12345678};

#define inst_id  (pc - 0x80000000)/4
int main() {
  uint32_t pc = 0x80000000;
  
  while(inst_id < 10){
    printf("%x\n", mem[inst_id]);
    pc += 4;
  }

}