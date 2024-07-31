#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
using namespace std;


int main(){
  uint64_t a = -1;
  void *p = &a;

  cout << *(uint64_t *)p << endl;
  cout << *(uint32_t *)p << endl;
  cout << *(uint16_t *)p << endl;
  cout << *(uint8_t *)p << endl;

  cout << *(int64_t *)p << endl;
  cout << *(int32_t *)p << endl;
  cout << *(int16_t *)p << endl;
  cout << *(int8_t *)p << endl;

  uint64_t b = *(uint8_t *)p;
  int64_t c = *(uint8_t *)p;
  cout << b << endl;
  cout << c << endl;
}