#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
#include <stdio.h>
#include <string.h>
using namespace std;

struct {
   /* use int to sign extended */
   int64_t age : 1;
   /* use uint to zero extended */
   uint64_t age1 : 1;
} Age;

int main( ) {
//     unsigned int a;
//    printf( "Sizeof( Age ) : %ld\n", sizeof(Age) );
// 
//    Age.age = 3;
//    printf( "Age.age : %d\n", Age.age );
// 
//    a = Age.age;
//    cout <<"a: " << a <<endl;
// 
//    Age.age = 4;
//    printf( "Age.age : %d\n", Age.age );
// 
//    a = Age.age;
//    cout <<"a: " << a <<endl;
// 
//    Age.age = 7;
//    printf( "Age.age : %d\n", Age.age );
// 
//    a = Age.age;
//    cout <<"a: " << a <<endl;
// 
//    Age.age = 8;
//    printf( "Age.age : %d\n", Age.age );
// 
//    a = Age.age;
//    cout <<"a: " << a <<endl;
// 
//    Age.age = 9;
//    printf( "Age.age : %d\n", Age.age );
// 
//    a = Age.age;
//    cout <<"a: " << a <<endl;
    Age.age = 1;
    Age.age1 = 1;

    uint64_t a = Age.age;
    cout << "a: " << a << endl;
    cout << "(int64_t)a: " << (int64_t)a << endl;
    cout << "a < 1: " << (a < 1) <<endl;
    cout << "(int64_t)a < 1: " << ((int64_t)a < 1) <<endl;

    int64_t b = Age.age;
    cout << "b: " << b << endl;

    a = Age.age1;
    cout << "a: " << a << endl;

    b = Age.age1;
    cout << "b: " << b << endl;

   return 0;
}