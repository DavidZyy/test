#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <any>
#include <vector>

int choose(int n) {
  return rand() % n;
}

int main() {
//     char hexString[] = "0xff";  // Example hexadecimal string
//     char* endPtr;
//     long decimal = strtol(hexString, &endPtr, 16);
// 
//     if (*endPtr != '\0') {
//         printf("Invalid hexadecimal string\n");
//     } else {
//         printf("Decimal: %ld\n", decimal);
//     }
    // char a[] = "0101";
    // int b = atoi(a);
    int seed = time(0);
    srand(seed);
    int n = choose(3);
    
    printf("%u\n", (1<<31) - 1 + (1<<31));

    return 0;
}
