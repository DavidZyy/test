#include <stdio.h>
#include <inttypes.h>

#define SEXT(x, len) ({ struct { int64_t n : len; } __x = { .n = x }; (uint64_t)__x.n; })
/* zero extension */
#define ZEXT(x, len) ({ struct { uint64_t n : len; } __x = { .n = x }; (uint64_t)__x.n; })

int main() {
    // uint64_t a = 0xf; 
    // uint64_t b = SEXT(a, 4);
    // uint64_t c = ZEXT(a, 4);
    // printf("a: %lx, b: %lx, c: %lx\n", a, b, c);

    // uint32_t a = 0xf; 
    // uint32_t b = SEXT(a, 4);
    // uint32_t c = ZEXT(a, 4);

    uint32_t a = 0x10000000;
    uint32_t b = 0x10000000;
    // uint64_t a = 0x10000000;
    // uint64_t b = 0x10000000;
    uint32_t c = a*b>>32;
    uint32_t d = a>>16 * b>>16;
    // uint64_t c = a*b>>32;
    // printf("a: %lx, b: %lx, c: %lx\n", a, b, c);
    printf("a: %x, b: %x, c: %x, d: %x\n", a, b, c, d);
}
