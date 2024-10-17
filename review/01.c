#include <stdio.h>

// & "address of" operator
// * "value at address" operator

int main() {
    printf("Hello, World!\n");
    int x = 42;
    int* ptr = &x; // here & is used to get the memory address of the variable x
    printf("address of x = %p\n", &x); // this is the address of i = 0x7ffee2b6e8c4
    printf("value of x = %d\n", x); // this is the value of i = 42

    printf("ptr = %p\n", ptr); // this is the address of ptr = 0x7ffee2b6e8c4
    printf("*ptr = %d\n", *ptr); // this is the value at address ptr = 42
    return 0;

    // i = 42
    // &i = 0x7ffee2b6e8c4
    // ptr = 0x7ffee2b6e8c4
    // *ptr = 42

    // where i is the actual value of i and &i is the address of i
    // ptr is a pointer to the address of i
    // *ptr is the value at the address of i
}