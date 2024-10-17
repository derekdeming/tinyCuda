#include <stdio.h>

int main() {
    int num = 10; 
    float fnum = 3.14;
    void* vptr;

    vptr = &num; 
    printf("integer: %d\n", *(int*)vptr); // output: 10
    // vptr is a memory address "&num" but it is stored as a void pointer (no data type)
    // we cannot dereference a void pointer, so we cast it to an integer pointer to store the integer value 
    // at that memory address "(int*)vptr"
    // so we need to cast it to an int pointer to access the value (we're dereferencing it with the final "*" to get 
    // the value at that memory address "*((*int*)vptr)
    vptr = &fnum;
    printf("float: %.2f\n", *(float*)vptr); // output: 3.14
}

// void pointers are useful when we don't know the data type of the value we're storing in memory
// we can use a void pointer to store the address of a value of any data type
// and then cast it to the appropriate data type when we need to access the value

// fun fact: malloc() returns a void pointer to the allocated memory
// so we can cast it to the appropriate data type when we need to access the value
// but we see it as a pointer to a specific data type after the cast (int*)malloc(4) or (float*)malloc(4) etc.