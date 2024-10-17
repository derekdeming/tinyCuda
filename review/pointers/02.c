#include <stdio.h> // this is the standard IO header file (for printf)

int main() {
    int value = 42;
    int* ptr1 = &value; 
    int** ptr2 = &ptr1; // ptr2 is a pointer to a pointer to an int
    int*** ptr3 = &ptr2; // ptr3 is a pointer to a pointer to a pointer to an int

    printf("value = %d\n", value);

    printf("ptr1 = %p\n", ptr1);
    printf("ptr2 = %p\n", ptr2);
    printf("ptr3 = %p\n", ptr3);

    printf("*ptr1 = %d\n", *ptr1); // this is the value at address ptr1 which is 42
    printf("*ptr2 = %p\n", *ptr2); // this is the value of the address ptr2 which is the address of ptr1
    printf("*ptr3 = %p\n", *ptr3); // this is the value of the address ptr3 which is the address of ptr2
}