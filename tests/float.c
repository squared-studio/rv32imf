#include "io.c"

int main () {
  float x = 5.5, y = 2.2;
  printf("Addition: %.2f + %.2f = %.2f\n", x, y, x + y);
  printf("Subtraction: %.2f - %.2f = %.2f\n", x, y, x - y);
  printf("Multiplication: %.2f * %.2f = %.2f\n", x, y, x * y);
  printf("Division: %.2f / %.2f = %.2f\n", x, y, x / y);
  return 0;
}
