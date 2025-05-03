#include "io.c"

union Data {
  int i;
  char str[20];
};

int main () {
  union Data data;

  data.i = 10;
  printf("data.i: %d\n", data.i);

  strcpy(data.str, "C Programming");
  printf("data.str: %s\n", data.str);

  return 0;
}
