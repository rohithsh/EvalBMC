#include <assert.h>
int main() {
  // variable declarations
  int x;
  int y;
  y = __VERIFIER_nondet_int();
  // pre-conditions
  (x = 1);
  // loop body
  while ((x < y)) {
    {
    (x  = (x + x));
    }

  }
  // post-condition
assert( (x >= 1) );
}
