#include <assert.h>
int main() {
  // variable declarations
  int x;
  int y;
  int z1;
  int z2;
  int z3;
  // pre-conditions
  __VERIFIER_assume((x >= 0));
  __VERIFIER_assume((x <= 2));
  __VERIFIER_assume((y <= 2));
  __VERIFIER_assume((y >= 0));
  // loop body
  while (__VERIFIER_nondet_bool()) {
    {
    (x  = (x + 2));
    (y  = (y + 2));
    }

  }
  // post-condition
if ( (y == 0) )
assert( (x != 4) );

}
