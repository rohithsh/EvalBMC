#include <assert.h>
int main() {
  // variable declarations
  int i;
  int x;
  int y;
  int z1;
  int z2;
  int z3;
  x = __VERIFIER_nondet_int();
  y = __VERIFIER_nondet_int();
  // pre-conditions
  (i = 0);
  __VERIFIER_assume((x >= 0));
  __VERIFIER_assume((y >= 0));
  __VERIFIER_assume((x >= y));
  // loop body
  while (__VERIFIER_nondet_bool()) {
    if ( (i < y) )
    {
    (i  = (i + 1));
    }

  }
  // post-condition
if ( (i < y) )
assert( (0 <= i) );
}
