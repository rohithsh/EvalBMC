#include <assert.h>
int main() {
  // variable declarations
  int c;
  int x1;
  int x2;
  int x3;
  int y;
  int z;
  y = __VERIFIER_nondet_int();
  // pre-conditions
  (c = 0);
  __VERIFIER_assume((y >= 0));
  __VERIFIER_assume((y >= 127));
  (z = (36 * y));
  // loop body
  while (__VERIFIER_nondet_bool()) {
    if ( (c < 36) )
    {
    (z  = (z + 1));
    (c  = (c + 1));
    }

  }
  // post-condition
if ( (c < 36) )
assert( (z >= 0) );
}
