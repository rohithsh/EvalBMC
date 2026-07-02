#include <assert.h>
int main() {
  // variable declarations
  int x;
  int y;
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
if ( (x == 4) )
assert( (y != 0) );

}
