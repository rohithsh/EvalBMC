#include <assert.h>
int main() {
  // variable declarations
  int x;
  int y;
  // pre-conditions
  __VERIFIER_assume((x >= 0));
  __VERIFIER_assume((x <= 10));
  __VERIFIER_assume((y <= 10));
  __VERIFIER_assume((y >= 0));
  // loop body
  while (__VERIFIER_nondet_bool()) {
    {
    (x  = (x + 10));
    (y  = (y + 10));
    }

  }
  // post-condition
if ( (y == 0) )
assert( (x != 20) );

}
