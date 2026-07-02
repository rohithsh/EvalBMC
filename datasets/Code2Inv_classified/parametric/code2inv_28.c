#include <assert.h>
int main() {
  // variable declarations
  int n;
  int x;
  n = __VERIFIER_nondet_int();
  // pre-conditions
  (x = n);
  // loop body
  while ((x > 0)) {
    {
    (x  = (x - 1));
    }

  }
  // post-condition
if ( (x != 0) )
assert( (n < 0) );

}
