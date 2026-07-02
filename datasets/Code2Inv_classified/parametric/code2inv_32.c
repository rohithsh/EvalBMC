#include <assert.h>
int main() {
  // variable declarations
  int n;
  int v1;
  int v2;
  int v3;
  int x;
  n = __VERIFIER_nondet_int();
  // pre-conditions
  (x = n);
  // loop body
  while ((x > 1)) {
    {
    (x  = (x - 1));
    }

  }
  // post-condition
if ( (n >= 0) )
assert( (x == 1) );

}
