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
  (x = 0);
  // loop body
  while ((x < n)) {
    {
    (x  = (x + 1));
    }

  }
  // post-condition
if ( (x != n) )
assert( (n < 0) );

}
