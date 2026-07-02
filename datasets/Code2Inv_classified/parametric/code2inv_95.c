#include <assert.h>
int main() {
  // variable declarations
  int i;
  int j;
  int x;
  int y;
  x = __VERIFIER_nondet_int();
  // pre-conditions
  (j = 0);
  (i = 0);
  (y = 1);
  // loop body
  while ((i <= x)) {
    {
    (i  = (i + 1));
    (j  = (j + y));
    }

  }
  // post-condition
if ( (y == 1) )
assert( (i == j) );

}
