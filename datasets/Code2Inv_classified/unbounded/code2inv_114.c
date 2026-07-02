#include <assert.h>
int main() {
  // variable declarations
  int sn;
  int x;
  // pre-conditions
  (sn = 0);
  (x = 0);
  // loop body
  while (__VERIFIER_nondet_bool()) {
    {
    (x  = (x + 1));
    (sn  = (sn + 1));
    }

  }
  // post-condition
if ( (sn != x) )
assert( (sn == -1) );

}
