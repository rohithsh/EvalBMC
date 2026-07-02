#include <assert.h>
int main() {
  // variable declarations
  int c;
  int n;
  int v1;
  int v2;
  int v3;
  n = __VERIFIER_nondet_int();
  // pre-conditions
  (c = 0);
  __VERIFIER_assume((n > 0));
  // loop body
  while (__VERIFIER_nondet_bool()) {
    {
      if ( __VERIFIER_nondet_bool() ) {
        if ( (c > n) )
        {
        (c  = (c + 1));
        }
      } else {
        if ( (c == n) )
        {
        (c  = 1);
        }
      }

    }

  }
  // post-condition
if ( (c != n) )
assert( (c >= 0) );
}
