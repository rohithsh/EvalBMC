#include <assert.h>
int main() {
  // variable declarations
  int c;
  // pre-conditions
  (c = 0);
  // loop body
  while (__VERIFIER_nondet_bool()) {
    {
      if ( __VERIFIER_nondet_bool() ) {
        if ( (c != 40) )
        {
        (c  = (c + 1));
        }
      } else {
        if ( (c == 40) )
        {
        (c  = 1);
        }
      }

    }

  }
  // post-condition
if ( (c < 0) )
if ( (c > 40) )
assert( (c == 40) );

}
