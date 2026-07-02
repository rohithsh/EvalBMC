#include <assert.h>
int main() {
  // variable declarations
  int i;
  int n;
  int x;
  int y;
  n = __VERIFIER_nondet_int();
  // pre-conditions
  __VERIFIER_assume((n >= 0));
  (i = 0);
  (x = 0);
  (y = 0);
  // loop body
  while ((i < n)) {
    {
    (i  = (i + 1));
      if ( __VERIFIER_nondet_bool() ) {
        {
        (x  = (x + 1));
        (y  = (y + 2));
        }
      } else {
        {
        (x  = (x + 2));
        (y  = (y + 1));
        }
      }

    }

  }
  // post-condition
assert( ((3 * n) == (x + y)) );
}
