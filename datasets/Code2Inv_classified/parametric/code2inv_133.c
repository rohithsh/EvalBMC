#include <assert.h>
int main() {
  // variable declarations
  int n;
  int x;
  n = __VERIFIER_nondet_int();
  // pre-conditions
  (x = 0);
  __VERIFIER_assume((n >= 0));
  // loop body
  while ((x < n)) {
    {
    (x  = (x + 1));
    }

  }
  // post-condition
assert( (x == n) );
}
