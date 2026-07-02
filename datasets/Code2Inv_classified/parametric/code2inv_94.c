#include <assert.h>
int main() {
  // variable declarations
  int i;
  int j;
  int k;
  int n;
  k = __VERIFIER_nondet_int();
  n = __VERIFIER_nondet_int();
  // pre-conditions
  __VERIFIER_assume((k >= 0));
  __VERIFIER_assume((n >= 0));
  (i = 0);
  (j = 0);
  // loop body
  while ((i <= n)) {
    {
    (i  = (i + 1));
    (j  = (j + i));
    }

  }
  // post-condition
assert( ((i + (j + k)) > (2 * n)) );
}
