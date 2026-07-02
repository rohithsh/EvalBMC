#include <assert.h>

int main() {
    int a,m,j,k;
  a = __VERIFIER_nondet_int();
  j = __VERIFIER_nondet_int();

    __VERIFIER_assume(a <= m);
    __VERIFIER_assume(j < 1);
    k = 0;

    while ( k < 1) {
        if(m < a) {
            m = a;
        }
        k = k + 1;
    }

    assert( a >= m);
}
