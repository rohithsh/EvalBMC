#include <assert.h>

int main() {
    int a,c,m,j,k;
  a = __VERIFIER_nondet_int();
  c = __VERIFIER_nondet_int();

    __VERIFIER_assume(a <= m);
    j = 0;
    k = 0;

    while ( k < c) {
        if(m < a) {
            m = a;
        }
        k = k + 1;
    }

    assert( a <=  m);
}
