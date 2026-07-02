#include <assert.h>

int main() {
    int n;
    int c = 0;
  n = __VERIFIER_nondet_int();
    __VERIFIER_assume (n > 0);

    while (__VERIFIER_nondet_bool()) {
        if(c == n) {
            c = 1;
        }
        else {
            c = c + 1;
        }
    }

    if(c == n) {
        //assert( c >= 0);
        assert( c <= n);
    }
}
