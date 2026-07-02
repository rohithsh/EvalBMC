
int main()
{
    int z1,z2,z3;
    int x = 1;
    int m = 1;
    int n;
  n = __VERIFIER_nondet_int();

    while (x < n) {
        if (__VERIFIER_nondet_bool()) {
            m = x;
        }
        x = x + 1;
    }

    if(n > 1) {
       //assert (m < n);
       assert (m >= 1);
    }
}
