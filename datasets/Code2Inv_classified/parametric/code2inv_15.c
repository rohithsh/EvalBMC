

int main()
{
    int x = 0;
    int m = 0;
    int n;
  n = __VERIFIER_nondet_int();

    while (x < n) {
        if (__VERIFIER_nondet_bool()) {
            m = x;
        }
        x = x + 1;
    }

    if(n > 0) {
       assert (m < n);
       //assert (m >= 0);
    }
}
