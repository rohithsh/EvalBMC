
int main() {
    int i = 0;
    int j, c, t;
  c = __VERIFIER_nondet_int();

    while( __VERIFIER_nondet_bool() ) {
        if(c > 48) {
            if (c < 57) {
                j = i + i;
                t = c - 48;
                i = j + t;
            }
        }
    } 
    assert (i >= 0);
}
