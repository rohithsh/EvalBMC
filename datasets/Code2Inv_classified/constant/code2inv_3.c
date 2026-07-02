
int main()
{
    int x = 0;
    int y, z;
  z = __VERIFIER_nondet_int();

    while(x < 5) {
       x += 1;
       if( z <= y) {
          y = z;
       }
    }

    assert (z >= y);
}
