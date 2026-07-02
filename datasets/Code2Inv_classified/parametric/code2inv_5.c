int main()
{
    int x = 0;
    int size;
    int y, z;
  size = __VERIFIER_nondet_int();
  z = __VERIFIER_nondet_int();

    while(x < size) {
       x += 1;
       if( z <= y) {
          y = z;
       }
    }

    if(size > 0) {
       assert (z >= y);
    }
}
