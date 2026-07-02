#include <assert.h>
int main()
{
  unsigned int n;
  __VERIFIER_assume(n >= 0);
  int x=n, y=0;
  n = __VERIFIER_nondet_int();
  while(x>0)
  {
    x--;
    y++;
  }
  assert(y==n);
}
