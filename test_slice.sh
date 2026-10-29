D=datasets/cleaned/svcomp_clean

# clean the slice so CBMC can compile it
python3 - <<'EOF'
import re
src = open('/tmp/sc3.c').read()
src = re.sub(r'#include "__fc_builtin\.h"\n', '', src)
src = re.sub(r'int __VERIFIER_nondet_int_slice_1\(void\)\s*\{.*?\n\}\n', '', src, flags=re.S)
src = src.replace('__VERIFIER_nondet_int_slice_1', '__VERIFIER_nondet_int')
src = 'extern int __VERIFIER_nondet_int(void);\n' + src
open('/tmp/sc3_clean.c','w').write(src)
EOF

# k* on the original
echo "=== original"
for k in $(seq 0 30); do
  r=$(cbmc $D/array-fpi/brs1.c --32 --unwind $k --unwinding-assertions \
        --no-standard-checks --json-ui 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
f=[r.get('property','') for it in d if isinstance(it,dict) and 'result' in it
   for r in it['result'] if r.get('status')=='FAILURE']
unw=[x for x in f if '.unwind.' in x]
oth=[x for x in f if '.unwind.' not in x]
print('bug' if (oth and not unw) else ('deeper' if unw else 'safe'))
")
  if [ "$r" != "deeper" ]; then echo "k=$k -> $r"; break; fi
done

# k* on the slice
echo "=== slice"
for k in $(seq 0 30); do
  r=$(cbmc /tmp/sc3_clean.c --32 --unwind $k --unwinding-assertions \
        --no-standard-checks --json-ui 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
f=[r.get('property','') for it in d if isinstance(it,dict) and 'result' in it
   for r in it['result'] if r.get('status')=='FAILURE']
unw=[x for x in f if '.unwind.' in x]
oth=[x for x in f if '.unwind.' not in x]
print('bug' if (oth and not unw) else ('deeper' if unw else 'safe'))
")
  if [ "$r" != "deeper" ]; then echo "k=$k -> $r"; break; fi
done

wc -l $D/array-fpi/brs1.c /tmp/sc3_clean.c