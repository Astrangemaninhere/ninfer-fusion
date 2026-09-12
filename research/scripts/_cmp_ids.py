a = open('/home/user/bench/cmp.qwen3_8_27b_nvfp4.ids').read().split()
b = open('/home/user/bench/cmp.qwen3_8_27b_nvfp4_foldmlp.ids').read().split()
n = min(len(a), len(b))
same = sum(1 for i in range(n) if a[i] == b[i])
print(f'baseline ids: {len(a)}, fold ids: {len(b)}')
print(f'compared {n}: identical {same} ({same/max(n,1)*100:.2f}%)')
# first divergence position
for i in range(n):
    if a[i] != b[i]:
        print(f'first divergence at token {i}: base={a[i]} fold={b[i]}')
        break
else:
    print('no divergence in compared range')
