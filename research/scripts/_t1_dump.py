import numpy as np
import glob
import os
import struct

D = '/home/user/dspark-dump'
for name in sorted(os.listdir(D)):
    p = os.path.join(D, name)
    n = os.path.getsize(p)
    if 'ids' in name or 'positions' in name:
        a = np.fromfile(p, dtype=np.int32)
        print('%-40s n=%-6d bytes=%-8d %s' % (name, a.size, n, a.tolist()))
    elif 'frontiers' in name or 'fullrows' in name or 'lanes' in name:
        a = np.fromfile(p, dtype=np.int32)
        print('%-40s n=%-6d bytes=%-8d %s' % (name, a.size, n, a.tolist()))
    elif 'context' in name:
        a = np.fromfile(p, dtype=np.uint16)
        print('%-40s n=%-6d bytes=%-8d elems=%d (=%d cols of 5120)' % (name, a.size, n, a.size, a.size / 5120))
    elif 'residual' in name:
        a = np.fromfile(p, dtype=np.uint16)
        print('%-40s n=%-6d bytes=%-8d elems=%d (=%d cols of 5120)' % (name, a.size, n, a.size, a.size / 5120))
    else:
        print('%-40s bytes=%-8d (unparsed)' % (name, n))
