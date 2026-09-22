# -*- coding: utf-8 -*-
"""gguf_tensors.py - tensor scan (proven parse from gguf_debug).

Every read goes through :func:`transient_read` / :class:`_RetryingReader`.  This reader
is the *independent* cross-check for ``gguf_names.coverage()``, and on 2026-09-14 a
single transient ``OSError(ENOMEM)`` on an 8-byte read aborted the whole importer
self-test with an uncaught traceback (measured at line 18 of this file; the retry
belongs in the reader, not at each call site).
"""
import errno
import struct
import sys
import time
from collections import Counter

GGML_TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
              8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
              14: 'Q6_K', 15: 'Q8_K', 30: 'BF16',
              # Prism-private ids, read off the fork's own enum (PrismML-Eng/llama.cpp,
              # branch prism-v7, ggml/include/ggml.h:431 and :436).  This table is a
              # TWIN of tools/convert/gguf_kquant.py's, and gguf_kquant's
              # ``assert_type_table_agrees`` cross-checks the two -- but it compares
              # only the keys THIS table already has (it filters its own table down to
              # this one's), so an id that was missing here was invisible to the check:
              # this reader named a PQ2_0 tensor "T142" while the converter that decodes
              # the same bytes named it "PQ2_0".  Measured 2026-09-20: ``grep 142`` in
              # this file was zero hits.  The three names below are gguf_kquant's,
              # verbatim, so one file cannot get two answers.
              #   GGML_TYPE_Q1_0   = 41   -- 1-bit, one fp16 mean-|x| scale per 128 values
              #   GGML_TYPE_PTQ1_0 = 143  -- ternary, base-3 trits, one fp16 scale per 128
              #   GGML_TYPE_PQ2_0  = 142  -- Prism-private Q2_0 at group 128 (fork
              #                              ggml-common.h:202)
              # Upstream's 42 (Q2_0) is still NOT added, and not because it is
              # unknown: it carries TWO different block layouts across upstream
              # releases, so a table that picks one of them reads the other silently.
              # A wrong NAME is a refusal at load time; a wrong LAYOUT is a wrong number.
              41: 'Q1_0', 142: 'PQ2_0', 143: 'PTQ1_0'}

#: Kernel errors a read can return TRANSIENTLY on this machine's /mnt/c relay.
#: Measured 2026-09-14 (two separate runs of the self-test): ``f.read(8)`` inside the
#: metadata walk raised ``OSError: [Errno 12] Cannot allocate memory`` at
#: ``gguf_tensors.py:18`` and at ``gguf_kquant.py:355``.  The failing call is EIGHT
#: BYTES, and an 8-byte read on a BufferedReader is a fill of its existing 8 KiB
#: buffer -- so neither the requested size nor any ulimit is the variable.  The same
#: file parsed in another process in the same minute, and a byte-identical 12 MB head
#: on ext4 never failed: the failure is a transient property of the mount under guest
#: memory pressure.  A retry is therefore the right shape, not a smaller/larger read.
TRANSIENT_READ_ERRNOS = frozenset((errno.EINTR, errno.EAGAIN, errno.ENOMEM, errno.EBUSY))
READ_ATTEMPTS = 5


def transient_read(fh, n):
    """``fh.read(n)``, retrying a transient kernel error a bounded number of times.

    Not silent: the first retry says so on stderr, and when the attempts run out the
    original error propagates unchanged -- an absorbed error becomes a *logged* one,
    and an unabsorbable one never turns into a short read (a short read would
    desynchronise the metadata walk and yield garbage names with no exception at all).
    """
    for attempt in range(READ_ATTEMPTS):
        try:
            return fh.read(n)
        except OSError as exc:
            if exc.errno not in TRANSIENT_READ_ERRNOS or attempt + 1 >= READ_ATTEMPTS:
                raise
            if attempt == 0:
                sys.stderr.write(
                    "[gguf] transient %s on a %s B read; retrying up to %d times\n"
                    % (exc, n, READ_ATTEMPTS - 1))
            time.sleep(0.25 * (attempt + 1))


class _RetryingReader:
    """Read-only wrapper making :func:`transient_read` the only way to read.

    ``scan``/``_parse`` open one file and then do hundreds of thousands of tiny reads,
    so wrapping the handle is the smallest change that covers all of them -- including
    the ones added later.
    """

    def __init__(self, fh):
        self._fh = fh

    def read(self, n=-1):
        return transient_read(self._fh, n)

    def seek(self, *args):
        return self._fh.seek(*args)

    def tell(self):
        return self._fh.tell()

    def close(self):
        return self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()



def scan(path, opener=open):
    """Every tensor the file declares, as ``(name, type_name, dims)``.

    ``opener`` exists so a test can hand in a file object that injects the measured
    transient read error without touching this module (test_convert_runner.py).
    """
    f = _RetryingReader(opener(path, 'rb'))
    assert f.read(4) == b'GGUF'
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))

    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def skip_val(t):
        if t == 0:
            f.read(1)
        elif t == 1:
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t == 7:
            f.read(1)
        elif t == 8:
            read_str()
        elif t == 9:
            (atype, count) = struct.unpack('<IQ', f.read(12))
            for _ in range(count):
                skip_val(atype)
        elif t in (10, 11, 12):
            f.read(8)
        else:
            raise SystemExit('unknown kv type %d' % t)

    for _ in range(n_kv):
        read_str()
        (t,) = struct.unpack('<I', f.read(4))
        skip_val(t)
    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        # GGUF v3: dims 为 u64
        dims = struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims))
        (ttype,) = struct.unpack('<I', f.read(4))
        f.read(8)
        tensors.append((name, GGML_TYPES.get(ttype, 'T%d' % ttype), dims))
    f.close()
    return tensors


def main():
    for p in sys.argv[1:]:
        t = scan(p)
        hist = Counter(x[1] for x in t)
        print('== %s tensors=%d' % (p.split('/')[-1], len(t)))
        print('types:', dict(hist))
        for name, ty, dims in t[:16]:
            print('  %-58s %-6s %s' % (name, ty, dims))


if __name__ == '__main__':
    main()
