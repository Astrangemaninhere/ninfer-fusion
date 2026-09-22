#!/usr/bin/env python3
"""
k40land opcheck v2 —— 「比较运算符改写」的静态闸门。两个相位：

  相位 1（漏比/多比）  对清单里每个 (文件, 结构体)：
      「显式 operator== 里被比较的成员集合」 == 「该结构体的非静态数据成员集合」
      不等就红，点名 文件:行，并列出少比/多比的具体字段名。
      `= default` 形态直接放行（编译器自己会跟上）。

  相位 2（声明形态，v2 新增）  对 --files 列出的每个文件：
      S1 重复的返回类型说明符：某个 `operator==` 的**声明头**里同一个返回类型关键字出现
         >= 2 次（`long long` 除外）。实测样本：`[[nodiscard]] friend constexpr bool`
         换行之后又被加了 `bool operator==` ⇒ 头里 `bool` 两次。
      S2 friend（非成员）`operator==` 的参数个数 != 2：二元运算符的非成员形态必须**恰好
         两个**参数（[over.binary]）。样本里同一个声明同时坏了这一条。
      L1 大括号平衡（去注释、保留字面量后逐字节数）。
      （v1 有一条 L1 圆括号平衡相位，v2 **删掉**：它把跨行的注释/字符串碎片当成括号不平衡，
        实测在 45 个文件里对 src/core/arch_caps.h（=612/598）与 src/spec/recall_identity.h
        （=190/192）报假红，而把字面量涂白后是 423/423 与 188/188，全平衡。见 report 第 2 节。）

为什么相位 2 存在：k40lam20 的 REPORT 第 9.8 条按名写了「opcheck 只查比较项 vs 成员表，
不查语法；context_cost.h 的重复 bool 是人工逐字对照 pre/post 才发现的，这条检查我没有
自动化的写法」。v2 就是那条自动化。

用法：
    python3 opcheck.py [ROOT] [MANIFEST] [FILES]
      ROOT     树根（缺省 $K40_ROOT 或 /home/user/ninfer-fusion）
      MANIFEST 每行 `relpath<TAB>StructName`（缺省与本脚本同目录的 opcheck.manifest）
      FILES     相位 2 的文件清单（每行一个 relpath；缺省与本脚本同目录的 opcheck.files）
    python3 opcheck.py --selftest
      **常驻非空转反控**：拿脚本内嵌的两个夹具跑一遍，要求相位 1 与相位 2 各报 1 条红、
      并且点到正确的字段/检查名。若闸门变成恒绿的空转，这一条会红。
退出码：0 = 全绿；1 = 有红；2 = 用法/IO 错。--selftest 期望「有红」，故它自己返回 0 才算自检通过。
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
FLAGS = set(a for a in sys.argv[1:] if a.startswith('--'))


# ---------------------------------------------------------------- 词法小工具
def strip_comments(src):
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            j = src.find('*/', i + 2)
            j = n if j < 0 else j
            out.append(''.join(ch if ch == '\n' else ' ' for ch in src[i:j + 2])); i = j + 2
        elif c in '"\'':
            q = c; j = i + 1
            while j < n:
                if src[j] == '\\': j += 2; continue
                if src[j] == q: j += 1; break
                if src[j] == '\n': break
                j += 1
            out.append(''.join(ch if ch == '\n' else ' ' for ch in src[i:j])); i = j
        else:
            out.append(c); i += 1
    return ''.join(out)


def strip_comments_keep_literals(src):
    """只去注释，字符串/字符字面量**整段原样保留**（L1 用这一版）。

    不能直接用 strip_comments：它把字面量涂白，于是测试代码里的 `'}'` 字符字面量消失，
    `}` 比 `{` 少 3 个（实测 tests/test_resource_manager.cpp：raw 569/569 → 569/566）。
    也不能「不特殊处理字面量」：src/core/arch_caps.h 里有含 `//` 的字符串字面量，
    那些 `//` 会被当行注释把该行涂白。所以：注释涂白、字面量原样拷、字面量内部不认注释。
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            j = src.find('*/', i + 2)
            j = n if j < 0 else j
            out.append(''.join(ch if ch == '\n' else ' ' for ch in src[i:j + 2])); i = j + 2
        elif c in '"\'':
            q = c; j = i + 1
            while j < n:
                if src[j] == '\\':
                    j += 2; continue
                if src[j] == q:
                    j += 1; break
                if src[j] == '\n':
                    break
                j += 1
            out.append(src[i:j]); i = j
        else:
            out.append(c); i += 1
    return ''.join(out)


def lineno(src, off):
    return src.count('\n', 0, off) + 1


def match_brace(s, o):
    d = 0
    for i in range(o, len(s)):
        if s[i] == '{': d += 1
        elif s[i] == '}':
            d -= 1
            if d == 0: return i
    return -1


def split_top(s, sep=';'):
    out, buf, d = [], [], 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '{':
            e = match_brace(s, i)
            if e < 0:
                buf.append(s[i:]); break
            buf.append(s[i:e + 1]); i = e + 1; continue
        if c in '<([':
            d += 1
        elif c in '>)]':
            d -= 1
        elif c == sep and d <= 0:
            out.append(''.join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    out.append(''.join(buf))
    return out


FUNC_LIKE = re.compile(
    r'^[^=;{]*\b([A-Za-z_]\w*|operator\s*[^\s(]+)\s*\([^()]*(?:\([^()]*\)[^()]*)*\)'
    r'\s*(?:const\b|noexcept\b|override\b|final\b|->[^;{]*|=\s*(?:0|default|delete)\b|\[\[[^\]]*\]\]|\s)*[;{]?\s*$')

SKIP_PREFIX = ('public', 'private', 'protected', 'friend', 'using', 'typedef', 'template',
               'enum', 'virtual', 'explicit', 'operator', '#', '~', 'class', 'struct',
               'union', 'static_assert', 'return')


def members_of(body):
    """返回 {成员名: 偏移} —— 只取非静态数据成员。"""
    n = len(body)
    ign = bytearray(n)
    i = 0
    while i < n:
        if body[i] == '{':
            e = match_brace(body, i)
            if e < 0:
                break
            lead_start = 0
            for sep in (';', '}', '{'):
                k = body.rfind(sep, 0, i)
                if k + 1 > lead_start:
                    lead_start = k + 1
            lead = body[lead_start:i]
            if ')' in lead:
                for k in range(lead_start, e + 1):
                    ign[k] = 1
            i = e + 1
            continue
        i += 1
    for m in re.finditer(r'(?m)^[ \t]*#.*$', body):
        for k in range(m.start(), m.end()):
            ign[k] = 1
    res = {}
    buf_start = None
    depth = 0
    i = 0
    while i < n:
        c = body[i]
        if ign[i]:
            i += 1
            continue
        if buf_start is None:
            buf_start = i
        if c in '{[(':
            depth += 1
        elif c in '}])':
            depth -= 1
        elif c == ';' and depth <= 0:
            handle_member(body[buf_start:i], buf_start, res)
            buf_start = None
        i += 1
    return res


def _member_name(p):
    p = re.sub(r'\[\[[^\]]*\]\]', ' ', p).strip()
    if not p:
        return None
    if re.match(r'^(struct|class|union|enum)\b', p):
        k = p.rfind('}')
        if k >= 0:
            p = p[k + 1:]
        else:
            return None
        toks = re.findall(r'[A-Za-z_]\w*', p)
        return toks[-1] if toks else None
    p2 = p.rstrip()
    if p2.endswith(')'):
        d = 0
        k = -1
        for j in range(len(p2) - 1, -1, -1):
            if p2[j] == ')':
                d += 1
            elif p2[j] == '(':
                d -= 1
                if d == 0:
                    k = j
                    break
        if k > 0:
            head = p2[:k]
            d3 = 0
            has_eq = False
            for i, c in enumerate(head):
                if c in '<([':
                    d3 += 1
                elif c in '>)]':
                    d3 -= 1
                elif d3 <= 0 and c == '=':
                    has_eq = True
                    break
            if not has_eq:
                p = head
    cut = len(p)
    depth = 0
    for i, c in enumerate(p):
        if c in '<([':
            depth += 1
        elif c in '>)]':
            depth -= 1
        elif depth <= 0 and c in '={':
            cut = i
            break
        elif depth <= 0 and c == ':':
            if not (i + 1 < len(p) and p[i + 1] == ':') and not (i > 0 and p[i - 1] == ':'):
                cut = i
                break
    pre = p[:cut].strip()
    pre = re.sub(r'\[[^\]]*\]\s*$', '', pre).strip()
    pre = re.sub(r'(const|volatile|mutable)\s*$', '', pre).strip()
    if not pre:
        return None
    toks = re.findall(r'[A-Za-z_]\w*', pre)
    if not toks:
        return None
    return toks[-1]


BUILTIN = {'void', 'bool', 'char', 'short', 'int', 'long', 'float', 'double', 'unsigned',
           'signed', 'auto', 'size_t', 'decltype', 'sizeof', 'struct', 'class', 'union',
           'enum', 'const', 'volatile', 'static', 'constexpr', 'inline', 'mutable'}


def looks_like_funcdecl(t):
    i, depth = 0, 0
    while i < len(t):
        c = t[i]
        if c in '<[':
            depth += 1
        elif c in '>]':
            depth -= 1
        elif c == '(' and depth <= 0:
            break
        i += 1
    else:
        return False
    d, j = 0, i
    while j < len(t):
        if t[j] == '(':
            d += 1
        elif t[j] == ')':
            d -= 1
            if d == 0:
                break
        j += 1
    else:
        return False
    head, rest = t[:i], t[j + 1:]
    d2 = 0
    for c in head:
        if c in '<([':
            d2 += 1
        elif c in '>)]':
            d2 -= 1
        elif d2 <= 0 and c in '={':
            return False
    m = re.search(r'([A-Za-z_]\w*)\s*$', head)
    if not m:
        return False
    if m.group(1) in BUILTIN:
        return False
    return bool(re.match(r'^\s*(const\b|volatile\b|noexcept\b|override\b|final\b|->|'
                         r'=\s*(0|delete|default)\b|\[\[[^\]]*\]\]|\s)*$', rest))


def handle_member(chunk, off, res):
    t = strip_comments(chunk).strip()
    while re.match(r'^(public|private|protected)\s*:\s*', t):
        t = re.sub(r'^(public|private|protected)\s*:\s*', '', t, count=1)
    if not t:
        return
    first = re.match(r'([A-Za-z_]\w*)', t)
    if first and first.group(1) in SKIP_PREFIX:
        return
    if t.startswith('static_assert') or t.startswith('return'):
        return
    if re.match(r'^static\b', t):
        return
    if looks_like_funcdecl(t):
        return
    for part in split_top(t, ','):
        nm = _member_name(part)
        if nm:
            res[nm] = off


def find_structs(src):
    out = []
    for m in re.finditer(r'\b(struct|class)\s+([A-Za-z_]\w*)[^;{]*\{', src):
        o = src.index('{', m.end() - 1)
        c = match_brace(src, o)
        if c < 0:
            continue
        out.append((m.group(2), o, c))
    return out


EQ = re.compile(r'operator\s*==\s*\(')


def explicit_eqs(src, o, c):
    out = []
    body = src[o + 1:c]
    for m in EQ.finditer(body):
        po = o + 1 + body.index('(', m.end() - 1)
        d = 0
        pe = -1
        for i in range(po, c):
            if src[i] == '(': d += 1
            elif src[i] == ')':
                d -= 1
                if d == 0: pe = i; break
        if pe < 0:
            continue
        params = src[po + 1:pe]
        j = pe + 1
        while j < c and src[j] in ' \t\n':
            j += 1
        while j < c and src[j] != '{':
            if src[j] == ';':
                break
            j += 1
        if j >= c or src[j] != '{':
            continue
        bo = j
        bc = match_brace(src, bo)
        if bc < 0 or bc > c:
            continue
        out.append((params, bo, bc))
    return out


def param_refs(eq_body, params):
    names = re.findall(r'\b([A-Za-z_]\w*)\s*(?:\.|->)\s*([A-Za-z_]\w*)', eq_body)
    pnames = set()
    for p in params.split(','):
        p = p.strip()
        p = re.sub(r'^\[\[[^\]]*\]\]\s*', '', p)
        p = re.sub(r'\b(const|constexpr|volatile|noexcept|auto)\b', ' ', p)
        p = re.sub(r'[&*]', ' ', p)
        p = re.sub(r'<[^<>]*>', ' ', p)
        toks = re.findall(r'[A-Za-z_]\w*', p)
        if toks:
            pnames.add(toks[-1])
    got = set()
    for base, mem in names:
        if base in pnames:
            got.add(mem)
    return got, pnames


# ---------------------------------------------------------------- 相位 2
TYPEKW = ('bool', 'int', 'void', 'char', 'short', 'float', 'double', 'unsigned', 'signed',
          'size_t')


def boundaries(src):
    """每层各记「上一个声明边界」：本层的 `;` 之后，或 `{`/`}` 之后。见 report 第 2 节。"""
    n = len(src)
    b = [0] * (n + 1)
    lv_cur = {0: 0}
    lv = 0
    for i, c in enumerate(src):
        b[i] = lv_cur[lv]
        if c == '{':
            lv += 1
            lv_cur[lv] = i + 1
        elif c == '}':
            if lv > 0:
                lv -= 1
            lv_cur[lv] = i + 1
        elif c == ';':
            lv_cur[lv] = i + 1
    b[n] = lv_cur[lv]
    return b


def params_of(src, po):
    d = 0
    i = po
    while i < len(src):
        if src[i] == '(':
            d += 1
        elif src[i] == ')':
            d -= 1
            if d == 0:
                break
        i += 1
    inner = src[po + 1:i]
    parts = [p.strip() for p in split_top(inner, ',')]
    parts = [p for p in parts if p and p != 'void']
    return inner, len(parts)


def phase2(root, files, out):
    red = 0
    for rel in files:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            print('RED %s: 文件不存在（相位 2 清单要求检查）' % rel); red += 1; continue
        raw = open(p, newline='', encoding='utf-8', errors='replace').read()
        for rel2, ln, chk, det in phase2_hits_with_lines(raw, rel):
            print('RED %s:%d: %s %s' % (rel2, ln, chk, det))
            red += 1
    return red


def phase2_hits_with_lines(raw, rel):
    out = []
    src = strip_comments(raw)
    cb = strip_comments_keep_literals(raw)
    for a, b, nm in (('{', '}', 'brace'),):
        if cb.count(a) != cb.count(b):
            out.append((rel, 0, 'L1-%s-UNBAL' % nm,
                        '%s=%d %s=%d (comments blanked, literals kept)'
                        % (a, cb.count(a), b, cb.count(b))))
    bnd = boundaries(src)
    for m in EQ.finditer(src):
        at = m.start()
        head = src[bnd[at]:at]
        ln = lineno(src, at)
        toks = re.findall(r'\b[A-Za-z_]\w*\b', head)
        for kw in TYPEKW:
            if toks.count(kw) >= 2:
                out.append((rel, ln, 'S1-DOUBLED-SPEC',
                            'declaration head has %s x%d: ...%s'
                            % (kw, toks.count(kw), ' '.join(head.split())[-70:])))
                break
        if re.search(r'\bfriend\b', head):
            po = src.index('(', m.end() - 1)
            inner, np = params_of(src, po)
            if np != 2:
                out.append((rel, ln, 'S2-FRIEND-ARITY',
                            'friend operator== with %d parameter(s): ...%s'
                            % (np, ' '.join(inner.split())[:70])))
    return out


# ---------------------------------------------------------------- 主流程
def phase1(root, man, out):
    red, n = 0, 0
    for line in open(man, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        rel, sname = (line.split('\t') + [''])[:2]
        rel = rel.strip(); sname = sname.strip()
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print('RED %s: 文件不存在（清单要求检查 %s）' % (rel, sname))
            red += 1
            continue
        src = strip_comments(open(path, newline='', encoding='utf-8', errors='replace').read())
        found = [(nm, o, c) for nm, o, c in find_structs(src) if nm == sname]
        if not found:
            print('RED %s: 找不到 struct/class %s' % (rel, sname))
            red += 1
            continue
        for nm, o, c in found:
            n += 1
            body = src[o + 1:c]
            mems = members_of(body)
            eqs = [e for e in explicit_eqs(src, o, c)
                   if re.search(r'\b' + re.escape(sname) + r'\b', e[0])]
            if not eqs:
                continue
            for params, bo, bc in eqs:
                got, pn = param_refs(src[bo + 1:bc], params)
                missing = sorted(set(mems) - got)
                extra = sorted(got - set(mems))
                if missing or extra:
                    red += 1
                    print('RED %s:%d: struct %s 的显式 operator== 与成员表不一致 '
                          '(成员 %d，比较 %d)' % (rel, lineno(src, bo), sname,
                                                len(mems), len(got)))
                    if missing:
                        print('      少比: %s' % ' '.join(missing))
                    if extra:
                        print('      多比: %s' % ' '.join(extra))
    print('opcheck 相位 1: 检查 %d 个结构体，红 %d 条' % (n, red))
    return red


# ---------------------------------------------------------------- 常驻反控
FIXTURE_BAD_MEMBER = """\
struct K40GateSelfTest {
    int alpha = 0;
    int beta  = 0;
    [[nodiscard]] friend constexpr bool operator==(const K40GateSelfTest& a,
                                                   const K40GateSelfTest& b) noexcept {
        return a.alpha == b.alpha;
    }
};
"""

FIXTURE_BAD_SYNTAX = """\
struct K40GateSelfTest2 {
    int alpha = 0;

    [[nodiscard]] friend constexpr bool
    bool operator==(const K40GateSelfTest2& o) noexcept {
        return alpha == o.alpha;
    }
};
"""


def selftest():
    """常驻反控：两个夹具必须各被判红，且点对字段/检查名。"""
    bad = 0
    # 夹具 1：少比一个成员 —— 相位 1 必须红并点名 beta
    src = strip_comments(FIXTURE_BAD_MEMBER)
    name, o, c = find_structs(src)[0]
    mems = members_of(src[o + 1:c])
    eqs = [e for e in explicit_eqs(src, o, c)
           if re.search(r'\b' + re.escape(name) + r'\b', e[0])]
    ok = False
    for params, bo, bc in eqs:
        got, pn = param_refs(src[bo + 1:bc], params)
        missing = sorted(set(mems) - got)
        if missing == ['beta']:
            ok = True
    print('selftest 相位 1 夹具: 期望 少比=[beta] -> %s' % ('红, 点名正确' if ok else '失败'))
    bad += 0 if ok else 1
    # 夹具 2：重复类型说明符 —— 相位 2 必须红并点 S1 与 S2
    hits = phase2_hits_with_lines(FIXTURE_BAD_SYNTAX, '<fixture>')
    chks = sorted(set(h[2] for h in hits))
    ok2 = ('S1-DOUBLED-SPEC' in chks) and ('S2-FRIEND-ARITY' in chks)
    print('selftest 相位 2 夹具: 期望 {S1-DOUBLED-SPEC, S2-FRIEND-ARITY} -> 实得 %s %s'
          % (chks, '红, 点名正确' if ok2 else '失败'))
    bad += 0 if ok2 else 1
    print('selftest: %s' % ('PASS（两个夹具都被判红）' if bad == 0 else 'FAIL（%d 条夹具没红）' % bad))
    return 0 if bad == 0 else 1


def main():
    if '--selftest' in FLAGS:
        return selftest()
    root = ARGS[0] if len(ARGS) > 0 else os.environ.get('K40_ROOT', '/home/user/ninfer-fusion')
    man = ARGS[1] if len(ARGS) > 1 else os.path.join(HERE, 'opcheck.manifest')
    fl = ARGS[2] if len(ARGS) > 2 else os.path.join(HERE, 'opcheck.files')
    if not os.path.exists(man):
        print('opcheck: manifest 不存在: %s' % man)
        return 2
    # 常驻非空转反控：每次运行都先自检两个夹具，自检不过直接红。见 SELFTEST 一节。
    st = selftest()
    red = phase1(root, man, None) + (1 if st else 0)
    if os.path.exists(fl):
        files = [l.strip() for l in open(fl, encoding='utf-8')
                 if l.strip() and not l.startswith('#')]
        red2 = phase2(root, files, None)
        print('opcheck 相位 2: 检查 %d 个文件，红 %d 条' % (len(files), red2))
        red += red2
    else:
        print('opcheck 相位 2: 跳过（没有文件清单 %s）' % fl)
    print('opcheck: 红 %d 条' % red)
    return 1 if red else 0


if __name__ == '__main__':
    sys.exit(main())
