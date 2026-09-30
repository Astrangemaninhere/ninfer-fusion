#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""artifact_verify.py — 产物加载验证门 (the engine-load gate), moved here from tools/gui.

WHY IT IS HERE (P1-b, dl/flowzoo/DISPATCH.md), measured rather than argued:
  * ``verify_artifact`` is the ONLY judgement in this tree that says "this artifact really loads
    and emits a token" -- the conversion chains' single acceptance gate.  It was reachable only
    as ``tools.gui.convert_runner``, and ``tools/convert/import_model.py`` imported it by that
    name, so a rename of a GUI module decided whether the CLI's gate existed at all:
    measured 2026-09-29, that import was a ModuleNotFoundError (rc=1) and the gate was dead on
    the CLI side while the GUI side kept working.
  * the implementation is now here and ONLY here.  ``tools/gui/convert_runner.py`` re-exports
    these names, so the GUI chain's own call site does not change, and neither chain can reach a
    different verdict about the same file.

WHAT THIS MODULE OWNS: the engine-independent container-header check
(:func:`artifact_header_error`), the engine half (:func:`verify_artifact`), the failure type
(:class:`ArtifactUnverified`), and the constants those three read (``ENGINE`` et al).  Note that
``ENGINE`` is read from THIS module's globals at call time, so a caller that wants to point the
engine somewhere else must set ``artifact_verify.ENGINE``.

Moved verbatim: no signature, docstring or byte of the implementation changed in the move.
"""
from __future__ import annotations

import os
import struct
import subprocess

#: The repository root, resolved from this file rather than from the caller's cwd -- the same
#: three-parents rule tools/gui/convert_runner.py used (tools/convert/x.py and tools/gui/x.py are
#: both three levels down), so ``ENGINE``'s default resolves to the same path as before.
REPO = os.environ.get('NINFER_REPO') or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 加载验证用的引擎。默认取仓库自己的构建产物; NINFER_ENGINE 指向其它二进制
#: (打包后的版本)。不存在时不会静默放行 —— 见 verify_artifact。
ENGINE = os.environ.get('NINFER_ENGINE') or os.path.join(REPO, 'build', 'apps', 'ninfer')
VERIFY_TIMEOUT = int(os.environ.get('NINFER_VERIFY_TIMEOUT') or 1800)
VERIFY_PROMPT = os.environ.get('NINFER_VERIFY_PROMPT') or 'Hello'
#: b'NINFER\x00\x02' — src/artifact/reader.cpp / tools/artifact/container.py
ARTIFACT_MAGIC = b'NINFER' + bytes([0, 2])
#: The v2 prefix is ``<8sQ`` magic + directory length (tools/artifact/container.py:16),
#: and the payload starts at the next 4096-byte boundary (:18/:19).
ARTIFACT_PREFIX = struct.Struct('<8sQ')
PAYLOAD_ALIGNMENT = 4096
#: The engine's one token-id line: a 12-wide ``tokens`` label, a 26-wide ``generated
#: ids`` label, then the ids (apps/cli/main.cpp:523).
TOKEN_ID_LABEL = 'tokens'
TOKEN_ID_LABEL_2 = 'generated ids'


class ArtifactUnverified(Exception):
    """The conversion finished but the artifact could not be shown to run."""


def artifact_header_error(path):
    """Why this file cannot be a v2 artifact at all, or '' when the header is sound.

    Engine-INDEPENDENT, and that is the point: parse what the container's own prefix
    declares (tools/artifact/container.py:16-19) and compare it with the file's
    length.  Before this existed the only check between a conversion and ``done`` was
    an 8-byte magic comparison, so *every* malformed artifact was handed to the
    engine and the answer took a full load attempt (measured: a 4096-byte file whose
    first 8 bytes are the magic spent 8 s and a 2 GiB truncation of a 21.5 GB
    artifact spent 30 s before the engine refused them).  A malformed directory
    cannot load, so refusing it here removes an engine run that could only ever
    fail -- and, unlike the magic test, it cannot be satisfied by an engine that
    answers anyway.

    NOT covered, deliberately named: this reads the prefix, not the directory.  A
    file whose directory is intact but whose *payload* is short (the 2 GiB
    truncation above) passes here  -- ``json_bytes`` bounds the directory, and the
    prefix carries no total-length field, so catching that needs the directory
    walked (tools/artifact/container.py does it: "object … extends beyond the
    file").  Wiring that reader in is a packaging decision, not a gate detail: the
    frozen GUI ships tools/gui and tools/archkit as datas and ``convert_runner``
    from tools/gui (tools/package/ninfer-gui.spec), and pulling tools/artifact into
    that bundle is unverified today -- see REPORT.md, the named blocker.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return 'cannot stat it: %s' % exc
    if size < ARTIFACT_PREFIX.size:
        return ('it is %d B, shorter than the %d-byte v2 prefix'
                % (size, ARTIFACT_PREFIX.size))
    with open(path, 'rb') as fh:
        magic, json_bytes = ARTIFACT_PREFIX.unpack(fh.read(ARTIFACT_PREFIX.size))
    if magic != ARTIFACT_MAGIC:
        return 'the first 8 bytes are %r, not the v2 magic' % (magic,)
    if json_bytes == 0:
        return 'the directory length field is 0, so there is no object directory'
    end = ARTIFACT_PREFIX.size + json_bytes
    start = -(-end // PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT
    if end > size or start > size:
        return ('the directory claims %d B from offset %d, ending at %d, past the '
                'file\'s %d B' % (json_bytes, ARTIFACT_PREFIX.size, end, size))
    return ''


def count_generated_tokens(stderr_text):
    """Number of token ids the engine reported, or None if it reported nothing.

    ``--print-token-ids`` writes one stderr line: the labels ``tokens`` /
    ``generated ids`` followed by the ids (apps/cli/main.cpp:523).  A run that
    generated nothing prints the labels and no numbers, which is the failure this
    catches.

    Measured line, real engine, real 21.5 GB artifact, 2026-09-14:
    ``tokens      generated ids             760`` -- one id, so this returns 1.
    The labels are words, so they are not counted as ids, and the summary lines
    ("generated tokens", "prompt tokens") are indented AND do not start with
    ``tokens`` as their first field, so the id line is the only line this matches.
    """
    for line in (stderr_text or '').splitlines():
        if line.strip().startswith(TOKEN_ID_LABEL):
            return sum(1 for field in line.split() if field.lstrip('-').isdigit())
    return None


def verify_artifact(path, prompt=None):
    """Prove the artifact loads and emits >= 1 token, else raise.

    Returns a one-line evidence string.  Every failure mode raises, so there is no
    branch that reports success without a real load.  A *missing* verifier is a
    failure rather than a pass: "we could not check" must never read as "it works".

    Two stages, deliberately: the container header is checked here (engine-free), and
    the load-and-emit half is delegated to the engine binary named by ``ENGINE``.
    The trust boundary is that binary: ``NINFER_ENGINE`` decides the second half, so
    a replacement that prints a plausible id line... passes.  That is a property of
    the design, not an oversight -- the engine is the only thing in this repository
    that can load an artifact -- and it is pinned in test_convert_runner.py so that
    nobody reads this function as a content-addressed proof of runnability.
    """
    if not path or not os.path.isfile(path):
        raise ArtifactUnverified('产物文件不存在: %s' % path)
    size = os.path.getsize(path)
    bad = artifact_header_error(path)
    if bad:
        raise ArtifactUnverified('产物不是 .ninfer 容器: %s (%d B)' % (bad, size))
    if not os.path.isfile(ENGINE) or not os.access(ENGINE, os.X_OK):
        raise ArtifactUnverified('找不到可执行的引擎 (%s), 因此无法证明产物能加载; '
                                 '设 NINFER_ENGINE 指向 ninfer 二进制' % ENGINE)
    cmd = [ENGINE, path, '--prompt', prompt or VERIFY_PROMPT, '--max-new', '1',
           '--greedy', '--print-token-ids']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT,
                           encoding='utf-8', errors='replace')
    except OSError as exc:
        raise ArtifactUnverified('引擎无法启动: %s: %s' % (type(exc).__name__, exc))
    except subprocess.TimeoutExpired:
        raise ArtifactUnverified('引擎在 %d s 内没有跑完一次 1-token 生成'
                                 % VERIFY_TIMEOUT)
    if r.returncode != 0:
        tail = ' | '.join((r.stderr or '').strip().splitlines()[-4:])
        raise ArtifactUnverified('引擎拒绝加载/运行产物 (rc=%d): %s'
                                 % (r.returncode, tail))
    n = count_generated_tokens(r.stderr)
    if not n:
        raise ArtifactUnverified(
            '引擎退出码 0 但一个 token 都没有产出 (%s); 产物无法运行'
            % ('未报告 token 计数' if n is None else '计数为 0'))
    return '引擎加载成功并产出 %d 个 token (%s)' % (n, os.path.basename(ENGINE))
