#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_runner.py — 模型导入后的全自动转换编排 (A 层: 受支持家族)。

输入: model_import.scan + verdict -> 输出可执行命令链:
  * hf bf16 + 支持家族        -> convert.py (groupwise-int) -> .ninfer
  * hf NVFP4 (unsloth 风格)   -> convert_nvfp4.py (需 bf16 源 + quant 源)
  * gguf F16/BF16/F32         -> gguf_extract.py (抽 bf16 safetensors) -> convert.py
  * ninfer                    -> 无需转换 (直接入模型库)
其余: 礼貌拒绝 (K-quant/家族不支持/显存不够)。

任务执行器: 后台线程跑命令链, 状态可查询。

状态机
------
``queued -> running -> converted -> done | failed``

``converted`` 只说明转换链 rc=0; **``done`` 只能由真实加载验证置位**（引擎
load 成功且至少产出 1 个 token）。这两件事不同：一个跑不起来的产物是**缺陷**，
不是成功。因此这里**不存在「没有任何可跑产物却报成功」的路径**：缺少验证器
同样是失败，不是放行。GUI 的发布步骤（``ninfer-gui.py`` 里的
``state == 'done'``）只会在验证通过后执行。
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import time

# Checkout-relative defaults, all overridable: the repository owns no machine path.
REPO = os.environ.get('NINFER_REPO') or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get('NINFER_PYTHON') or sys.executable
MODELS_DIR = os.environ.get('NINFER_MODELS_DIR') or os.path.join(REPO, 'models')
CONVERT_DIR = os.path.join(REPO, 'tools', 'convert', 'qwen3_8_27b')

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


class Job:
    """单个转换任务的运行状态。"""
    def __init__(self, job_id, steps):
        self.job_id = job_id
        self.steps = steps          # [(label, cmd_list), ...]
        self.state = 'queued'       # queued|running|converted|done|failed
        self.log = []
        self.step = 0
        self.done_steps = 0
        self.out_artifact = None
        self.error = None
        #: Absolute path the load check runs against.  For a real conversion it is
        #: <MODELS_DIR>/<out_name>; when the input is already a .ninfer there are no
        #: steps and the source itself is the artifact.
        self.artifact_path = None
        #: Set only by verify_artifact: True means the engine really loaded it.
        self.verified = False
        self.verification = None

    def as_dict(self):
        return {'job_id': self.job_id, 'state': self.state, 'step': self.step,
                'done_steps': self.done_steps, 'total': len(self.steps),
                'log': self.log[-80:], 'out_artifact': self.out_artifact,
                'error': self.error, 'artifact_path': self.artifact_path,
                'verified': self.verified, 'verification': self.verification}


def _gguf_reader_verdict(scan):
    """Can tools/convert/gguf_extract.py decode every tensor in this GGUF?

    Returns None when the scan recorded no ggml type ids (nothing to check, the
    extractor will make its own decision), else a refusal string or ''.

    The previous test was ``any(q in quant_name for q in ('Q', 'K'))``: it refused
    every K-quant GGUF because the reader could not read one.  Now that
    tools/convert/gguf_kquant decodes Q4_K and Q6_K, refusing them by name would be
    a stale "unsupported" flag -- the exact failure this project treats as worse than
    doing nothing.  The set of decodable types is asked of the reader itself, so this
    cannot disagree with it.
    """
    types = scan.get('gguf_types')
    if not types:
        return None
    try:
        sys.path.insert(0, REPO)
        from tools.convert import gguf_kquant as K          # noqa: PLC0415
    except Exception:                                       # noqa: BLE001
        return None
    unreadable = {int(t): c for t, c in types.items() if int(t) not in K.DEQUANTIZERS}
    if not unreadable:
        return ''
    detail = ', '.join('%s x%d' % (K.type_name(t), c) for t, c in sorted(unreadable.items()))
    return ('GGUF 里有本树读不了的量化类型 (%s)；可读的是 %s。'
            '补救：换 F16/BF16 源，或在 tools/convert/gguf_kquant.py 里补该类型的反量化表。'
            % (detail, K.SUPPORTED_NAMES))


#: llama.cpp's architecture string for the qwen3.5/3.6/3.8 text family.  Read from the
#: scan's ``arch_names[0]``, which is the GGUF's own ``general.architecture`` (measured on
#: Ornith-1.5-9B-Q4_K_M: ``arch_names == ['qwen35', 'Ornith-1.5-9B-Q4_K_M']``), not from
#: ``arch_note``, which is prose for humans.
QWEN3_5_GGUF_ARCH = 'qwen35'
QWEN3_5_OUT_NAME = 'qwen3_5_9b_auto.ninfer'


def _is_qwen3_5_gguf(scan):
    """Is this a GGUF the tree has a qwen3.5-family target and converter for?"""
    if scan.get('kind') != 'gguf':
        return False
    arch_names = scan.get('arch_names') or ()
    return bool(arch_names) and arch_names[0] == QWEN3_5_GGUF_ARCH


def _qwen3_5_resources():
    """The six frontend resources, found the way the front door looks for them.

    ``NINFER_RESOURCE_ROOTS`` is the same variable ``import_model.py:63`` reads for the
    same purpose, so there is one convention and not two.  The names are asked of
    ``tools/convert/qwen3_5_9b/inventory.py`` rather than restated here: a second copy
    of a six-name list is a fork waiting to disagree with the converter.

    Refuses by name when nothing fits.  This deliberately does NOT return a plausible
    default: a recipe that emits a command the converter will reject at run time is the
    "recognised but not doable" shape this project treats as worse than a refusal.
    """
    sys.path.insert(0, REPO) if REPO not in sys.path else None
    try:
        from tools.convert.qwen3_5_9b import inventory as q35            # noqa: PLC0415
    except Exception as exc:                                            # noqa: BLE001
        raise ValueError(
            'qwen3.5-9b 的转换器资源清单读不到（tools/convert/qwen3_5_9b/inventory.py）: '
            '%s: %s' % (type(exc).__name__, exc))
    names = [os.path.basename(n) for n in q35.RESOURCE_NAMES]
    roots = [r for r in os.environ.get('NINFER_RESOURCE_ROOTS', '').split(os.pathsep) if r]
    tried = []
    for root in roots:
        missing = [n for n in names if not os.path.isfile(os.path.join(root, n))]
        if not missing:
            return root
        tried.append('%s (缺 %s)' % (root, ', '.join(missing)))
    raise ValueError(
        'qwen3.5-9b 的转换器需要六个前端资源 (%s)；%s。'
        '补救：把 NINFER_RESOURCE_ROOTS 指向含这六个文件的目录（可多个，用 %r 分隔）。'
        % (', '.join(names),
           '；'.join(tried) if tried else 'NINFER_RESOURCE_ROOTS 未设置',
           os.pathsep))


def plan_conversion(scan, model_id_out=None):
    """根据 scan 生成步骤链; 返回 (steps, out_name) 或抛 ValueError(不可转)。"""
    kind = scan.get('kind')
    fam = scan.get('model_id') or model_id_out
    src = scan.get('path')
    if kind == 'ninfer':
        return [], os.path.basename(src)
    if kind not in ('hf', 'gguf'):
        raise ValueError('该输入无法自动转换 (%s)' % kind)
    # The qwen3.5 family gets its own single step, and it is not the generic chain:
    # tools/convert/qwen3_5_9b/convert.py reads the GGUF directly and streams row blocks,
    # because the generic gguf_extract -> convert chain materialises a ~18 GB bf16
    # intermediate and never finished on this machine (measured: 12-14 GB peak on a
    # 22 GB box -- p29's converter docstring and n7land REPORT.md section 5.3).
    if _is_qwen3_5_gguf(scan):
        verdict = _gguf_reader_verdict(scan)
        if verdict:
            raise ValueError(verdict)
        resources = _qwen3_5_resources()
        return ([('convert_qwen3_5_9b',
                  [PY, os.path.join(REPO, 'tools', 'convert', 'qwen3_5_9b', 'convert.py'),
                   '--gguf', src, '--resources', resources,
                   '--out', os.path.join(MODELS_DIR, QWEN3_5_OUT_NAME)])],
                QWEN3_5_OUT_NAME)
    if fam not in ('qwen3.8-27b', 'qwen3.6-27b'):
        if kind == 'gguf':
            raise ValueError(
                '这个 GGUF 的架构 (%s) 不在本软件已注册的转换家族里；'
                '已注册的是 qwen3.8-27b / qwen3.6-27b。'
                '注意：认出格式 != 引擎能跑，引擎侧还缺该架构的 target。'
                % (scan.get('arch_note') or fam or '未声明'))
        raise ValueError('家族 %s 尚无自动转换 recipe (ARCHKIT B 层)' % fam)
    out_name = 'qwen3_8_27b_auto.ninfer'
    steps = []
    if kind == 'gguf':
        verdict = _gguf_reader_verdict(scan)
        if verdict:
            raise ValueError(verdict)
        # GGUF -> bf16 safetensors (F32/F16/BF16 直读 + Q4_K/Q6_K 反量化)
        extract_dir = os.path.join(os.path.dirname(src), '_ninfer_bf16')
        steps.append(('gguf_extract', [PY, os.path.join(REPO, 'tools', 'convert',
                                                        'gguf_extract.py'),
                                       '--src', src, '--out', extract_dir]))
        src = extract_dir
    steps.append(('convert_groupwise',
                  [PY, os.path.join(CONVERT_DIR, 'convert.py'),
                   '--model', src, '--out', os.path.join(MODELS_DIR, out_name),
                   '--device', 'cuda']))
    return steps, out_name


class Runner:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._seq = 0

    def start(self, scan):
        with self._lock:
            self._seq += 1
            job = Job('imp%03d' % self._seq, [])
        try:
            steps, out_name = plan_conversion(scan)
        except ValueError as e:
            job.state = 'failed'
            job.error = str(e)
            with self._lock:
                self._jobs[job.job_id] = job
            return job
        job.steps = steps
        job.out_artifact = out_name
        job.artifact_path = (os.path.join(MODELS_DIR, out_name) if steps
                             else str(scan.get('path') or ''))
        job.state = 'queued'
        with self._lock:
            self._jobs[job.job_id] = job
        t = threading.Thread(target=self._run, args=(job,), daemon=True)
        t.start()
        return job

    def _run(self, job):
        job.state = 'running'
        for i, (label, cmd) in enumerate(job.steps):
            job.step = i
            job.log.append('[%s] %s' % (time.strftime('%H:%M:%S'), label))
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                   encoding='utf-8', errors='replace')
                job.log.append((r.stdout or '')[-2000:])
                if r.returncode != 0:
                    job.state = 'failed'
                    job.error = '%s rc=%d: %s' % (label, r.returncode,
                                                  (r.stderr or '')[-300:])
                    return
            except Exception as e:
                job.state = 'failed'
                job.error = '%s exc: %s' % (label, e)
                return
            job.done_steps = i + 1
        # Every step exited 0, which means the conversion *ran*.  It does not mean
        # the artifact works, and only the second claim is worth reporting.
        job.state = 'converted'
        job.log.append('[%s] 转换完成, 开始加载验证: %s'
                       % (time.strftime('%H:%M:%S'), job.artifact_path))
        try:
            job.verification = verify_artifact(job.artifact_path)
        except ArtifactUnverified as exc:
            job.state = 'failed'
            job.error = ('产物加载验证未通过 (转换 rc=0, 但产物跑不了): %s' % exc)
            job.log.append('[%s] 验证失败: %s' % (time.strftime('%H:%M:%S'), job.error))
            return
        job.verified = True
        job.state = 'done'
        job.log.append('[%s] 验证通过: %s' % (time.strftime('%H:%M:%S'), job.verification))
        if job.out_artifact:
            job.log.append('产物: %s' % os.path.join(MODELS_DIR, job.out_artifact))

    def status(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
        return job.as_dict() if job else None


runner = Runner()
