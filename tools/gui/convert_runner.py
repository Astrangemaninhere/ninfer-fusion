#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_runner.py — 模型导入后的全自动转换编排 (A 层: 受支持家族)。

输入: model_import.scan + verdict -> 输出可执行命令链。

ROUTING LIVES IN ONE PLACE NOW
------------------------------
Which converter adapts which source is declared in ``tools/convert/source_registry.py``
(``ADAPTATIONS``), not here.  This module asks it (``registry.route``), builds the command
chain from the row (``registry.build_steps``) and refuses whatever the row refuses.  The
branch chain that used to live in ``plan_conversion`` is deleted, not rewritten.

That deletion is the fix for a defect F1065 found and named: the old ``_is_qwen3_5_gguf``
tested ``arch_names[0] == 'qwen35'`` ALONE and was tested BEFORE the family test, so every
64-layer ``qwen35`` GGUF was handed to the 32-layer package while the correct 27B branch
below it was dead code for those files.  The rule was never missing --
``tools/convert/import_model.py:791-804`` already guards an arch-owned converter with the
file's own ``(block_count, nextn)`` -- so the fix is that rule, reused as data (the row's
``geometry``) by both front doors.

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

import importlib
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

#: The registry is imported as ``tools.convert.source_registry`` even when this file is run
#: by path, so the repository root goes on sys.path first -- the same reason and the same
#: shape as ``import_model.py:41-50``.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.convert import source_registry as registry  # noqa: E402

#: THE LOAD GATE NOW LIVES IN tools/convert/artifact_verify.py (P1-b).  This module is a
#: RE-EXPORTER for it: the GUI chain's own call site -- ``job.verification = verify_artifact(...)``
#: in Runner._run below -- does not change, and the CLI chain no longer has to import ``tools.gui``
#: to reach a gate that was never the GUI's.
#: The WHOLE moved surface is re-exported, not only the two names the chains call, so a name that
#: was importable from here stays importable.  One trap this cannot paper over, stated rather than
#: hidden: ``ENGINE`` is now read from artifact_verify's globals at call time, so assigning
#: ``convert_runner.ENGINE = ...`` no longer redirects the engine -- set
#: ``tools.convert.artifact_verify.ENGINE`` instead (the self-test does exactly that).
from tools.convert.artifact_verify import (        # noqa: E402,F401
    ARTIFACT_MAGIC, ARTIFACT_PREFIX, ENGINE, PAYLOAD_ALIGNMENT, TOKEN_ID_LABEL,
    TOKEN_ID_LABEL_2, VERIFY_PROMPT, VERIFY_TIMEOUT, ArtifactUnverified,
    artifact_header_error, count_generated_tokens, verify_artifact,
)




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


def _frontend_resources(converter):
    """The frontend resources for a converter's target, found the way the front door does.

    ``NINFER_RESOURCE_ROOTS`` is the same variable ``import_model.py:63`` reads for the
    same purpose, so there is one convention and not two.  The names are asked of the
    target's own ``inventory.py`` rather than restated here: a second copy of a six-name
    list is a fork waiting to disagree with the converter.

    This used to hardcode ``tools/convert/qwen3_5_9b``.  It now takes the package from the
    row's converter path, so a second adaptation that needs frontend resources gets the
    same lookup and the same refusal without a second copy of it.

    Refuses by name when nothing fits.  This deliberately does NOT return a plausible
    default: a recipe that emits a command the converter will reject at run time is the
    "recognised but not doable" shape this project treats as worse than a refusal.
    """
    package = converter.rsplit('.', 1)[0]
    try:
        inventory = importlib.import_module(package + '.inventory')      # noqa: PLC0415
    except Exception as exc:                                            # noqa: BLE001
        raise ValueError(
            '%s 的转换器资源清单读不到 (%s/inventory.py): %s: %s'
            % (package.rsplit('.', 1)[-1], package.replace('.', '/'),
               type(exc).__name__, exc))
    names = [os.path.basename(n) for n in inventory.RESOURCE_NAMES]
    roots = [r for r in os.environ.get('NINFER_RESOURCE_ROOTS', '').split(os.pathsep) if r]
    tried = []
    for root in roots:
        missing = [n for n in names if not os.path.isfile(os.path.join(root, n))]
        if not missing:
            return root
        tried.append('%s (缺 %s)' % (root, ', '.join(missing)))
    raise ValueError(
        '%s 的转换器需要六个前端资源 (%s)；%s。'
        '补救：把 NINFER_RESOURCE_ROOTS 指向含这六个文件的目录（可多个，用 %r 分隔）。'
        % (package.rsplit('.', 1)[-1], ', '.join(names),
           '；'.join(tried) if tried else 'NINFER_RESOURCE_ROOTS 未设置',
           os.pathsep))


def _as_scan_dict(scan):
    """The scan as a plain dict.  Accepts a ``model_import.ModelScan`` too."""

    if isinstance(scan, dict):
        return scan
    as_dict = getattr(scan, 'as_dict', None)
    return as_dict() if callable(as_dict) else dict(vars(scan))


def plan_conversion(scan, model_id_out=None):
    """根据 scan 生成步骤链; 返回 (steps, out_name) 或抛 ValueError(不可转)。

    Routing is ``tools/convert/source_registry.route``: it refuses a source no row claims,
    and it refuses an arch-owned converter at a geometry that converter's own inventory
    does not implement (the guard ``import_model.py:791-804`` already applies).  Nothing in
    this function decides which converter runs.
    """
    scan = _as_scan_dict(scan)
    kind = scan.get('kind')
    src = scan.get('path')
    if kind == 'ninfer':
        return [], os.path.basename(src)
    if kind not in ('hf', 'gguf'):
        raise ValueError('该输入无法自动转换 (%s)' % kind)
    # ``model_id_out`` is the caller's family hint; it only fills a hole the scan left, it
    # never overrides what the source itself declared.
    if model_id_out and not scan.get('model_id'):
        scan = dict(scan, model_id=model_id_out)
    if kind == 'gguf':
        verdict = _gguf_reader_verdict(scan)
        if verdict:
            raise ValueError(verdict)
    row = registry.route(scan)
    # The anti-optimism gate: every source format the SCAN declared a tensor of must be
    # bindable by the row we just picked.  ``bindings_for`` existed and had NO production
    # caller -- its only call sites were the module's own __main__ self-test -- so the one
    # refusal that names the MISSING PART ("cannot carry N declared source format(s): ...
    # To fix: either add a FormatBinding ... or the artifact needs a new one") never ran
    # in service.
    # ``declared`` is None when the source's own format set is not available from the
    # scan.  That is UNMEASURED, not clear: the gate cannot fire and the caller must be
    # told so, because "the gate did not fire" and "the gate passed" are different facts.
    declared = registry.declared_source_formats(scan)
    if declared is None:
        plan_conversion.format_gate = "unmeasured"
    else:
        registry.bindings_for(row, declared)          # raises ValueError, named
        plan_conversion.format_gate = "measured:%s" % (",".join(declared),)
    resources = _frontend_resources(row.converter) if row.needs_resources else None
    out_path = os.path.join(MODELS_DIR, row.output)
    steps = registry.build_steps(row, registry.facts_from_scan(scan), src, out_path,
                                resources)
    return steps, row.output


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
        scan = _as_scan_dict(scan)
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
