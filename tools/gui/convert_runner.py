#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_runner.py — 模型导入后的全自动转换编排 (A 层: 受支持家族)。

输入: model_import.scan + verdict -> 输出可执行命令链:
  * hf bf16 + 支持家族        -> convert.py (groupwise-int) -> .ninfer
  * hf NVFP4 (unsloth 风格)   -> convert_nvfp4.py (需 bf16 源 + quant 源)
  * gguf F16/BF16/F32         -> gguf_extract.py (抽 bf16 safetensors) -> convert.py
  * ninfer                    -> 无需转换 (直接入模型库)
其余: 礼貌拒绝 (K-quant/家族不支持/显存不够)。

任务执行器: 后台线程跑命令链, 状态可查询; 产物自动登记到模型目录。
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

REPO = r'C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo'
PY = r'C:\Program Files\Python312\python.exe'
MODELS_DIR = r'C:\Users\User\Documents\ziqinzhang\models'
CONVERT_DIR = os.path.join(REPO, 'tools', 'convert', 'qwen3_8_27b')


class Job:
    """单个转换任务的运行状态。"""
    def __init__(self, job_id, steps):
        self.job_id = job_id
        self.steps = steps          # [(label, cmd_list), ...]
        self.state = 'queued'       # queued|running|done|failed
        self.log = []
        self.step = 0
        self.done_steps = 0
        self.out_artifact = None
        self.error = None

    def as_dict(self):
        return {'job_id': self.job_id, 'state': self.state, 'step': self.step,
                'done_steps': self.done_steps, 'total': len(self.steps),
                'log': self.log[-80:], 'out_artifact': self.out_artifact,
                'error': self.error}


def plan_conversion(scan, model_id_out=None):
    """根据 scan 生成步骤链; 返回 (steps, out_name) 或抛 ValueError(不可转)。"""
    kind = scan.get('kind')
    fam = scan.get('model_id') or model_id_out
    src = scan.get('path')
    if kind == 'ninfer':
        return [], os.path.basename(src)
    if kind not in ('hf', 'gguf'):
        raise ValueError('该输入无法自动转换 (%s)' % kind)
    if fam not in ('qwen3.8-27b', 'qwen3.6-27b'):
        raise ValueError('家族 %s 尚无自动转换 recipe (ARCHKIT B 层)' % fam)
    out_name = 'qwen3_8_27b_auto.ninfer'
    steps = []
    if kind == 'gguf':
        quant = scan.get('quant_name', '')
        if any(q in quant for q in ('Q', 'K')):
            raise ValueError('K-quant GGUF 需先换 F16 源')
        # GGUF -> bf16 safetensors (F16/BF16/F32 直读)
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
        job.state = 'done'
        if job.out_artifact:
            job.log.append('产物: %s' % os.path.join(MODELS_DIR, job.out_artifact))

    def status(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
        return job.as_dict() if job else None


runner = Runner()
