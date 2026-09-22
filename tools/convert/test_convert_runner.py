# -*- coding: utf-8 -*-
"""convert_runner 自测 —— 加载验证门槛的失败矩阵 + 独立 mapped/total。

退出码契约 (三态, 与 _ACCEPTANCE.md 元规则 B 一致):
  0  每一条检查都跑了, 而且都通过 (含真文件的交叉校验)
  1  有检查**失败**
  3  **不可用**: 有检查没做成 —— 真文件在一个读不动的挂载点上读不到。
     "读不到" 既不是通过也不是失败, 所以它有自己的退出码, 而且**绝不**打印 ALL PASS。
     为什么必须有这一态: 2026-09-14 两次真跑在这里死掉, 未捕获的 OSError(ENOMEM)
     冒到 main() 外面 ⇒ 进程 rc=1 而 FAIL=0。只看 FAIL 计数的门会把那种形状读成绿的
     (简报 §12.6 "rc 不能被吞成数据点" 的同型)。

为什么这个文件存在: ``verify_artifact()`` 是 GUI 导入链与 CLI 转换链上**唯一**一条
"产物真能跑"的判据 (convert_runner.py:66, import_model.py:784), 而它**一个自测都没有** ——
也就是说"这道门槛存在、而且真的会拦"这件事本身没有机器检查。"转换器 rc=0" 不等于
"产物能跑": 一个跑不起来的产物是**缺陷**, 不是成功 (这个模块自己的 docstring 就这么写)。

不需要 GPU: 门槛的"引擎那一半"由**桩引擎**驱动, 桩让每一条失败路径都能被走到 ——
这正是"没有对照组的检查等于没有检查"要求的东西。真引擎的正面证据 (真 artifact 真 load
出 token) 在 /home/user/scratch/res_imp2/REPORT.md 的 T1 节: 那需要 GPU 和一份
21.5 GB 的 .ninfer, 不适合放进一个随手可跑的自测。

    python3 tools/convert/test_convert_runner.py
"""
import atexit
import errno
import os
import shutil
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import convert_runner as cr                      # noqa: E402

#: Bounded re-attempts around the ONE step that touches a 5.78 GB file which lives on
#: /mnt/c (a Windows drive over this machine's relay).  Two real runs (15:39, 15:43)
#: died there on an EIGHT-BYTE read -- ``gguf_tensors.py:18`` and ``gguf_kquant.py:355``
#: -- while another process parsing the same file was fine, so a transient kernel error
#: is expected to clear and is worth re-attempting.  Nothing here turns a persistent
#: failure into a pass: after the attempts run out the reason is *reported* (rc 3).
REAL_FILE_ATTEMPTS = 4


class _FlakyFile:
    """Wraps a real file object and raises the MEASURED error on given read numbers.

    ``which reads`` is what makes this a control and not a mock: the injected error is
    exactly ``OSError(ENOMEM, "Cannot allocate memory")``, the one the two real runs
    raised, at a read *index* the retry logic has to survive.
    """

    def __init__(self, fh, fail_on):
        self._fh = fh
        self._fail_on = set(fail_on)
        self.n = 0

    def read(self, size=-1):
        self.n += 1
        if self.n in self._fail_on:
            raise OSError(errno.ENOMEM, "Cannot allocate memory")
        return self._fh.read(size)

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


def _flaky_opener(fail_on):
    """An ``opener`` for ``gguf_tensors.scan`` that injects transient read failures."""

    def opener(path, mode='rb'):
        return _FlakyFile(open(path, mode), fail_on)

    return opener


def _attempt(call, *args, **kwargs):
    """``(value, None)`` or ``(None, reason)`` -- an unreadable file is UNAVAILABLE.

    Not a failure (nothing about the code is disproved) and not a pass (nothing was
    checked), so it is returned as its own outcome for ``main()`` to report.
    """
    reason = None
    for attempt in range(REAL_FILE_ATTEMPTS):
        try:
            return call(*args, **kwargs), None
        except OSError as exc:
            # Name the errno: "ENOMEM" is what the two real runs raised, and a bare
            # "[Errno 12]" in a report is not a searchable fact.
            reason = "%s: %s" % (errno.errorcode.get(exc.errno, exc.errno), exc)
            if attempt + 1 < REAL_FILE_ATTEMPTS:
                time.sleep(0.3 * (attempt + 1))
    return None, reason


def main():
    d = tempfile.mkdtemp(prefix="ninfer_convrunner_test_")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    ok = True
    unavailable = []

    def check(label, cond, extra=""):
        nonlocal ok
        print(("%s  %s%s" % ("PASS" if cond else "FAIL", label, extra)))
        if not cond:
            ok = False

    def stub(name, body, rc=0, mode=0o755):
        """A fake engine.  Shell script, so its exit code and its output are ours."""
        p = os.path.join(d, name)
        with open(p, "w") as fh:
            fh.write("#!/bin/sh\n%s\nexit %d\n" % (body, rc))
        os.chmod(p, mode)
        return p

    def artifact(name, magic=cr.ARTIFACT_MAGIC, json_bytes=32, size=4096, body=None):
        p = os.path.join(d, name)
        with open(p, "wb") as fh:
            if body is not None:
                fh.write(body)
            else:
                head = cr.ARTIFACT_PREFIX.pack(magic, json_bytes)
                fh.write(head + b"\x00" * (size - len(head)))
        return p

    saved_engine = cr.ENGINE
    # The engine's own line, verbatim from a real run: 21.5 GB artifact, real engine,
    # 2026-09-14 (dl/ri_t1_real.txt).  One id, so the parser must say one.
    REAL_LINE = "tokens      generated ids             760\n"

    try:
        # ---- 1b. the ornith (qwen3.5-family GGUF) route ---------------------------
        # What this pins, and why each line can fail:
        #   * the front door's gate and the tier's own contract must name the same
        #     family.  Both directions are asserted, so an edit to either one that
        #     breaks the agreement goes red here instead of at a user.
        #   * the qwen3.5 recipe must be the STREAMING converter, not the generic
        #     gguf_extract -> convert chain whose ~18 GB bf16 intermediate never
        #     finished on this box.  Asserting "convert.py in argv" alone would pass
        #     on the old chain too (it also names a convert.py), so the check is that
        #     gguf_extract.py is ABSENT -- that is the load-bearing half.
        #   * and the door must not have been opened for everything: a GGUF whose arch
        #     is not qwen35 still gets refused.
        import importlib
        imp_mod = importlib.import_module("tools.convert.import_model")
        ext_mod = importlib.import_module("tools.convert.gguf_extract")

        check("ornith: qwen35 maps to a model_type the front door's own gate accepts",
              ext_mod.ARCH_TO_MODEL_TYPE["qwen35"][0] in imp_mod.DECODER_FAMILIES,
              " | %r vs %r" % (ext_mod.ARCH_TO_MODEL_TYPE["qwen35"][0],
                               imp_mod.DECODER_FAMILIES))
        # NEGATIVE CONTROL: the value this mapping actually had is outside the gate, so
        # the check above is a real comparison and not a tautology.
        check("ornith control: the pre-fix value 'qwen3' is NOT in that gate (so the "
              "check can go red)",
              "qwen3" not in imp_mod.DECODER_FAMILIES)
        check("ornith: the qwen3_5_9b tier is in the front door's REGISTERED_TARGETS",
              "qwen3_5_9b" in imp_mod.REGISTERED_TARGETS,
              " | %r" % (imp_mod.REGISTERED_TARGETS,))

        # The front door must not hand the user the generic chain for a family that has
        # its own converter -- that chain's bf16 intermediate is why ornith never
        # converted here -- and the file it names must really exist, or the advice is
        # not a route.
        own = imp_mod.ARCH_OWN_CONVERTER.get("qwen35")
        check("ornith: the front door names the family's own converter for qwen35",
              own == os.path.join("tools", "convert", "qwen3_5_9b", "convert.py"),
              " | %r" % (own,))
        check("ornith: that converter really exists (so the printed command is a route)",
              bool(own) and os.path.isfile(
                  os.path.join(str(imp_mod.REPO_ROOT), own)),
              " | %s" % os.path.join(str(imp_mod.REPO_ROOT), own or ""))

        # A scan shaped exactly like the real one (values measured on the real file,
        # dl/ornith_probe12.txt), so no multi-gigabyte read is needed to exercise it.
        ornith_scan = {
            "kind": "gguf",
            "path": os.path.join(d, "Ornith-1.5-9B-Q4_K_M.gguf"),
            "model_id": "qwen3-candidate",
            "arch_names": ["qwen35", "Ornith-1.5-9B-Q4_K_M"],
            "arch_note": "qwen35 (tensor table measures hidden=4096, 32 layers)",
            "gguf_types": {14: 35, 0: 184, 12: 223},
        }
        res = os.path.join(d, "resources")
        os.makedirs(res, exist_ok=True)
        try:
            from tools.convert.qwen3_5_9b import inventory as q35
            res_names = [os.path.basename(n) for n in q35.RESOURCE_NAMES]
        except Exception:                                              # noqa: BLE001
            res_names = []
            unavailable.append("ornith recipe (tools/convert/qwen3_5_9b/inventory.py "
                               "unimportable) -- a skip is NOT a pass")
        for name in res_names:
            with open(os.path.join(res, name), "wb") as fh:
                fh.write(b"{}")

        saved_roots = os.environ.get("NINFER_RESOURCE_ROOTS")
        try:
            os.environ["NINFER_RESOURCE_ROOTS"] = res
            if res_names:
                try:
                    steps, out_name = cr.plan_conversion(ornith_scan)
                    argv = steps[0][1] if steps else []
                    joined = " ".join(str(a) for a in argv)
                    check("ornith: the recipe is one streaming step",
                          len(steps) == 1 and steps[0][0] == "convert_qwen3_5_9b",
                          " | %r" % ([s[0] for s in steps],))
                    check("ornith: it invokes tools/convert/qwen3_5_9b/convert.py",
                          os.path.join("convert", "qwen3_5_9b", "convert.py") in joined,
                          " | %s" % joined)
                    check("ornith: it does NOT go through the 18 GB gguf_extract path",
                          "gguf_extract.py" not in joined, " | %s" % joined)
                    check("ornith: --gguf and --resources are both passed",
                          "--gguf" in argv and "--resources" in argv, " | %s" % joined)
                    check("ornith: the artifact is named for the family",
                          out_name == cr.QWEN3_5_OUT_NAME, " | %r" % out_name)
                except ValueError as exc:
                    check("ornith: plan_conversion routes the real qwen35 scan", False,
                          " | it raised: %s" % exc)

            # NEGATIVE CONTROL: a GGUF of another family must still be refused.
            other = dict(ornith_scan, arch_names=["gemma4", "gemma-4-31b"])
            try:
                cr.plan_conversion(other)
                red = "it was ROUTED -- the door was opened for everything"
            except ValueError as exc:
                red = str(exc)
            check("ornith control: an unregistered arch is still refused",
                  "不" in red or "refused" in red.lower(), " | %s" % red)

            # NEGATIVE CONTROL: without the six resources the recipe refuses by name
            # rather than emitting a command the converter cannot run.
            os.environ["NINFER_RESOURCE_ROOTS"] = os.path.join(d, "nowhere")
            try:
                cr.plan_conversion(ornith_scan)
                red2 = "it planned anyway -- a silent bad recipe"
            except ValueError as exc:
                red2 = str(exc)
            check("ornith control: missing resources are refused by name",
                  "NINFER_RESOURCE_ROOTS" in red2 and "tokenizer.json" in red2,
                  " | %s" % red2[:160])
        finally:
            if saved_roots is None:
                os.environ.pop("NINFER_RESOURCE_ROOTS", None)
            else:
                os.environ["NINFER_RESOURCE_ROOTS"] = saved_roots

        # ---- 1. the container header check, on its own (no engine involved) -------
        good = artifact("good.ninfer")
        check("header: a sound prefix passes", cr.artifact_header_error(good) == "",
              " | %r" % cr.artifact_header_error(good))
        zero_dir = artifact("zero_dir.ninfer", json_bytes=0)
        check("header: a zero-length directory is refused",
              "length field is 0" in cr.artifact_header_error(zero_dir),
              " | %s" % cr.artifact_header_error(zero_dir))
        wrong = artifact("wrong_magic.ninfer", magic=b"NOTNINF!")
        check("header: a wrong magic is refused",
              "not the v2 magic" in cr.artifact_header_error(wrong),
              " | %s" % cr.artifact_header_error(wrong))
        tiny = artifact("tiny.ninfer", body=b"NINF")
        check("header: a file shorter than the prefix is refused",
              "shorter than" in cr.artifact_header_error(tiny),
              " | %s" % cr.artifact_header_error(tiny))
        past = artifact("past_eof.ninfer", json_bytes=1 << 20, size=4096)
        check("header: a directory that ends past EOF is refused",
              "past the file" in cr.artifact_header_error(past),
              " | %s" % cr.artifact_header_error(past))
        check("header: a missing file is refused, not crashed on",
              "cannot stat" in cr.artifact_header_error(os.path.join(d, "nope.ninfer")),
              " | %s" % cr.artifact_header_error(os.path.join(d, "nope.ninfer")))

        # The named hole: the payload can still be short, because json_bytes bounds the
        # directory and the prefix carries no total-length field.  Pinned so the check is
        # not read as stronger than it is.
        truncated = os.path.join(d, "truncated.ninfer")
        with open(truncated, "wb") as fh:
            fh.write(cr.ARTIFACT_PREFIX.pack(cr.ARTIFACT_MAGIC, 32))
            fh.write(b"\x00" * (4096 - 16))
        check("header: NAMED HOLE -- an intact directory over a short payload passes here",
              cr.artifact_header_error(truncated) == "",
              " | %s (the engine and tools/artifact/container.py still refuse it)"
              % (cr.artifact_header_error(truncated) or "(no error)"))

        # ---- 2. the gate refuses a malformed artifact WITHOUT running the engine ---
        marker = os.path.join(d, "engine_was_called")
        touched = stub("engine_touch.sh", "touch %s" % marker)
        cr.ENGINE = touched
        for label, bad in (("zero_dir", zero_dir), ("wrong_magic", wrong)):
            if os.path.exists(marker):
                os.unlink(marker)
            try:
                cr.verify_artifact(bad)
                check("gate: %s is refused" % label, False, " | it PASSED")
            except cr.ArtifactUnverified as exc:
                check("gate: %s is refused" % label,
                      "不是 .ninfer 容器" in str(exc) and not os.path.exists(marker),
                      " | %s | engine called=%s" % (str(exc)[:80], os.path.exists(marker)))
        # control for the line above: with a SOUND artifact the same marker IS created,
        # so "engine called=False" is a real difference and not a stub that never ran.
        if os.path.exists(marker):
            os.unlink(marker)
        try:
            cr.verify_artifact(good)
        except cr.ArtifactUnverified:
            pass
        check("control: with a sound header the engine stub IS invoked",
              os.path.exists(marker), " | marker=%s" % os.path.exists(marker))

        # ---- 3. the load half: every failure mode must raise, and one must pass -----
        cases = [
            ("engine missing", os.path.join(d, "no_engine"), False, None),
            ("engine not executable", stub("engine_noexec.sh", "true\n", mode=0o644),
             False, None),
            ("engine rc != 0", stub("engine_rc1.sh",
                                    "echo 'error: cannot load' >&2", rc=1), False, None),
            ("engine rc=0, no output at all", stub("engine_silent.sh", "true\n"),
             False, None),
            ("engine rc=0, label with no ids", stub("engine_no_ids.sh",
                                                    "echo 'tokens      generated ids' >&2"),
             False, None),
            # POSITIVE CONTROL: this is the shape the real engine prints, and it goes to
            # stderr exactly as apps/cli/main.cpp:523 does (std::cerr).
            ("engine rc=0, one real id line", stub("engine_ok.sh",
                                                   "echo 'tokens      generated ids             760' >&2"),
             True, "1 个 token"),
        ]
        for label, engine, should_pass, needle in cases:
            cr.ENGINE = engine
            try:
                ev = cr.verify_artifact(good)
                got, detail = True, ev
            except cr.ArtifactUnverified as exc:
                got, detail = False, str(exc)
            check("load: %s -> %s" % (label, "PASS" if should_pass else "raise"),
                  got == should_pass and (needle is None or got is False or needle in detail),
                  " | %s" % detail[:90])

        # ---- 4. the trust boundary, pinned instead of assumed ----------------------
        # The second half of the gate is the engine binary named by ENGINE.  A stub that
        # prints a plausible id line while loading NOTHING passes -- by design (the engine
        # is the only thing here that can load an artifact), and this check exists so the
        # property is visible: do not read verify_artifact() as a proof that is
        # independent of the verifier.
        liar = stub("engine_liar.sh",
                    "echo 'tokens      generated ids 11 22 33' >&2 ; touch %s"
                    % os.path.join(d, "liar_ran"))
        cr.ENGINE = liar
        try:
            ev = cr.verify_artifact(good)
            check("trust boundary: a plausible line from a stub that loaded nothing PASSES",
                  "3 个 token" in ev and os.path.exists(os.path.join(d, "liar_ran")),
                  " | %s" % ev)
        except cr.ArtifactUnverified as exc:
            check("trust boundary: a plausible line from a stub that loaded nothing PASSES",
                  False, " | it raised: %s" % exc)

        # ---- 5. count_generated_tokens, against the line shapes that occur ---------
        table = [
            ("", None), ("no token line here\n", None),
            ("tokens      generated ids\n", 0),
            (REAL_LINE, 1),
            ("tokens      generated ids 1 2 3\n", 3),
            ("tokens      generated ids -1 -2 3\n", 3),        # negative ids are ids
            ("  generated tokens  5\n", None),                 # an indented metric, not the id line
            ("prompt tokens 12\ntokens      generated ids 7\n", 1),   # first match wins
            ("tokens      generated ids 4\nmore 9 9 9\n", 1),
        ]
        for text, want in table:
            got = cr.count_generated_tokens(text)
            check("parse: %r -> %r" % (text[:38], want), got == want, " | got %r" % got)

        # ---- 6. independent mapped/total -------------------------------------------
        # The claim under test is gguf_names.coverage()'s "N/N tensors mapped".  The
        # denominator is re-derived from the FILE by a second parser
        # (tools/archkit/gguf_tensors.py, whose header layout is spec-literal <IQQ), and
        # the ratio's identity is checked instead of read off the report.
        real_gguf = ("/mnt/c/Users/User/Documents/ziqinzhang/models/ornith-1.5-9b/"
                     "Ornith-1.5-9B-Q4_K_M.gguf")
        from tools.archkit import gguf_tensors as GT          # noqa: PLC0415
        from tools.convert import gguf_kquant as K            # noqa: PLC0415
        from tools.convert import gguf_names as GN            # noqa: PLC0415

        # 6a. the fixture the wizard's own test builds must be spec-conformant, i.e.
        # readable by a parser that shares NO code with it -- the "fixture and reader
        # share the wrong assumption" defect this project has already paid for once.
        sys.path.insert(0, os.path.join(REPO, "tools", "gui"))
        import test_model_import as tmi                        # noqa: PLC0415
        fx = os.path.join(d, "fixture.gguf")
        tmi.build_gguf(fx, [("token_embd.weight", 2, [5120, 151936], 1),
                            ("blk.0.attn_q.weight", 2, [5120, 5120], 12)])
        indep_fx = GT.scan(fx)
        check("fixture: a THIRD parser reads the wizard test's GGUF back",
              [t[0] for t in indep_fx] == ["token_embd.weight", "blk.0.attn_q.weight"]
              and tuple(indep_fx[0][2]) == (5120, 151936),
              " | %s" % [(t[0], tuple(t[2])) for t in indep_fx])

        def real_file_checks(report_unavailable):
            """Everything that needs the real 5.78 GB GGUF.  Returns None, or a reason.

            The readers live on tools/{archkit,convert}, and they retry a transient
            kernel read error themselves (``transient_read``); this wrapper is the
            second half of that fix -- a read that still cannot happen must be
            *reported* here, never raised out of ``main()``.
            """
            # "the fix is wired" is a claim about a call graph, so it is counted rather
            # than asserted: a retry helper nothing calls would pass every check below
            # (the retry would simply never fire) -- the "改了但没跑" defect this file
            # exists to make visible.  The counters wrap the SAME two calls the checks
            # use, so the extra cost is zero.
            saved, counted = {}, {}

            def counting(box, inner):
                def wrapped(fh, n):
                    box.append(n)
                    return inner(fh, n)
                return wrapped

            for mod, key in ((GT, "gguf_tensors"), (K, "gguf_kquant")):
                saved[key] = mod.transient_read
                counted[key] = []
                mod.transient_read = counting(counted[key], saved[key])
            try:
                independent, why_a = _attempt(GT.scan, real_gguf)        # (name, type, dims)
                # cache=False on purpose: a cache HIT is not a read, and the wiring count
                # below is a claim about reads.  With the module cache left on, any second
                # call inside one process reads nothing and the check would fail for a
                # reason that has nothing to do with the retry.
                table, why_k = _attempt(K.read_tensor_table, real_gguf, cache=False)
            finally:
                for mod, key in ((GT, "gguf_tensors"), (K, "gguf_kquant")):
                    mod.transient_read = saved[key]
            for key, box in counted.items():
                check("retry helper is wired into %s reading the real file" % key,
                      len(box) > 0, " | %d read(s)" % len(box))
            if why_a or why_k:
                for label, why in (("gguf_tensors.scan", why_a),
                                   ("gguf_kquant.read_tensor_table", why_k)):
                    if why:
                        report_unavailable("%s: %s" % (label, why))
                return "unreadable"
            kv, tensors, _off = table
            rep = GN.coverage(kv, tensors)
            check("mapped/total: two independent parsers agree on the tensor count",
                  len(independent) == len(tensors) == rep["total"],
                  " | gguf_tensors=%d gguf_kquant=%d report=%d"
                  % (len(independent), len(tensors), rep["total"]))
            check("mapped/total: both parsers see the same names",
                  {t[0] for t in independent} == {t[0] for t in tensors},
                  " | %d vs %d names" % (len({t[0] for t in independent}),
                                         len({t[0] for t in tensors})))
            check("mapped/total: mapped + unmapped == total (the identity it rests on)",
                  rep["mapped"] + len(rep["unmapped"]) == rep["total"],
                  " | %d + %d == %d" % (rep["mapped"], len(rep["unmapped"]), rep["total"]))
            check("mapped/total: the per-role denominators sum to total",
                  sum(int(v.split("/")[1]) for v in rep["roles"].values()) == rep["total"],
                  " | %s" % rep["roles"])
            check("mapped/total: the real file is fully mapped, no Hf collisions",
                  rep["mapped"] == rep["total"] and not rep["unmapped"]
                  and not rep["collisions"],
                  " | %d/%d unmapped=%s collisions=%s"
                  % (rep["mapped"], rep["total"], rep["unmapped"][:3],
                     rep["collisions"][:2]))
            check("mapped/total: architecture and layer split are the ones established",
                  rep["arch"] == "qwen35" and (rep["main_layers"], rep["nextn_layers"]) == (32, 1),
                  " | %s %d+%d" % (rep["arch"], rep["main_layers"], rep["nextn_layers"]))
            # control: the same coverage() function DOES name an unmapped tensor, so the
            # "unmapped == []" above is a fact about the file and not about the function.
            fake = list(tensors) + [("blk.0.not_a_real_role.weight", (10, 10), 0, 0)]
            rep2 = GN.coverage(kv, fake)
            check("control: coverage() names an unmapped tensor when there is one",
                  rep2["unmapped"] == ["blk.0.not_a_real_role.weight"]
                  and rep2["mapped"] == rep["mapped"],
                  " | %s" % rep2["unmapped"])

            # ---- 7. controls: is the transient-read retry load-bearing? ----------
            # (a) POSITIVE: the very same real file parses through injected read errors.
            #     The injection is at fixed READ INDICES, so the retry has to survive
            #     three distinct ones, spread over the metadata walk and the tensor table.
            try:
                flaky = GT.scan(real_gguf, opener=_flaky_opener({5, 9, 40}))
                check("retry: the real file still parses through 3 injected ENOMEM reads",
                      [x[0] for x in flaky] == [x[0] for x in independent],
                      " | %d tensors, injected at reads 5/9/40" % len(flaky))
            except OSError as exc:
                check("retry: the real file still parses through 3 injected ENOMEM reads",
                      False, " | it raised: %s" % exc)
            # (b) NEGATIVE (the "会红" control): switch the retry OFF and the SAME
            #     injection escapes.  Without this the check above is not evidence that
            #     the retry did anything -- the injection might simply never have fired.
            saved_attempts = GT.READ_ATTEMPTS
            GT.READ_ATTEMPTS = 1
            try:
                try:
                    GT.scan(real_gguf, opener=_flaky_opener({5}))
                    red = "it SURVIVED -- the injection is broken, so (a) proves nothing"
                except OSError as exc:
                    red = "%s" % exc
            finally:
                GT.READ_ATTEMPTS = saved_attempts
            check("control: with READ_ATTEMPTS=1 the same injection escapes "
                  "(so the retry is what saves it)",
                  red == "[Errno 12] Cannot allocate memory", " | %s" % red)

            # (c) the UNAVAILABLE guard itself, both directions, with an injected
            #     PERMANENT read failure -- deterministic, no mount luck involved.
            def _always_enomem(*_a, **_k):
                raise OSError(errno.ENOMEM, "Cannot allocate memory")

            saved_scan = GT.scan
            GT.scan = _always_enomem
            try:
                val, why2 = _attempt(GT.scan, real_gguf)
                check("control: an unreadable real file yields UNAVAILABLE "
                      "(no exception, no pass)",
                      val is None and "ENOMEM" in (why2 or ""), " | %s" % why2)
                try:
                    GT.scan(real_gguf)
                    pre = "it survived"
                except OSError as exc:
                    pre = "%s" % exc
                check("control: WITHOUT the guard the same error escapes (rc 1 with "
                      "FAIL=0 -- the shape both real runs had)",
                      pre == "[Errno 12] Cannot allocate memory", " | %s" % pre)
            finally:
                GT.scan = saved_scan
            return None

        if not os.path.exists(real_gguf):
            unavailable.append("独立 mapped/total (no real GGUF at %s) -- a skip is NOT a pass"
                               % real_gguf)
        else:
            why = real_file_checks(lambda r: unavailable.append("真文件交叉校验 %s" % r))
            if why:
                print("不可用  真文件读不到 ⇒ 独立 mapped/total 与它的对照**没做成**, "
                      "既不算通过也不算失败")
    finally:
        cr.ENGINE = saved_engine

    for line in unavailable:
        print("不可用  %s  (NOT a pass, NOT a failure)" % line)
    if not ok:
        print("== FAILURES PRESENT")
        return 1
    if unavailable:
        print("== 不可用: %d 节没做成 (见上) —— 这不是通过" % len(unavailable))
        return 3
    print("== ALL PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                       # noqa: BLE001
        # A traceback used to be the ONLY output of a bad run, and it left rc=1 with
        # FAIL=0 -- a shape in which "count the FAILs" reads as green.  The traceback
        # stays (it is the evidence), but the exit code is now also spelled out.
        print("== FAILURES PRESENT (uncaught exception above) -- this is rc 1, NOT a pass")
        raise
