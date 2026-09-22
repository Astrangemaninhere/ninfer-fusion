#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_fold_route.py — THE ONE PLACE the question "who applies the transform?" is decided.

WHY THIS FILE EXISTS, AND WHY IT IS SEPARATE FROM gguf_hadamard.py
------------------------------------------------------------------
``gguf_hadamard.py`` is the *arithmetic*: the matrix, the butterfly, the sign order, the head
permutation.  It answers "what does ``A`` equal".  It deliberately answers nothing about who
applies it, because that is not arithmetic.

This file is the *decision*: a rotated checkpoint's tensors are expressed in the basis
``A = H D``, and the map from that basis into the primal one must be applied **exactly once**.
Applied zero times the model is silently wrong (weights rotated, activation not).  Applied
twice it is *also* silently wrong, and this is the part that is easy to miss: ``A`` is an
orthonormal involution, so ``A^-1 A = I`` and the second application *cancels the first*,
leaving the same wrong answer as zero applications.  Measured on the real file, the two wrong
answers agree to fp32 roundoff (max|y_2x - y_0x| = 7.153e-07) while both differ from the right
answer by 2.474e+00 = 1.35 x |y|max -- i.e. the roundoff is 3.459e+06 times SMALLER than the
wrongness it conceals.  There is consequently no runtime self-check that can detect "did we
transform twice"; there is nothing to see at the output.  The discriminator must therefore be
**declared by the checkpoint and checked by both sides before a byte moves**.

THE DECISION IS A TOTAL FUNCTION OF THREE INPUTS
------------------------------------------------
    decide(basis, runtime_supports_transform, requested_policy) -> verdict

* ``basis`` is a FACT ABOUT THE CHECKPOINT, read from the checkpoint (never from a flag):
  ``primal`` (weights are in the natural basis) / ``folded`` (weights are in ``A``'s basis) /
  ``unstated`` (the checkpoint does not say).
* ``runtime_supports_transform`` is a FACT ABOUT THE RUNTIME (its build), not a preference.
* ``requested_policy`` is the operator's rail.  **It is a filter: it can only make the verdict
  stricter.**  No value of it can open the transform on a ``primal`` checkpoint, and none can
  close it on a ``folded`` one.

WHEN THE OPERATOR NAMES NO ROUTE, THE CHECKPOINT NAMES IT
--------------------------------------------------------
A checkpoint that declares its basis has already answered the question this module exists for.
A ``folded`` file still owes the transform; a ``primal`` file owes nothing and has nothing to
fold back.  So the answer to "which route is this file's" is IN the file, and a command line
that names no route must not invent one -- least of all ``refuse``, which means "emit nothing"
and is a value an operator types when that is what they want.

This distinction is not cosmetic.  A default of ``refuse`` reads as caution and behaves as a
refusal of every file, including the sources that were importing fine before the flag existed:
``tools/convert/convert_runner.py:320`` builds exactly the unflagged command, and
``tools/convert/import_model.py:733`` prints it as the advice (measured: rc=0 -> rc=3 on a
source that declares no fold at all).

:func:`declares_fold` reads the declaration, :func:`declared_basis_of` turns it into a basis,
:func:`route_for_declared_basis` turns the basis into a route, and no input selects ``refuse``.
An explicitly typed ``--fold-route`` still wins over all three: the derivation is what happens
when the operator said nothing.

THE TABLE, AND WHY EVERY CELL IS WRITTEN OUT
--------------------------------------------
Every cell is listed.  ``unstated`` is refused in all six of its cells rather than defaulted,
because "the checkpoint did not say" is not a licence to guess: for this model family the
folded and unrotated files differ in one tensor's inner head order *under the same tensor name*
(see ``gguf_hadamard.ssm_out_head_order``), so a guess is a corruption, not a default.

    basis    | runtime supports | policy           | verdict
    ---------+------------------+------------------+------------------
    primal   | any              | auto             | RUN_NO_TRANSFORM
    primal   | any              | require_primal   | RUN_NO_TRANSFORM
    primal   | any              | require_folded   | REFUSE   <- the double-apply door
    folded   | yes              | auto             | RUN_TRANSFORM
    folded   | yes              | require_folded   | RUN_TRANSFORM
    folded   | yes              | require_primal   | REFUSE   <- the skipped-transform door
    folded   | no               | any              | REFUSE
    unstated | any              | any              | REFUSE

``c + r == 1`` holds on every ADMITTED cell, where ``c`` counts the converter's ``A^-1`` and
``r`` counts the runtime's ``A``.  It holds *by construction* rather than by discipline:

* ``r`` is a function of ``basis`` alone (``r = 1`` iff ``basis == folded``) and no policy can
  change ``basis``;
* ``c`` is a function of the converter's route alone, and the route is checked against a
  **ledger of what the arithmetic actually did**, not against the flag's value
  (:func:`converter_admission`).  ``fold_back`` requires the ledger to show a completed,
  self-verified unfold of every declared tensor; ``pass_through`` requires the ledger to be
  EMPTY.  Those two requirements are disjoint, so no execution can satisfy both, so no
  checkpoint can be emitted by a converter that both folded and did not fold.

THE C++ MIRROR
--------------
``src/ops/linear/prism_fold/prism_fold.h`` implements the same function with the same enum
spellings and the same 18 cells.  ``tools/ternfold_probe/`` dumps both tables and compares them
cell for cell; that comparison is the only thing that keeps the two from drifting.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The declaration reader below names the contract's metadata keys, so it imports the module that
# owns those spellings.  ``gguf_hadamard`` imports neither this one nor any other of the tree's,
# so this edge cannot become a cycle; the path bootstrap is the same one gguf_extract.py and
# gguf_fold_back.py use, so this module is importable from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert import gguf_hadamard as H      # noqa: E402

# --------------------------------------------------------------------------- enums
# Spellings are duplicated verbatim in src/ops/linear/prism_fold/prism_fold.h.  They are part
# of the contract between the two sides, not internal names: a divergence here is exactly the
# class of defect this file exists to make impossible, so the probe compares the STRINGS.

BASIS_PRIMAL = "primal"
BASIS_FOLDED = "folded"
BASIS_UNSTATED = "unstated"
BASIS_VALUES = (BASIS_PRIMAL, BASIS_FOLDED, BASIS_UNSTATED)

POLICY_AUTO = "auto"
POLICY_REQUIRE_PRIMAL = "require_primal"
POLICY_REQUIRE_FOLDED = "require_folded"
POLICY_VALUES = (POLICY_AUTO, POLICY_REQUIRE_PRIMAL, POLICY_REQUIRE_FOLDED)

VERDICT_RUN_NO_TRANSFORM = "RUN_NO_TRANSFORM"
VERDICT_RUN_TRANSFORM = "RUN_TRANSFORM"
VERDICT_REFUSE = "REFUSE"
VERDICT_VALUES = (VERDICT_RUN_NO_TRANSFORM, VERDICT_RUN_TRANSFORM, VERDICT_REFUSE)

#: What the converter is asked to do with a rotated source.  Single-valued on purpose: the
#: two routes are two ways to the SAME correctness, so "both" is not a value this enum has.
ROUTE_PASS_THROUGH = "pass_through"   # Route A: emit the folded weights untouched, DECLARE them
ROUTE_FOLD_BACK = "fold_back"         # Route B: apply A^-1 here, emit primal, runtime does nothing
ROUTE_REFUSE = "refuse"               # a value an OPERATOR types; never a default --
                                      # see route_for_declared_basis()
ROUTE_VALUES = (ROUTE_PASS_THROUGH, ROUTE_FOLD_BACK, ROUTE_REFUSE)

#: The metadata key the emitted channel carries the basis under.  A ``folded`` checkpoint that
#: does not carry this key is inadmissible by construction (:func:`converter_admission`):
#: without it the runtime cannot tell a folded file from an unrotated one, and the difference is
#: a silently wrong matmul.
BASIS_KEY = "prism.fold.basis"

#: ``general.basename``'s spelling on a release whose weights are in the rotation's
#: basis.  The key is optional and the two files of this family disagree about carrying
#: it, which is why the contract block beside it is tested too -- see
#: :func:`declares_fold`.
FOLDED_BASENAME = "folded"


class FoldRouteRefused(RuntimeError):
    """A route decision that could not be made safely; the caller writes nothing."""


# ------------------------------------------------------------------ the source's own statement
def declares_fold(kv) -> bool:
    """Does this file say that its weights are in the rotation's basis?

    Two declarations count, and either one alone is enough, because they are the two the
    artefacts actually use:

    * ``prism.hadamard.version`` -- the contract block.  This is the fork's own test for "this
      file is rotated" (``llama-model.cpp:1195``), and it is the one that carries the arithmetic
      an unfold needs.
    * ``general.basename == 'folded'`` -- the release's name for itself.

    Measured on the two files this family ships (2026-09-20, header only,
    ``dl/foldroute/libs/hdr.txt``):

        Ternary-Bonsai-2-27B-PTQ1_0.gguf   version present,  basename 'folded'  -> True
        Bonsai-27B-Q1_0.gguf               no version key,   no basename key    -> False

    The two clauses can disagree in principle, and the direction that matters is the one they
    fail in: a file that calls itself ``folded`` and carries no contract is still a rotated file,
    and it is refused downstream by name -- "the contract does not parse" -- rather than passed
    through as primal, which would be the silent wrong.

    The answer is a property of the METADATA alone: nothing is read off disk and no state is
    carried in.
    """
    if (H.PREFIX + "version") in kv:
        return True
    return str(kv.get("general.basename") or "") == FOLDED_BASENAME


def declared_basis_of(kv) -> str:
    """The checkpoint's own statement about its weights: ``folded`` or ``primal``.

    The DECLARATION and not the proof.  Whether a declared fold's head order is proven is a
    separate question, asked by ``gguf_hadamard.gdn_ssm_out_is_grouped`` at the point where a
    caller needs the answer; this function must not pre-empt it, because "the file declares a
    fold and did not say which head order it kept" is a refusal (:class:`UnprovenRotation`) and
    not a basis.
    """
    return BASIS_FOLDED if declares_fold(kv) else BASIS_PRIMAL


def route_for_declared_basis(declared_basis) -> str:
    """The route the checkpoint's own declaration selects when the operator names none.

    ``folded`` -> ``fold_back``: the transform is still owed, and Route B is the one route whose
    output no runtime needs anything extra to run, which is why it is this converter's answer for
    a rotated source.  ``primal`` -> ``pass_through``: nothing is owed, so the converter's work is
    to copy the tensors and declare that they are primal.

    **No input returns** ``ROUTE_REFUSE``.  ``refuse`` means "emit nothing", and a derivation that
    answered it would refuse every file -- which is what the delivered default did, measured: the
    unflagged command at ``tools/convert/convert_runner.py:320`` went from rc=0 to rc=3 on a
    source that declares no fold.  A basis outside :data:`BASIS_VALUES` -- ``unstated`` included
    -- is refused here rather than answered, so no caller can turn "the file did not say" into a
    route.
    """
    if declared_basis == BASIS_FOLDED:
        return ROUTE_FOLD_BACK
    if declared_basis == BASIS_PRIMAL:
        return ROUTE_PASS_THROUGH
    raise FoldRouteRefused(
        "no route can be derived from a source that declares basis=%r; this tree derives one "
        "only from %s.  Nothing was written."
        % (declared_basis, " or ".join(repr(b) for b in (BASIS_FOLDED, BASIS_PRIMAL))))


# --------------------------------------------------------------------------- the decision
def decide(basis, runtime_supports_transform, requested_policy):
    """The total three-input verdict.  Never returns a value outside :data:`VERDICT_VALUES`.

    Refuses (rather than defaulting) on an unknown ``basis``, an unknown ``policy``, and on
    ``unstated``.  The order of the tests is part of the function: a checkpoint whose basis is
    unstated is refused BEFORE the policy is consulted, so no policy can rescue it.
    """
    if basis not in BASIS_VALUES:
        return VERDICT_REFUSE
    if requested_policy not in POLICY_VALUES:
        return VERDICT_REFUSE
    if basis == BASIS_UNSTATED:
        return VERDICT_REFUSE
    if basis == BASIS_PRIMAL:
        # The weights are already primal, so the converter has already done c = 1 and the
        # runtime must do nothing.  A policy that asks the runtime to transform anyway is
        # asking for the SECOND application; that is the double-apply door and it is refused.
        if requested_policy == POLICY_REQUIRE_FOLDED:
            return VERDICT_REFUSE
        return VERDICT_RUN_NO_TRANSFORM
    # basis == folded
    # A policy that forbids the one application this checkpoint needs is the skipped-transform
    # door.  Refused, not silently obeyed.
    if requested_policy == POLICY_REQUIRE_PRIMAL:
        return VERDICT_REFUSE
    if not runtime_supports_transform:
        # The runtime has no transform, so nothing can bring the activation into the weights'
        # basis.  This is a REFUSAL and not a warning: the alternative is the silent wrong.
        return VERDICT_REFUSE
    return VERDICT_RUN_TRANSFORM


def transform_counts(basis, verdict):
    """``(c, r)``: did the converter apply ``A^-1``, did the runtime apply ``A``.

    ``c`` is 1 exactly when the checkpoint is primal, because the only way a checkpoint IS
    primal is that the inverse was applied to the source -- by the converter for this family,
    since the release is folded.  A REFUSE verdict applies nothing, so it contributes ``0``.
    """
    if verdict == VERDICT_REFUSE:
        return (0, 0)
    if verdict == VERDICT_RUN_TRANSFORM:
        return (0, 1)
    return (1, 0)


def truth_table(runtime_supports_choices=(True, False)):
    """``{(basis, supports, policy): verdict}`` over the whole input space, in fixed order.

    The probe dumps this and the C++ mirror's table and compares them cell for cell.  Iteration
    order is fixed by the tuple constants above so that a diff is a diff of VALUES.
    """
    return {
        (basis, supports, policy): decide(basis, supports, policy)
        for basis in BASIS_VALUES
        for supports in runtime_supports_choices
        for policy in POLICY_VALUES
    }


def double_apply_reachable(table=None):
    """Every ``(basis, supports, policy)`` whose counts sum to 2.  Must be empty.

    This is criterion C1 checked against the executable table rather than against the prose.
    """
    table = truth_table() if table is None else table
    bad = []
    for key, verdict in table.items():
        basis, _supports, _policy = key
        c, r = transform_counts(basis, verdict)
        if c + r == 2:
            bad.append(key)
    return bad


def policies_that_open_the_transform(basis, supports=True):
    """Policies that yield RUN_TRANSFORM for this basis.  Empty for anything but ``folded``.

    Criterion C3, executable: ``requested_policy`` cannot move ``r`` off ``basis``.
    """
    return [p for p in POLICY_VALUES
            if decide(basis, supports, p) == VERDICT_RUN_TRANSFORM]


# --------------------------------------------------------------------------- the ledger
class FoldLedger:
    """What the converter's fold-back arithmetic ACTUALLY did, per declared tensor.

    The admission check reads this and not the flag, which is the whole point.  A route is a
    claim; the ledger is the record.  ``fold_back`` is admitted only when the ledger shows a
    completed, self-verified unfold for every declared name; ``pass_through`` only when the
    ledger is empty.  A converter that "selects" fold_back and folds nothing produces an empty
    ledger and is refused by the same predicate that admits a real fold.
    """

    __slots__ = ("expected", "records")

    def __init__(self, expected_names=()):
        self.expected = list(expected_names)
        self.records = []          # (name, unfolded, verified, max_delta)

    def record(self, name, unfolded, verified, max_delta):
        self.records.append((name, bool(unfolded), bool(verified), float(max_delta)))

    @property
    def unfolded_names(self):
        return [n for n, u, _v, _d in self.records if u]

    @property
    def unverified(self):
        return [n for n, u, v, _d in self.records if u and not v]

    @property
    def worst_delta(self):
        return max((d for _n, _u, _v, d in self.records), default=0.0)

    def render(self):
        return ("ledger: expected=%d recorded=%d unfolded=%d unverified=%d worst_delta=%.3e"
                % (len(self.expected), len(self.records), len(self.unfolded_names),
                   len(self.unverified), self.worst_delta))


def route_precheck(route, declared_basis, emitted_declaration):
    """The part of the gate that is knowable BEFORE the traverse runs.

    Exists because :func:`converter_admission` is the POST-condition: for ``fold_back`` it
    requires the ledger to already show a complete verified unfold, which by construction it
    cannot show before the work is done.  Calling the full admission as a pre-gate therefore
    refuses every honest ``fold_back`` run -- which is what the first version of
    ``gguf_fold_back.main`` did.

    So the two halves are split along the line of what each one can know:

      * this function refuses EARLY on what is a property of the REQUEST and the DECLARATIONS
        (route known, basis stated, a folded output has somewhere to declare itself);
      * :func:`converter_admission` refuses LATE on what is a property of the TRAVERSE, and it
        must be called INSIDE the output transaction so that its refusal still leaves zero files.

    This split does not make anything more permissive: the late check is the binding one and it
    runs on every emission.  A ``fold_back`` run that this function admits and that then folds
    nothing is refused by :func:`converter_admission` with the ledger in hand.
    """
    if route not in ROUTE_VALUES:
        raise FoldRouteRefused(
            "unknown --fold-route %r; this tree knows %s.  Nothing was written."
            % (route, ", ".join(ROUTE_VALUES)))
    if route == ROUTE_REFUSE:
        raise FoldRouteRefused(
            "--fold-route=refuse: the operator asked for no route, and no route is not the same "
            "as the permissive one.  Nothing was written.")
    if declared_basis == BASIS_UNSTATED:
        raise FoldRouteRefused(
            "the source does not state whether its weights are folded; the two files of this "
            "model family disagree about one tensor's inner head order under the same name, so "
            "guessing is a corruption rather than a default.  Nothing was written.")
    if declared_basis not in (BASIS_PRIMAL, BASIS_FOLDED):
        raise FoldRouteRefused("unknown declared basis %r.  Nothing was written."
                               % (declared_basis,))
    if declared_basis == BASIS_PRIMAL and route == ROUTE_FOLD_BACK:
        raise FoldRouteRefused(
            "--fold-route=fold_back on a source that declares no fold: there is nothing to fold "
            "back.  The operator selected the route for the wrong file, and the declaration says "
            "so.  Nothing was written.")
    if declared_basis == BASIS_FOLDED and route == ROUTE_PASS_THROUGH:
        if emitted_declaration != BASIS_FOLDED:
            raise FoldRouteRefused(
                "--fold-route=pass_through emits a checkpoint whose weights are in the rotation's "
                "basis, so the runtime must apply the transform to the activation.  That is only "
                "admissible if the checkpoint SAYS so, and this output channel would declare %r "
                "instead of %r.  Nothing was written." % (emitted_declaration, BASIS_FOLDED))
    if declared_basis == BASIS_FOLDED and route == ROUTE_FOLD_BACK:
        if emitted_declaration != BASIS_PRIMAL:
            raise FoldRouteRefused(
                "--fold-route=fold_back makes the weights primal, so the output must declare %r "
                "and it would declare %r.  Nothing was written."
                % (BASIS_PRIMAL, emitted_declaration))
    return True


def converter_admission(route, declared_basis, ledger, emitted_declaration):
    """The converter's own gate.  Returns the verdict; raises :class:`FoldRouteRefused` with the
    reason named when the verdict is REFUSE.

    ``declared_basis``   -- what the SOURCE declares (``folded`` for a rotated GGUF, ``primal``
                            for an unrotated one, ``unstated`` when nothing is declared).
    ``ledger``           -- what this run's arithmetic did.
    ``emitted_declaration`` -- what the OUTPUT will carry, or ``None`` if the channel cannot
                            carry one.

    The three cases, and the reason each is a refusal rather than a warning:

    1. source is not rotated       -> nothing to transform, nothing to declare; the route must
                                      not be ``fold_back`` (nothing to fold back => the operator
                                      believed the file was folded) and must not be
                                      ``pass_through`` on a file we cannot even identify.
    2. source is rotated + fold_back -> the ledger must show every declared name unfolded AND
                                      verified, and the output must declare ``primal``.
                                      A short ledger means the traversal did not happen.
    3. source is rotated + pass_through -> the ledger must be EMPTY, and the output MUST carry a
                                      declaration saying ``folded``.  No declaration channel =>
                                      REFUSE; that is the case that would otherwise be a
                                      silently-wrong checkpoint for every runtime without the
                                      transform, which is exactly what the guard has always
                                      refused.
    """
    if route not in ROUTE_VALUES:
        raise FoldRouteRefused(
            "unknown --fold-route %r; this tree knows %s.  Nothing was written."
            % (route, ", ".join(ROUTE_VALUES)))
    if route == ROUTE_REFUSE:
        raise FoldRouteRefused(
            "--fold-route=refuse: the operator asked for no route, and no route is not the same "
            "as the permissive one.  Nothing was written.")

    if declared_basis == BASIS_UNSTATED:
        raise FoldRouteRefused(
            "the source does not state whether its weights are folded; the two files of this "
            "model family disagree about one tensor's inner head order under the same name, so "
            "guessing is a corruption rather than a default.  Nothing was written.")
    if declared_basis not in (BASIS_PRIMAL, BASIS_FOLDED):
        raise FoldRouteRefused("unknown declared basis %r.  Nothing was written." % (declared_basis,))

    if declared_basis == BASIS_PRIMAL:
        if route == ROUTE_FOLD_BACK:
            raise FoldRouteRefused(
                "--fold-route=fold_back on a source that declares no fold: there is nothing to "
                "fold back.  The operator selected the route for the wrong file, and the "
                "declaration says so.  Nothing was written.")
        if ledger.unfolded_names:
            raise FoldRouteRefused(
                "--fold-route=pass_through on a source that declares no fold, but this run "
                "unfolded %d tensor(s): the route and the arithmetic disagree.  Nothing was "
                "written." % len(ledger.unfolded_names))
        if emitted_declaration not in (None, BASIS_PRIMAL):
            raise FoldRouteRefused(
                "the output would declare basis=%r while the source declares %r and no unfold "
                "was performed.  Nothing was written." % (emitted_declaration, declared_basis))
        return VERDICT_RUN_NO_TRANSFORM

    # ---- declared_basis == BASIS_FOLDED: the only case where a transform must happen
    if route == ROUTE_FOLD_BACK:
        if len(ledger.unfolded_names) != len(ledger.expected) or not ledger.expected:
            raise FoldRouteRefused(
                "--fold-route=fold_back was selected and the traverse unfolded %d of %d declared "
                "tensor(s).  A partial fold-back leaves a checkpoint whose tensors are in two "
                "different bases at once and nothing in the output would say which.  Nothing was "
                "written.  %s" % (len(ledger.unfolded_names), len(ledger.expected), ledger.render()))
        if ledger.unverified:
            raise FoldRouteRefused(
                "--fold-route=fold_back: %d tensor(s) were unfolded without their round-trip "
                "proof (%s).  An unproven unfold is a guess with a matrix attached.  Nothing was "
                "written.  %s"
                % (len(ledger.unverified), ", ".join(ledger.unverified[:4]), ledger.render()))
        if emitted_declaration != BASIS_PRIMAL:
            raise FoldRouteRefused(
                "--fold-route=fold_back: the arithmetic made the weights primal, so the output "
                "must declare basis=%r, and it would declare %r.  A checkpoint whose bytes and "
                "whose declaration disagree is the defect this gate exists for.  Nothing was "
                "written." % (BASIS_PRIMAL, emitted_declaration))
        return VERDICT_RUN_NO_TRANSFORM

    # route == ROUTE_PASS_THROUGH, source folded: the output stays folded and MUST say so.
    #
    # The test is `unfolded_names`, NOT `records`.  A pass_through traverse legitimately records
    # every tensor it CONSIDERED (with `unfolded=False`), so `records` is non-empty on a correct
    # pass_through run; the predicate that means "the arithmetic disagreed with the route" is that
    # something was actually UNFOLDED.  Written the other way this refused every honest
    # pass_through run -- which is how the mistake was found, by the post-condition re-check in
    # gguf_fold_back.py firing on a run that had done nothing wrong.
    if ledger.unfolded_names:
        raise FoldRouteRefused(
            "--fold-route=pass_through: the output is supposed to keep the weights folded, but "
            "this run unfolded %d tensor(s) (%s).  Applying the route AND the fold is the double "
            "transform.  Nothing was written.  %s"
            % (len(ledger.unfolded_names), ", ".join(ledger.unfolded_names[:4]), ledger.render()))
    if emitted_declaration != BASIS_FOLDED:
        raise FoldRouteRefused(
            "--fold-route=pass_through emits a checkpoint whose weights are in the rotation's "
            "basis, so the runtime must apply the transform to the activation.  That is only "
            "admissible if the checkpoint SAYS so, and this output channel would declare %r "
            "instead of %r.  A folded checkpoint that does not declare itself is read as an "
            "unrotated one and every folded matmul is silently wrong -- which is what this guard "
            "has always refused.  Nothing was written."
            % (emitted_declaration, BASIS_FOLDED))
    return VERDICT_RUN_TRANSFORM


def _selftest():
    """Prove the table against itself.  Exits non-zero on any disagreement."""
    t = truth_table()
    assert len(t) == 18, "the table must be total over 3 x 2 x 3, got %d" % len(t)
    assert all(v in VERDICT_VALUES for v in t.values()), "a cell returned a value outside the enum"

    bad = double_apply_reachable(t)
    assert not bad, "c + r == 2 is reachable: %r" % (bad,)

    # every admitted cell has c + r == 1, and every refused cell applies nothing
    for (basis, _s, _p), v in t.items():
        c, r = transform_counts(basis, v)
        assert c + r == (0 if v == VERDICT_REFUSE else 1), (basis, v, c, r)

    # C3: no policy opens the transform on anything but a folded checkpoint
    for basis in BASIS_VALUES:
        openers = policies_that_open_the_transform(basis)
        assert openers == ([POLICY_AUTO, POLICY_REQUIRE_FOLDED] if basis == BASIS_FOLDED else []), \
            (basis, openers)

    # the four RED cells: each must be shown to be red by a concrete input
    red = [(BASIS_PRIMAL, True, POLICY_REQUIRE_FOLDED),
           (BASIS_PRIMAL, False, POLICY_REQUIRE_FOLDED),
           (BASIS_FOLDED, True, POLICY_REQUIRE_PRIMAL),
           (BASIS_FOLDED, False, POLICY_AUTO),
           (BASIS_UNSTATED, True, POLICY_AUTO),
           (BASIS_UNSTATED, False, POLICY_REQUIRE_FOLDED)]
    for key in red:
        assert t[key] == VERDICT_REFUSE, "a REFUSE cell did not go red: %r -> %r" % (key, t[key])

    # the unknown-input cells are refusals too, not exceptions and not defaults
    assert decide("primal-ish", True, POLICY_AUTO) == VERDICT_REFUSE
    assert decide(BASIS_PRIMAL, True, "force_it") == VERDICT_REFUSE

    # ---- the ledger gate: the two routes' requirements are disjoint
    ok = FoldLedger(["blk.0.ssm_out.weight"])
    ok.record("blk.0.ssm_out.weight", True, True, 0.0)
    # fold_back + a real fold + the right declaration -> admitted, and the verdict is
    # RUN_NO_TRANSFORM because the CONVERTER did the work (c=1, r=0).  The verdict names what
    # the RUNTIME does, which is why it reads "no transform" for the route that transforms.
    v = converter_admission(ROUTE_FOLD_BACK, BASIS_FOLDED, ok, BASIS_PRIMAL)
    assert v == VERDICT_RUN_NO_TRANSFORM, v
    assert transform_counts(BASIS_PRIMAL, v) == (1, 0)
    # fold_back + the WRONG declaration -> refused (bytes and declaration disagree)
    try:
        converter_admission(ROUTE_FOLD_BACK, BASIS_FOLDED, ok, BASIS_FOLDED)
    except FoldRouteRefused as exc:
        assert BASIS_PRIMAL in str(exc) and BASIS_FOLDED in str(exc)
    else:
        raise AssertionError("fold_back accepted a folded declaration")
    # fold_back + an EMPTY ledger -> refused (the flag said fold, nothing folded)
    try:
        converter_admission(ROUTE_FOLD_BACK, BASIS_FOLDED, FoldLedger(["a"]), BASIS_PRIMAL)
    except FoldRouteRefused:
        pass
    else:
        raise AssertionError("fold_back accepted an empty ledger")
    # fold_back + a fold that did not verify -> refused
    nv = FoldLedger(["a"])
    nv.record("a", True, False, 1e9)
    try:
        converter_admission(ROUTE_FOLD_BACK, BASIS_FOLDED, nv, BASIS_PRIMAL)
    except FoldRouteRefused:
        pass
    else:
        raise AssertionError("fold_back accepted an UNVERIFIED unfold")
    # pass_through + a non-empty ledger -> refused (the double transform)
    try:
        converter_admission(ROUTE_PASS_THROUGH, BASIS_FOLDED, ok, BASIS_FOLDED)
    except FoldRouteRefused as exc:
        assert "double" in str(exc)
    else:
        raise AssertionError("pass_through accepted a non-empty ledger")
    # pass_through + no declaration channel -> refused
    try:
        converter_admission(ROUTE_PASS_THROUGH, BASIS_FOLDED, FoldLedger(["a"]), None)
    except FoldRouteRefused:
        pass
    else:
        raise AssertionError("pass_through accepted an undeclarable folded checkpoint")
    # pass_through + an empty ledger + the folded declaration -> admitted, and the runtime transforms
    assert converter_admission(ROUTE_PASS_THROUGH, BASIS_FOLDED, FoldLedger(["a"]),
                               BASIS_FOLDED) == VERDICT_RUN_TRANSFORM
    # refuse route, and a route the enum does not have -> both refused
    for r in (ROUTE_REFUSE, "both", ""):
        try:
            converter_admission(r, BASIS_FOLDED, FoldLedger(["a"]), BASIS_PRIMAL)
        except FoldRouteRefused:
            pass
        else:
            raise AssertionError("route %r was admitted" % (r,))
    # an unrotated source: pass_through is fine, fold_back is the wrong route
    assert converter_admission(ROUTE_PASS_THROUGH, BASIS_PRIMAL, FoldLedger(),
                               BASIS_PRIMAL) == VERDICT_RUN_NO_TRANSFORM
    try:
        converter_admission(ROUTE_FOLD_BACK, BASIS_PRIMAL, FoldLedger(), BASIS_PRIMAL)
    except FoldRouteRefused:
        pass
    else:
        raise AssertionError("fold_back on an unrotated source was admitted")

    # ---- the derivation: the source's own declaration picks the route, and `refuse` is
    # never the answer.  These are the cells the two Bonsai files land on.
    assert declares_fold({H.PREFIX + "version": 1}) is True
    assert declares_fold({"general.basename": "folded"}) is True
    assert declares_fold({}) is False
    assert declares_fold({"general.basename": "Hf"}) is False
    assert declared_basis_of({H.PREFIX + "version": 1,
                              H.GDN_V_GROUPED_KEY: True}) == BASIS_FOLDED
    assert declared_basis_of({}) == BASIS_PRIMAL
    # a folded source derives the route that folds it back, and the guard admits that pair
    # exactly because the output declares primal -- the two statements are checked together
    # here so that they cannot drift apart
    assert route_for_declared_basis(BASIS_FOLDED) == ROUTE_FOLD_BACK
    assert route_precheck(ROUTE_FOLD_BACK, BASIS_FOLDED, BASIS_PRIMAL)
    # a primal source derives the route that copies it, and nothing is owed
    assert route_for_declared_basis(BASIS_PRIMAL) == ROUTE_PASS_THROUGH
    for basis in (BASIS_FOLDED, BASIS_PRIMAL):
        assert route_for_declared_basis(basis) != ROUTE_REFUSE
    for unknown in (BASIS_UNSTATED, "folded-ish", None):
        try:
            route_for_declared_basis(unknown)
        except FoldRouteRefused:
            pass
        else:
            raise AssertionError("a route was derived from %r" % (unknown,))

    print("derived: folded -> fold_back, primal -> pass_through; refuse is not an answer")
    print("table: %d cells, total, no cell yields c+r==2" % len(t))
    print("C3: the only basis whose transform a policy can open is 'folded'")
    print("red cells: %d, all red" % len(red))
    print("ledger gate: fold_back needs a full VERIFIED unfold; pass_through needs an EMPTY ledger"
          " and a folded declaration; the two requirements are disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
