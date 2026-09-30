"""The one place a source ADAPTATION is declared.

WHY THIS MODULE EXISTS
----------------------
``tools/convert/import_model.py`` calls itself the single front door, and it is the more
complete of the two that exist: it declares its targets (``REGISTERED_TARGETS``, :55-80),
discovers each target's contract by convention (``target_source_contract``, :430-451), and
-- the part that matters -- it refuses to route an ARCH-owned converter at a geometry that
converter's own inventory does not implement (``ARCH_OWN_CONVERTER``, :111-113, guarded by
``_arch_owner_geometry``, :454-477, applied at :791-804).

``tools/gui/convert_runner.py`` is the auto-conversion front door, and it states the same
routing fact a second time with the guard dropped: ``_is_qwen3_5_gguf`` (:237-242) tests
``arch_names[0] == 'qwen35'`` alone, and ``plan_conversion`` tests it (:294) BEFORE the
family test (:304), so every 64-layer ``qwen35`` GGUF is handed to the 32-layer package
while the correct branch below is dead code for those files.  That defect is F1065's finding.

So the tree did not lack a mechanism; it had two, one of them the missing half of the other.
This module is that mechanism, promoted to data:

* the declared-table + derived-index + ``validate_*()`` idiom of
  ``tools/convert/qwen3_8_27b/recipe_nvfp4.py:330-461``
  (``FP8_WEIGHT_RECIPES`` -> ``FP8_WEIGHTS_BY_NAME`` -> ``validate_recipe()``); and
* the declared-routing + geometry-guard idiom of ``import_model.py:111-113``/:454-477.

It is also the Python-side analogue of ``src/core/arch_caps.h``: that file walks every
``NumericFormat`` and makes an uncovered format a BUILD FAILURE.  This table does the same
for SOURCES -- a source whose declaration is not covered by a row's format bindings is a
NAMED REFUSAL, never an optimistic acceptance.

THE CONTRACT, IN THE ORDER A NEW ADAPTATION IS ADDED
----------------------------------------------------
1. Append an :class:`Adaptation` to :data:`ADAPTATIONS`.  It is data: no branch anywhere.
2. If its source declares a quantisation the row does not bind, add a
   :class:`FormatBinding`.  One line closes one format.
3. Run :func:`validate_registry` (or ``python3 -m tools.convert.source_registry``).  It
   checks the row against the converter it names and against the artifact's own format
   registry, so a row cannot describe a converter or a format that does not exist.

Nothing here imports torch, and nothing here imports a converter at module import time:
the table is plain data, and the checks that need the converters import them inside
``validate_registry`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import sys
from types import MappingProxyType

#: The interpreter the step builder names.  ``convert_runner.py:36`` reads the same
#: variable for the same purpose, so there is one convention and not two.
PY = os.environ.get("NINFER_PYTHON") or sys.executable


def _repo_root() -> str:
    """The repository root, resolved from this file rather than from the caller's cwd.

    ``tools/convert/source_registry.py`` -> parents[2].  A step built from a relative path
    would silently depend on the caller's working directory, which is the failure
    ``import_model.py:41-50`` documents for the front door itself.
    """

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: Where a new adaptation is declared.  The refusals below quote this pair, so the message
#: cannot drift from the file -- the same reason ``import_model.py:1162`` names
#: ``REGISTERED_TARGETS`` instead of restating a list.
REGISTRY_FILE = "tools/convert/source_registry.py"
REGISTRY_LINE = "ADAPTATIONS, below"


@dataclass(frozen=True, slots=True)
class FormatBinding:
    """One source-side format declaration -> the artifact format that carries it.

    ``input_divisor`` is the field the EfficientThink/modelopt case turned on.  An FP8
    Linear whose source declares a static per-tensor activation multiplier
    (``input_scale`` F32 ``[]``) is NOT the same artifact object as the same codes with
    dynamic per-token activation quantisation: it needs one companion FP32 object holding
    ``fp32(1 / input_scale)``.  That companion, its name vocabulary, its ``fp32(1/x)``
    convention and its C++ consumer all already exist --
    ``recipe_nvfp4.InputDivisorRecipe`` (``recipe_nvfp4.py:69-70``),
    ``inventory_nvfp4.py:135``/``:143``, ``convert_modelopt.py:939`` and
    ``bind_nvfp4_weight`` (``bindings.cpp:80-103``) -- but they are declared only for
    NVFP4 parents (``recipe_nvfp4.py:159-192``, inside ``if layer in
    inventory.NVFP4_MLP_LAYERS:``) and the loader's FP8 branch calls ``bind_weight``
    (``bindings.cpp:143``), which takes no divisor parameter.

    F1087 RESOLVED THE FP8 HALF OF THIS AS AN EXPLICIT ACCEPTANCE, NOT AS A CARRIED
    OBJECT, and the deciding fact is a measurement rather than a preference: NOTHING in
    the engine reads ``Weight::input_scale_divisor`` for an FP8 weight.  ``grep -rn
    input_scale_divisor src/`` reaches five NVFP4 kernels
    (``ops/linear/nvfp4/nvfp4_w4a4.cu``, ``nvfp4_codec.cuh``, ``nvfp4_w4a4_plan.h``,
    ``nvfp4_w4a4_mma.cuh``, ``ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu``) and
    ``core/tensor.h:75``, and no FP8 op at all: ``ops/linear/fp8/fp8_a8.cu``'s
    ``fp8_a8_quantize_kernel`` (:19-68) derives the activation scale from the token's own
    amax, and the A16 route carries BF16 activations and needs no scale.  So an
    ``input_divisor=True`` FP8 binding would write a companion object the compute path
    ignores -- the same "admits a file it then cannot read" failure this module exists to
    prevent, one layer further in than ``bindings.cpp:143``.  ``FP8`` therefore declares
    ``input_divisor=False`` and names the consequence in its ``note``, and the converter's
    own ledger keeps ``D-ACTIVATION-SCALE`` for it.
    """

    source_format: str            # the source's own spelling of its quantisation
    artifact_format: str          # a key of tools.artifact.numeric.NUMERIC_FORMATS
    input_divisor: bool = False    # the artifact carries a paired fp32(1/input_scale) object
    #: Whether the LOADER can consume this binding today.  A binding the loader would
    #: ignore must not be declarable as available: that is the same "admits a file it then
    #: cannot read" failure F1068 proved for widening convert_nvfp4.py's literals, one layer
    #: up.  MEASURED state: NVFP4 + divisor is ready (``bind_nvfp4_weight`` reads and
    #: validates it, ``bindings.cpp:80-103``, reached from ``:392-395``).  FP8 + divisor is
    #: NOT -- ``bind_declared_weight``'s FP8 branch calls ``bind_weight``
    #: (``bindings.cpp:143``), which takes no divisor parameter at all -- and F1087 measured
    #: that no FP8 kernel reads the field either, so that binding is declared
    #: ``input_divisor=False`` rather than declared unavailable.
    loader_ready: bool = True
    note: str = ""


@dataclass(frozen=True, slots=True)
class Adaptation:
    """One source shape -> one converter, with every guard that decides whether it applies.

    ``geometry`` is ``((main_layers, nextn_layers), ...)``: the geometries this adaptation's
    own inventory implements, in the terms a source declares them.  It is REQUIRED for an
    arch-owned row -- one whose only entry is a weight-carrying ``convert.py`` that opens
    with a geometry gate and therefore has no pure-config validator to ask (the distinction
    ``import_model.py:480-509`` draws) -- and empty otherwise, because a row whose converter
    owns a ``validate_config`` must let that validator answer rather than keep a second copy
    of its pins (the rule stated at ``import_model.py:441-444``).
    """

    name: str                     # the target package, e.g. "qwen3_5_9b"
    converter: str                # dotted module that does the work
    source_kinds: tuple[str, ...] # ("gguf",) / ("hf",) -- the container sniffed
    arch: tuple[str, ...]         # GGUF general.architecture it claims; () = any
    model_ids: tuple[str, ...]    # scan model_id it claims; () = any
    quant_methods: tuple[str, ...]  # the source's own quantisation declaration; () = any
    geometry: tuple[tuple[int, int], ...]  # ((main, nextn), ...); () = the entry decides
    recipe_id: str                # must equal the converter module's own RECIPE_ID
    output: str                   # the artifact basename it writes
    argv: tuple[str, ...]         # the entry script, relative to the repository root
    #: The switch template, as data, so the step builder is a formatter rather than a
    #: dispatcher.  ``{source}``, ``{output}`` and ``{resources}`` are substituted.
    argv_switches: tuple[str, ...] = ()
    #: True when this converter needs a bf16 safetensors intermediate for a GGUF source.
    #: False when it reads the container itself -- ``tools/convert/qwen3_5_9b/convert.py``
    #: streams row blocks straight out of the GGUF precisely because the generic chain
    #: materialises an ~18 GB intermediate that never finished on this machine
    #: (``convert_runner.py:289-293``, measured).
    pre_extract: bool = False
    needs_resources: bool = False  # the six frontend resources
    formats: tuple[FormatBinding, ...] = ()   # THE BOUNDARY OF WHAT IS EXPRESSIBLE
    deviations: tuple[str, ...] = ()          # ledger codes this row is allowed to record

    @property
    def block_count_geometry(self) -> tuple[tuple[int, int], ...]:
        """The same geometries in the ``(block_count, nextn)`` terms a GGUF declares.

        ``block_count`` counts the draft blocks too, so it is ``main + nextn``.  Derived
        here rather than stored, because ``import_model.py:474-477`` reads the same pair out
        of the target's inventory for the same reason: two statements of one number drift.
        """

        return tuple((main + nextn, nextn) for main, nextn in self.geometry)


#: THE ONE PLACE.  Order is precedence: claimants are tried as declared, most specific
#: first, which is how "the arch-owned row, then the generic family row" reproduces the
#: decision the tree makes today -- minus the routing bug, because the arch-owned row
#: carries the geometry its converter implements.
ADAPTATIONS: tuple[Adaptation, ...] = (
    Adaptation(
        name="qwen3_5_9b",
        converter="tools.convert.qwen3_5_9b.convert",
        source_kinds=("gguf",),
        arch=("qwen35",),
        model_ids=(),
        quant_methods=(),
        # inventory.LAYERS=32, MTP_LAYERS=1.  In (block_count, nextn) terms these are the
        # pair tools/convert/qwen3_5_9b/convert.py:359 accepts, ((32,0),(33,1)) -- the
        # geometry gate whose absence from the GUI front door is F1065's defect.
        geometry=((32, 0), (32, 1)),
        recipe_id="qwen3_5_9b-gguf-v1",
        output="qwen3_5_9b_auto.ninfer",
        argv=("tools/convert/qwen3_5_9b/convert.py",),
        argv_switches=("--gguf", "{source}", "--resources", "{resources}",
                       "--out", "{output}"),
        pre_extract=False,
        needs_resources=True,
        formats=(
            FormatBinding("Q4_K", "Q4G64_F16S", note="gguf_kquant dequantizer"),
            FormatBinding("Q5_K", "Q5G64_F16S", note="gguf_kquant dequantizer"),
            FormatBinding("Q6_K", "Q6G64_F16S", note="gguf_kquant dequantizer"),
            FormatBinding("F32", "FP32"),
            FormatBinding("F16", "BF16", note="exact widening"),
            FormatBinding("BF16", "BF16"),
        ),
    ),
    Adaptation(
        name="qwen3_8_27b_groupwise",
        converter="tools.convert.qwen3_8_27b.convert",
        source_kinds=("gguf", "hf"),
        arch=(),
        model_ids=("qwen3.8-27b", "qwen3.6-27b"),
        quant_methods=("native", ""),
        geometry=(),   # the converter owns validate_config and pins its own geometry
        recipe_id="qwen3_8_27b-v1",
        output="qwen3_8_27b_auto.ninfer",
        argv=("tools/convert/qwen3_8_27b/convert.py",),
        argv_switches=("--model", "{source}", "--out", "{output}", "--device", "cuda"),
        pre_extract=True,
        formats=(
            FormatBinding("BF16", "BF16"),
            FormatBinding("F32", "FP32"),
        ),
    ),
    Adaptation(
        name="qwen3_8_27b_modelopt",
        converter="tools.convert.qwen3_8_27b.convert_modelopt",
        source_kinds=("hf",),
        arch=(),
        model_ids=("qwen3.8-27b",),
        quant_methods=("modelopt",),   # hf_quant_config.json MIXED_PRECISION + quant_algo
        geometry=(),
        recipe_id="qwen3_8_27b_modelopt-single-source-v1",
        output="qwen3_8_27b_nvfp4_modelopt.ninfer",
        argv=("tools/convert/qwen3_8_27b/convert_modelopt.py",),
        argv_switches=("--model", "{source}", "--out", "{output}", "--device", "cuda"),
        pre_extract=False,
        formats=(
            FormatBinding("NVFP4", "NVFP4", input_divisor=True,
                          note="weight_scale_2 F32 [] + input_scale F32 []"),
            # THE WORKED EXAMPLE, AND F1087'S FINDING: what this source actually carries
            # is the E4M3 CODES, not the activation multiplier.  D-ACTIVATION-SCALE is
            # discharged here as an EXPLICIT ACCEPTANCE with its consequence stated,
            # because the alternative is not one line but 144 new artifact objects plus a
            # kernel consumer (see the FormatBinding docstring for the measurement), and
            # because the engine's per-token dynamic scale is a strict superset of the
            # source's static per-tensor one: it is recomputed from the activations
            # actually being multiplied, so it can only reduce quantisation error.  The
            # owner accepted this on 2026-09-28 against that measurement.
            FormatBinding("FP8", "FP8_E4M3FN_ROW_BF16S",
                          note="E4M3 codes carried verbatim, narrowed once to per-row BF16 "
                               "scales (D-ROW-SCALE-BROADCAST, worst measured relative "
                               "error 3.3332e-03 <= the format's own 2**-8).  The "
                               "source's static per-tensor input_scale is NOT carried: "
                               "D-ACTIVATION-SCALE, accepted, because no FP8 kernel reads "
                               "a divisor (fp8_a8.cu:19-68 scales per token) and a carried "
                               "object the binder ignored is the failure this gate "
                               "prevents.  input_divisor=False is therefore the honest "
                               "declaration, not a disabled one."),
            FormatBinding("BF16", "BF16"),
        ),
        deviations=("D-ROW-SCALE-BROADCAST", "D-ACTIVATION-SCALE", "D-RECODED-FROM-BF16"),
    ),

    # ---------------------------------------------------------------------------------------
    # Muse-Glimmer-30B.  THE ROW F1169 NAMED, and the row that closes the two doors' disagreement:
    # import_model.py has carried "muse_glimmer_30b" in REGISTERED_TARGETS while this table had no
    # row for it, so the CLI claimed a source the seam refused by name ("no registered adaptation
    # claims this source ... kind='hf', model_id='muse-glimmer-30b'").
    #
    # quant_methods=("modelopt",) IS MEASURED, NOT COPIED.  The NVFP4-QAT revision's own config.json
    # carries quantization_config.quant_method = "modelopt" (quant_algo "MIXED_PRECISION", producer
    # name "modelopt"; G:\models\muse\Muse-Glimmer-30B-NVFP4\config.json, 88,682 B, sha256
    # f3add3cfff62038f...), and the target states the same fact about itself at
    # tools/convert/muse_glimmer_30b/config_pins.py:48 SOURCE_QUANT_METHODS = ("modelopt",).
    # "compressed-tensors" is a DIFFERENT model's spelling (gemma4_31b's -- import_model.py:71-74);
    # writing it here would give the one outcome worse than no row: a row that never matches.
    #
    # formats: the three the converter really carries -- NVFP4 direct pass-through for mlp/lm_head,
    # fp8 row-scale re-encode for the attention projections, bf16 for the norms -- which are the
    # same three F1169 measured as bindable under the modelopt row.
    #
    # needs_resources=False because this converter SUPPLIES the artifact's six frontend resources
    # from pinned copies of its own (config_pins.py:50-61, SUPPLIES_FRONTEND_RESOURCES = True), so a
    # caller must NOT hand it a resource root -- and there is nothing to hand: this package carries
    # no inventory.py for the chain's _frontend_resources() to read RESOURCE_NAMES from.
    Adaptation(
        name="muse_glimmer_30b",
        converter="tools.convert.muse_glimmer_30b.convert",
        source_kinds=("hf",),
        arch=(),
        model_ids=("muse-glimmer-30b",),   # the converter's own MODEL_ID (convert.py:45)
        quant_methods=("modelopt",),
        geometry=(),
        recipe_id="muse_glimmer_30b-modelopt-v1",
        output="muse_glimmer_30b_nvfp4.ninfer",
        argv=("tools/convert/muse_glimmer_30b/convert.py",),
        argv_switches=("--model", "{source}", "--out", "{output}", "--device", "cuda"),
        pre_extract=False,
        needs_resources=False,
        formats=(
            FormatBinding("NVFP4", "NVFP4"),
            FormatBinding("FP8", "FP8_E4M3FN_ROW_BF16S"),
            FormatBinding("BF16", "BF16"),
        ),
    ),
)

BY_NAME = MappingProxyType({row.name: row for row in ADAPTATIONS})


def registered_packages() -> tuple[str, ...]:
    """The converter PACKAGES this table routes to, in first-declaration order.

    Derived from ``converter`` rather than stored, because the package name is exactly what
    ``importlib.import_module("tools.convert.<package>.convert")`` needs.  Two rows may name
    the same package (``qwen3_8_27b`` has two adaptations), so this de-duplicates while
    keeping the order -- which is what makes it usable as ``import_model.py``'s
    ``REGISTERED_TARGETS`` without a second hardcoded list.
    """

    return tuple(dict.fromkeys(
        row.converter.rsplit(".", 1)[0].rsplit(".", 1)[-1] for row in ADAPTATIONS
    ))


def arch_owned_converters() -> dict:
    """``{gguf architecture: converter path}`` for the arch-owned rows, derived.

    This is ``import_model.py``'s ``ARCH_OWN_CONVERTER`` (``:111-113``), and deriving it is
    the point: that literal and ``convert_runner.py``'s ``QWEN3_5_GGUF_ARCH`` were the same
    fact stated twice, and only one of the two carried the geometry guard.
    """

    return {row.arch[0]: row.argv[0] for row in ADAPTATIONS if row.arch}


def geometry_for_converter(converter_path: str) -> tuple[tuple[int, int], ...]:
    """The accepted geometries of the row naming *converter_path*, in ``(main, nextn)``.

    THE UNITS ARE THE POINT.  The row stores ``(main_layers, nextn_layers)``, which is the
    same pair ``tools/convert/gguf_names.py:154`` derives from a GGUF's metadata and
    ``tools/gui/model_import.py:765`` prints (主栈 N 层 + 草稿块 M).  A caller comparing it
    against a file's own declaration must use THAT pair, not ``(block_count, nextn)``:
    ``block_count`` counts the draft blocks too, so the two differ by ``nextn`` and a
    comparison between them is always false.  ``block_count_geometry`` is available on the
    row for the callers that need the gate's own spelling.

    ``()`` when no row names it, so a caller keeps its previous behaviour rather than
    gaining a fabricated answer -- the rule ``import_model.py:464-467`` states for an
    unreadable inventory.
    """

    for row in ADAPTATIONS:
        if row.argv and row.argv[0] == converter_path:
            return row.geometry
    return ()


def facts_from_scan(scan: dict) -> dict:
    """The routing facts a scan declares, with UNDECLARED read as ``None``, never guessed.

    ``main_layers`` is what the source declares for its decoder tower; ``nextn_layers`` is
    the draft block it declares, or 0 when it declares the field and none is present, or
    ``None`` when the source never said.  The distinction matters: ``None`` must not be
    silently treated as "geometry matches", because that is exactly the optimistic routing
    F1065 found.
    """

    def _pick(*names):
        for name in names:
            value = scan.get(name)
            if isinstance(value, int) and value >= 0:
                return value
        return None

    return {
        "kind": scan.get("kind"),
        "arch": (scan.get("arch_names") or [""])[0],
        "model_id": scan.get("model_id") or "",
        "quant_method": scan.get("quant_method") or "",
        "main_layers": _pick("main_layers", "n_layers"),
        "nextn_layers": _pick("nextn_layers", "gguf_nextn_layers", "mtp_layers"),
    }


def declared_source_formats(scan: dict):
    """The source formats THIS SCAN declared a tensor of, or ``None`` when undeclared.

    THE THIRD STATE IS THE POINT.  ``None`` is not ``()``: an empty tuple would mean
    "the source declares no quantisation" (true for a bf16 checkpoint) and an
    undeclared set means "nobody looked" -- and the two must not be read alike, for
    the same reason *gaps* carries a ``gaps_measured`` flag rather than an absent key.

    GGUF can be answered TODAY: ``ModelScan.gguf_types`` is already
    ``{ggml type id: tensor count}``, read off the file's own tensor table, and
    ``gguf_kquant.GGML_TYPES`` is the id->name table.  So the set of source formats a
    GGUF declares is DERIVED, not listed -- which is what turns the format check from a
    second identity list into a named-missing-part check.

    HF is NOT answerable from the scan yet: a safetensors checkpoint declares its
    quantisation per tensor in ``quantization_config.quantized_layers`` /
    ``config_groups`` (muse's NVFP4 revision carries 58,268 characters of the former),
    and ``ModelScan`` does not carry that map.  Returning ``None`` here is the honest
    answer and it is a ONE-FIELD work item, named in the refusal path rather than
    guessed at.
    """
    kind = scan.get("kind")
    if kind == "gguf":
        types = scan.get("gguf_types") or {}
        if not types:
            return None
        from tools.convert import gguf_kquant as _kquant
        names = set()
        for type_id in types:
            try:
                names.add(_kquant.type_name(int(type_id)))
            except Exception:                                     # noqa: BLE001
                names.add("type%s" % (type_id,))
        return tuple(sorted(names))
    return None


def adaptations_for(scan: dict) -> tuple[Adaptation, ...]:
    """Every row that claims this source, in declaration order; () when none does."""

    facts = facts_from_scan(scan)
    declared = None
    if facts["main_layers"] is not None:
        declared = (facts["main_layers"], facts["nextn_layers"])
    claimed = []
    for row in ADAPTATIONS:
        if facts["kind"] not in row.source_kinds:
            continue
        if row.arch and facts["arch"] not in row.arch:
            continue
        if row.quant_methods and facts["quant_method"] not in row.quant_methods:
            continue
        if not row.quant_methods and facts["quant_method"]:
            continue
        if row.model_ids and facts["model_id"] not in row.model_ids:
            continue
        if row.geometry:
            # An arch-owned row claims a source only at a geometry its own inventory
            # implements.  A source that never declared the geometry is NOT claimed by
            # this row: routing it here is the F1065 defect, and refusing to decide is
            # the honest answer until the fact is available.
            if declared is None or facts["nextn_layers"] is None:
                continue
            if declared not in row.geometry:
                continue
        claimed.append(row)
    return tuple(claimed)


def route(scan: dict) -> Adaptation:
    """The one row that adapts this source, else a refusal that names the fix."""

    claimed = adaptations_for(scan)
    if claimed:
        return claimed[0]
    facts = facts_from_scan(scan)
    registered = ", ".join("%s(%s)" % (r.name, "/".join(r.source_kinds)) for r in ADAPTATIONS)
    # The refusal has to distinguish three different failures, because the fix differs:
    # a source whose geometry is unknown, a source inside a registered family at a
    # geometry no row implements, and a source outside the table entirely.
    if facts["main_layers"] is None or facts["nextn_layers"] is None:
        why = ("this source does not declare (main_layers, nextn_layers), and at least one "
               "registered row is arch-owned, so its geometry gate cannot be evaluated")
        fix = ("supply main_layers and nextn_layers in the scan (for a GGUF: block_count "
               "minus nextn_predict_layers, and nextn_predict_layers)")
    elif any(facts["arch"] in r.arch for r in ADAPTATIONS if r.arch):
        why = ("arch %r is claimed by an arch-owned row only at geometries this source is "
               "not at (declared (%r, %r))"
               % (facts["arch"], facts["main_layers"], facts["nextn_layers"]))
        fix = ("this file is not the geometry that arch-owned converter implements; route "
               "it through the family row, or add a row for this geometry to %s"
               % REGISTRY_FILE)
    else:
        why = ("kind=%r, arch=%r, model_id=%r, quant=%r is claimed by no row"
               % (facts["kind"], facts["arch"], facts["model_id"],
                  facts["quant_method"] or "-"))
        fix = ("append an Adaptation to %s (%s) naming the converter, its geometry if its "
               "only entry is a weight-carrying convert.py, and one FormatBinding per "
               "source format it can carry -- it is a data edit; no dispatcher control "
               "flow changes" % (REGISTRY_FILE, REGISTRY_LINE))
    raise ValueError("no registered adaptation claims this source: %s. Registered: %s. "
                     "To fix: %s." % (why, registered, fix))


def bindings_for(adaptation: Adaptation, source_formats) -> dict:
    """Map each declared source format to its binding; refuse the ones with no binding.

    This is the anti-optimism gate, and it is the reason the seam does not weaken the
    tree's strongest property.  ``convert_nvfp4.py`` refuses by matching literals, and
    F1068 proved that WIDENING those literals is wrong: that reader looks for
    ``weight_packed`` and per-row BF16 scales, so a widened predicate would admit a file
    it then cannot read.  The same trap opens for this table, in the other direction:
    a row that advertises a source format it has no encoder for would route a file to a
    converter that must then fail.  So an unbound format is a refusal HERE, by name, with
    the two distinct fixes -- bind it, or teach the artifact the format.
    """

    bound = {item.source_format: item for item in adaptation.formats}
    unmapped = [name for name in source_formats if name not in bound]
    if unmapped:
        artifact_vocabulary = ", ".join(_artifact_formats())
        raise ValueError(
            "adaptation %r cannot carry %d declared source format(s): %s. It carries: %s. "
            "To fix: either add a FormatBinding(source_format=%r, artifact_format=<one of "
            "%s>) to its row in %s (%s) -- one line, IF an existing artifact format "
            "already describes the codes -- or, when no registered format does, the "
            "artifact needs a new one: add its description to tools/artifact/numeric.py "
            "and its encoder/layout to tools/artifact/layouts.py first, because a "
            "converter must not emit a format the container cannot name."
            % (adaptation.name, len(unmapped), ", ".join(sorted(unmapped)),
               ", ".join(sorted(bound)) or "nothing",
               unmapped[0], artifact_vocabulary, REGISTRY_FILE, REGISTRY_LINE)
        )
    # A binding exists but its CONSUMER does not.  Refusing here is the whole point of
    # keeping this gate in the seam rather than in prose: declaring "we can carry static
    # activation scales for FP8" before the loader binds one would route a file to an
    # artifact whose own binder drops the scale -- a silent numerics change, which is
    # precisely what D-ACTIVATION-SCALE refuses to make.
    not_ready = [item for item in bound.values() if not item.loader_ready
                 and item.source_format in source_formats]
    if not_ready:
        item = not_ready[0]
        raise ValueError(
            "adaptation %r declares source format %r -> %s%s, but the LOADER cannot consume "
            "that binding yet, so emitting it would write an object the binder drops. "
            "To fix, land the consumer first: declare the divisor objects for this parent "
            "class (tools/convert/qwen3_8_27b/inventory_nvfp4.py, "
            "tools/convert/qwen3_8_27b/recipe_nvfp4.py -- mirror the NVFP4 rows at "
            "recipe_nvfp4.py:178-192, which are gated on `if layer in "
            "inventory.NVFP4_MLP_LAYERS:`), route the source's input_scale into "
            "fp32(1/input_scale) (convert_modelopt.py:939 already computes it), and bind it "
            "in bind_declared_weight's FP8 branch (src/targets/qwen3_6_27b/impl/load/"
            "bindings.cpp:143, which currently calls bind_weight, a function with no "
            "divisor parameter). That is the +41-line, one-CUDA-rebuild adaptation priced "
            "for F1078; flip loader_ready=True on this binding in the SAME change. "
            "F1087 CORRECTS THAT PRICE: those five files build the OBJECT, not a CONSUMER "
            "-- `grep -rn input_scale_divisor src/` shows no FP8 op reads the field, so a "
            "landed binding alone would still leave the companion unread, and a complete "
            "closure needs the 144 FP8 objects this artifact has no specs for plus an "
            "fp8_a8.cu static-scale path. Price the consumer before flipping this flag. "
            "Until then the honest answer is this refusal, not an optimistic route."
            % (adaptation.name, item.source_format, item.artifact_format,
               "+divisor" if item.input_divisor else ""))
    return {name: bound[name] for name in source_formats}


def build_steps(row: Adaptation, facts: dict, source: str, out_path: str,
                resources: str | None) -> list:
    """The command chain for one adaptation, as a pure formatter over row data.

    This is deliberately NOT a dispatcher: every difference between converters -- the
    switches, whether a GGUF needs a bf16 intermediate, whether frontend resources are
    required -- is read off the row.  Adding an adaptation adds no branch here, which is
    the property the order asked for.
    """

    repo = _repo_root()
    steps: list = []
    src = source
    if row.pre_extract and facts.get("kind") == "gguf":
        # GGUF -> bf16 safetensors (F32/F16/BF16 直读 + Q4_K/Q6_K 反量化), then convert.
        extract_dir = os.path.join(os.path.dirname(src), "_ninfer_bf16")
        steps.append(("gguf_extract", [PY, os.path.join(repo, "tools", "convert",
                                                        "gguf_extract.py"),
                                       "--src", src, "--out", extract_dir]))
        src = extract_dir
    argv = [PY, os.path.join(repo, row.argv[0])]
    substitutions = {"{source}": src, "{output}": out_path, "{resources}": resources or ""}
    for token in row.argv_switches:
        argv.append(substitutions.get(token, token))
    steps.append(("convert_%s" % row.name, argv))
    return steps


def _artifact_formats() -> tuple[str, ...]:
    """The artifact's own format names, asked of the registry rather than restated."""

    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.artifact.numeric import NUMERIC_FORMATS

    return tuple(NUMERIC_FORMATS)


def validate_registry() -> None:
    """Check every row against the converter and the formats it names.

    Same contract as ``recipe_nvfp4.validate_recipe``: the table is the one statement, and
    this is the check that it still describes something real.  It also gives ``RECIPE_ID``
    its first consumer in the tree -- today ``grep -rn RECIPE_ID`` finds only tests and the
    converters' own report JSON, so a ``RECIPE_ID`` that disagrees with the module it lives
    in is currently unobservable.
    """

    import importlib

    vocabulary = _artifact_formats()
    seen = set()
    for row in ADAPTATIONS:
        if row.name in seen:
            raise ValueError("duplicate adaptation name: %r" % row.name)
        seen.add(row.name)
        if row.source_kinds and not set(row.source_kinds) <= {"gguf", "hf", "ninfer"}:
            raise ValueError("%s: unknown source kind(s) %r" % (row.name, row.source_kinds))
        if row.geometry:
            for main, nextn in row.geometry:
                if not isinstance(main, int) or not isinstance(nextn, int) or nextn > main:
                    raise ValueError("%s: geometry (%r, %r) is not (main, nextn)"
                                     % (row.name, main, nextn))
        for binding in row.formats:
            if binding.artifact_format not in vocabulary:
                raise ValueError(
                    "%s: FormatBinding names artifact format %r, which is not in "
                    "tools/artifact/numeric.py's NUMERIC_FORMATS (%s)"
                    % (row.name, binding.artifact_format, ", ".join(vocabulary)))
            if binding.input_divisor and binding.artifact_format not in (
                    "NVFP4", "FP8_E4M3FN_ROW_BF16S"):
                raise ValueError(
                    "%s: input_divisor=True is declared for artifact format %r, but the "
                    "loader binds a paired input divisor for NVFP4 only "
                    "(bindings.cpp:441-442); declaring it elsewhere would promise a "
                    "consumer that does not exist"
                    % (row.name, binding.artifact_format))
        try:
            module = importlib.import_module(row.converter)
        except Exception as exc:                                  # noqa: BLE001
            raise ValueError("%s: converter %r cannot be imported: %s: %s"
                             % (row.name, row.converter, type(exc).__name__, exc))
        declared = getattr(module, "RECIPE_ID", None)
        if declared is None:
            # A thin target re-exports nothing.  MEASURED: ``qwen3_5_9b/convert.py``
            # declares no RECIPE_ID at all -- that package's identity lives in
            # ``qwen3_5_9b/inventory.py:41``, and convert.py reaches for it as
            # ``inv.RECIPE_ID`` (:439).  So look on the module AND on the modules it holds,
            # which is the idiom ``import_model._find_validator`` already uses (:405-421)
            # for the same problem on the validator.  Assuming where a target keeps its
            # identity is what made this check fail the first time it ran.
            package = row.converter.rsplit(".", 1)[0]
            for suffix in (".inventory", ".inventory_nvfp4"):
                try:
                    holder = importlib.import_module(package + suffix)
                except Exception:                                 # noqa: BLE001
                    continue
                declared = getattr(holder, "RECIPE_ID", None)
                if declared is not None:
                    break
        if declared is None:
            raise ValueError(
                "%s: neither %s nor that package's inventory declares RECIPE_ID, so this "
                "row's recipe_id %r cannot be checked against anything.  Fix the row "
                "rather than the check: name the module that owns the identity, or drop "
                "the field -- an unverifiable claim is the thing this table exists to "
                "replace" % (row.name, row.converter, row.recipe_id))
        if declared != row.recipe_id:
            raise ValueError(
                "%s: row says recipe_id %r, %s declares %r -- the row and the converter "
                "must agree" % (row.name, row.recipe_id, row.converter, declared))


if __name__ == "__main__":
    REACHED = (
        ("33-block qwen35 (Ornith-1.5-9B)",
         {"kind": "gguf", "arch_names": ["qwen35", "Ornith"],
          "main_layers": 32, "nextn_layers": 1, "model_id": ""}),
        ("65-block qwen35 (Qwen3.8-27B GGUF)",
         {"kind": "gguf", "arch_names": ["qwen35", "Q38"],
          "main_layers": 64, "nextn_layers": 1, "model_id": "qwen3.8-27b"}),
        ("HF bf16 (Qwen3.8-27B)",
         {"kind": "hf", "arch_names": ["Qwen3_5ForCausalLM"], "model_id": "qwen3.8-27b"}),
        ("HF W4A4+W8A8 (modelopt)",
         {"kind": "hf", "arch_names": ["Qwen3_5ForCausalLM"], "model_id": "qwen3.8-27b",
          "quant_method": "modelopt"}),
        ("qwen35 with NO declared geometry",
         {"kind": "gguf", "arch_names": ["qwen35", "x"], "model_id": "qwen3.8-27b"}),
        ("llama GGUF (outside the table)",
         {"kind": "gguf", "arch_names": ["llama"], "main_layers": 32,
          "nextn_layers": 0, "model_id": ""}),
    )
    for label, scan in REACHED:
        try:
            row = route(scan)
            text = row.name
            if row.geometry:
                text += "  geometry(block_count,nextn)=%s" % (row.block_count_geometry,)
            print("%-38s -> %s" % (label, text))
        except ValueError as exc:
            print("%-38s -> REFUSED: %s" % (label, exc))

    print()
    print("format boundary (the modelopt row, the worked example):")
    row = BY_NAME["qwen3_8_27b_modelopt"]
    print("  carries   : %s" % ", ".join(
        "%s->%s%s%s" % (b.source_format, b.artifact_format,
                        "+divisor" if b.input_divisor else "",
                        "" if b.loader_ready else " [loader_ready=False]")
        for b in row.formats))
    try:
        bindings_for(row, ("NVFP4", "FP8", "BF16", "MXFP4"))
    except ValueError as exc:
        print("  MXFP4     -> REFUSED: %s" % exc)
    print()
    # F1087: this row's FP8 binding used to be refused here.  It is now ACCEPTED, because
    # the seam declares what the source actually carries (E4M3 codes, no activation
    # multiplier) instead of declaring an object nothing consumes.  Printing the
    # acceptance is not enough on its own -- a gate that is never seen to fire is a gate
    # that is not there -- so the refusal is re-demonstrated below on a binding whose
    # consumer genuinely is not landed, built from THIS row so the demonstration cannot
    # drift away from it.
    print("the loader_ready gate (the seam must not admit what the binder would drop):")
    print("  NVFP4      -> %s" % bindings_for(row, ("NVFP4",))["NVFP4"].artifact_format)
    for name in ("FP8", "BF16"):
        binding = bindings_for(row, (name,))[name]
        print("  %-10s -> %s%s" % (name, binding.artifact_format,
                                   "+divisor" if binding.input_divisor else
                                   " (no divisor object carried)"))
    print("  all three  -> ACCEPTED: the ET config routes")
    gate_probe = replace(
        row,
        formats=row.formats + (FormatBinding("MXFP4", "NVFP4", loader_ready=False,
                                             note="gate demonstration"),),
    )
    try:
        bindings_for(gate_probe, ("NVFP4", "FP8", "MXFP4"))
        print("  gate       -> DID NOT FIRE (a defect: an unlanded consumer was admitted)")
    except ValueError as exc:
        import textwrap
        print("  gate       -> still refuses a binding whose consumer is not landed:")
        for line in textwrap.wrap(str(exc), 100):
            print("    " + line)
    print()
    print("validate_registry(): ", end="")
    try:
        validate_registry()
        print("OK (every recipe_id matches its module, every format is in NUMERIC_FORMATS)")
    except ValueError as exc:
        print("FAILED: %s" % exc)
