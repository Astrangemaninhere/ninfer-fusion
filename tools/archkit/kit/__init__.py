# -*- coding: utf-8 -*-
"""tools.archkit.kit — the shared half of "support model X".

Import surface (the whole contract, four names):

    from tools.archkit.kit import Declaration, ObjectDecl, KitRefusal, driver

Everything else in this package is an implementation detail that a model-side
module never calls directly.  See `CONTRACT.md` for the one-page interface and
`AI_GUIDE.md` for the fill-in-the-blanks procedure.
"""

from .contract import (  # noqa: F401
    Declaration,
    EmitContext,
    KitRefusal,
    ObjectDecl,
    RESOURCE,
    TENSOR,
)
from .source import SafetensorsSource  # noqa: F401
from . import driver  # noqa: F401
from . import gate  # noqa: F401

__all__ = [
    "Declaration",
    "EmitContext",
    "KitRefusal",
    "ObjectDecl",
    "RESOURCE",
    "TENSOR",
    "SafetensorsSource",
    "driver",
    "gate",
]
