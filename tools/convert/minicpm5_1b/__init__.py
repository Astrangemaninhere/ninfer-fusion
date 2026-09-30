# -*- coding: utf-8 -*-
"""MiniCPM5-1B (HF `LlamaForCausalLM`) — the worked example for tools/archkit/kit.

Five files, and the four roles they play:

  ``spec.json``      the geometry and the layer schedule (an archkit spec; the tree's
                     one spec directory holds the canonical copy)
  ``inventory.py``   roles: which artifact object, built from which source key
  ``recipe.py``      expressions: role -> payload, plus the tie rule
  ``declaration.py`` the three bound together, as a pure function of the checkpoint
  ``convert.py``     `driver.main(build_declaration)` and nothing else

See ``tools/archkit/kit/CONTRACT.md`` for the interface and
``tools/archkit/kit/AI_GUIDE.md`` for the fill-in-the-blanks procedure.
"""
