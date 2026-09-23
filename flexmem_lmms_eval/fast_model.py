"""lmms-eval model class for the repository's FlexMem-fast variant.

The two upstream variants intentionally share the top-level ``llava`` and
``flexmem`` import names. This module puts FlexMem-fast first on ``sys.path``
before importing the common adapter, so each lmms-eval process uses one
consistent implementation. A wheel bundles this source tree as package data;
an editable ``uv sync`` install uses the repository's original directory.
"""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path
from typing import Any


def _fast_source_root() -> Path:
    checkout_source = Path(__file__).resolve().parents[1] / "FlexMem-fast"
    if checkout_source.is_dir():
        return checkout_source

    packaged_source = Path(files("flexmem_lmms_eval").joinpath("fast_source"))
    if packaged_source.is_dir():
        return packaged_source
    raise FileNotFoundError("FlexMem-fast source files are unavailable in this installation.")


_FAST_SOURCE_ROOT = _fast_source_root()

for _module_name in ("flexmem", "llava"):
    _loaded = sys.modules.get(_module_name)
    _loaded_path = Path(getattr(_loaded, "__file__", "")) if _loaded is not None else None
    if _loaded_path and _FAST_SOURCE_ROOT not in _loaded_path.parents:
        raise RuntimeError(
            "flexmem_fast must be started in a fresh Python process; another FlexMem implementation is already imported."
        )

if str(_FAST_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FAST_SOURCE_ROOT))

from .model import FlexMem


class FlexMemFast(FlexMem):
    """FlexMem-fast with its original MLVU-oriented settings and implementation."""

    def __init__(self, config_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(config_path=config_path or str(_FAST_SOURCE_ROOT / "config.yaml"), **kwargs)
