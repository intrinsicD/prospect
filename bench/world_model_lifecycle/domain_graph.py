"""Compatibility alias for the generic Prospect domain-graph codec.

WM-001 originally owned this experiment-neutral serializer. The authoritative
implementation now lives in :mod:`prospect.storage.domain_graph`. Aliasing the
module object preserves legacy imports and monkeypatch-based bound tests while
keeping one codec implementation.
"""

from __future__ import annotations

import sys

from prospect.storage import domain_graph as _domain_graph

sys.modules[__name__] = _domain_graph
