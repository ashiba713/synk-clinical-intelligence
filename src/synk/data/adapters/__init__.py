"""Dataset adapters.

Adapters normalise external data sources (CSV exports, MIMIC-IV extracts) into
the canonical SYNK schemas defined in :mod:`synk.data.schemas`.  No adapter
downloads anything: restricted datasets must be obtained by the user under the
source provider's terms and placed in a local directory.
"""

from synk.data.adapters.csv_adapter import CsvAdapter
from synk.data.adapters.mimic_iv import MimicIVAdapter

REGISTRY = {
    "csv": CsvAdapter,
    "mimic_iv": MimicIVAdapter,
}


def get_adapter(name: str, **kwargs):
    """Return an adapter instance by name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset adapter '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["get_adapter", "CsvAdapter", "MimicIVAdapter"]
