"""Generic CSV adapter.

Loads locally provided CSV files and maps their columns onto the canonical
SYNK schema using a user-supplied mapping, so arbitrary institutional exports
can be ingested without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from synk.data.schemas import coerce_notes, coerce_observations, coerce_outcomes
from synk.data.validation import validate_notes, validate_observations, validate_outcomes
from synk.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CsvAdapter:
    """Load canonical CSVs, optionally renaming columns via ``column_mapping``.

    ``column_mapping`` maps canonical name -> source column name, e.g.
    ``{"heart_rate": "HR", "mean_arterial_pressure": "MAP"}``.
    """

    directory: Path
    column_mapping: Optional[dict[str, str]] = None
    drop_columns: list[str] = field(default_factory=list)

    def _read(self, filename: str) -> Optional[pd.DataFrame]:
        path = Path(self.directory) / filename
        if not path.exists():
            logger.warning("CSV adapter: file not found, skipping: %s", path)
            return None
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[c for c in self.drop_columns if c in frame.columns])
        if self.column_mapping:
            inverse = {}
            for canonical, source in self.column_mapping.items():
                if source in frame.columns:
                    inverse[source] = canonical
            frame = frame.rename(columns=inverse)
        return frame

    def load(self):
        observations = self._read("observations.csv")
        notes = self._read("notes.csv")
        outcomes = self._read("outcomes.csv")
        if observations is None:
            raise FileNotFoundError(f"observations.csv not found in {self.directory}")
        observations = coerce_observations(observations)
        report_obs = validate_observations(observations)
        logger.info(report_obs.summary())

        notes_df = coerce_notes(notes) if notes is not None else notes
        if notes_df is not None:
            logger.info(validate_notes(notes_df).summary())

        outcomes_df = coerce_outcomes(outcomes) if outcomes is not None else None
        if outcomes_df is not None:
            logger.info(validate_outcomes(outcomes_df, observations).summary())
        return observations, notes_df, outcomes_df, patients_from_observations(observations, notes_df)


def patients_from_observations(
    observations: pd.DataFrame, notes: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Derive a minimal patients table (id level only) when none is provided.

    Demographics are not present in this fallback table; static features will
    be treated as missing by the pipeline.  Use a full patients.csv when
    available.
    """
    ids = observations[["patient_id"]].drop_duplicates().reset_index(drop=True)
    ids["age"] = float("nan")
    ids["sex"] = float("nan")
    ids["weight"] = float("nan")
    ids["is_synthetic"] = False
    return ids
