"""MIMIC-IV extract adapter.

SYNK does NOT download or redistribute MIMIC-IV.  Researchers must:
 1. complete PhysioNet credentialing and the CITI training,
 2. obtain the MIMIC-IV module files locally,
 3. export a *tabular extract* (the adapter consumes flat CSVs, not the raw
    Hive/Postgres dumps), placing ``observations.csv``, ``notes.csv`` and
    ``outcomes.csv`` under a private local directory (e.g. ``data/raw/mimic``).

The expected extract is produced by the user's own SQL; the queries below are
guidance committed as documentation strings so the mapping is transparent.
"""

from __future__ import annotations

from pathlib import Path

from synk.data.adapters.csv_adapter import CsvAdapter
from synk.utils.logging import get_logger

logger = get_logger(__name__)

OBSERVATIONS_SQL_GUIDANCE = """
-- Reference shape for observations.csv (hourly wide format):
--   patient_id  = subjects.subject_id
--   encounter_id= hadm_id (or stay_id for icu module)
--   timestamp   = charttime floored to the hour
--   heart_rate  = d_items itemid for HR from chartevents
--   ... map each canonical column with your local itemid dictionary.
--   antibiotics_given/culture_drawn are hourly 0/1 flags derived from
--   prescriptions/pharmacy and microbiologyevents respectively.
"""

NOTES_SQL_GUIDANCE = """
-- Reference shape for notes.csv:
--   patient_id, encounter_id (hadm_id), timestamp (charttime),
--   note_type (e.g. 'nursing_note', 'radiology', 'discharge'),
--   author_type, note_text.
-- Respect the source dataset's data use agreement: never redistribute notes.
"""

OUTCOMES_SQL_GUIDANCE = """
-- Reference shape for outcomes.csv:
--   patient_id, encounter_id, admission_timestamp, discharge_timestamp,
--   sepsis_onset_timestamp (NULL for non-septic stays).
-- The onset definition is YOUR study design decision (e.g. Sepsis-3):
--   suspected_infection time = min(antibiotic, culture) with 24h window,
--   plus SOFA >= 2 increase.  SYNK's label engine can also derive labels
--   from criteria (labels.source=criteria); configure it explicitly.
"""


class MimicIVAdapter(CsvAdapter):
    """CSV adapter pre-configured for a user-produced MIMIC-IV extract."""

    def __init__(self, directory: Path, **kwargs):
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(
                f"MIMIC-IV extract directory not found: {directory}. "
                "SYNK never downloads restricted datasets - export the extract yourself "
                f"per docs/deployment.md.\n{OBSERVATIONS_SQL_GUIDANCE}"
            )
        logger.info("Loading user-provided MIMIC-IV extract from %s (authorized use assumed)", directory)
        logger.info("Notes guidance: %s", " ".join(NOTES_SQL_GUIDANCE.split()))
        super().__init__(directory=directory, **kwargs)
