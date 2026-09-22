"""Data layer tests: synthetic generator, validation, preprocessing, labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synk.data.synthetic import generate_cohort
from synk.data.validation import validate_observations
from synk.labels.sepsis_labels import build_label_table
from synk.preprocessing.vitals import VitalsPreprocessor


def test_synthetic_cohort_shapes(cohort):
    assert len(cohort.observations) > 0
    assert len(cohort.outcomes) > 0
    assert cohort.manifest["is_synthetic"] is True
    assert "not real patient data" in cohort.manifest["banner"]
    # patients table carries demographics for the static branch
    assert {"patient_id", "age", "sex", "weight"} <= set(cohort.patients.columns)
    # prevalence must match the outcome table, not be invented
    prev = cohort.outcomes["sepsis_onset_timestamp"].notna().mean()
    assert cohort.manifest["observed_prevalence"] == pytest.approx(prev)


def test_synthetic_notes_have_timestamps(cohort):
    if len(cohort.notes):
        assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(cohort.notes["timestamp"]))
        assert (cohort.notes["note_text"].str.len() > 0).all()


def test_validation_catches_bad_data(config, cohort):
    report = validate_observations(cohort.observations, raise_on_error=False)
    assert report.ok, report.errors
    broken = cohort.observations.copy()
    broken.loc[broken.index[0], "heart_rate"] = 9999.0
    report2 = validate_observations(broken, raise_on_error=False)
    assert not report2.ok or report2.warnings  # flagged somehow
    missing = cohort.observations.drop(columns=["encounter_id"])
    with pytest.raises(Exception):
        validate_observations(missing)


def test_preprocessor_fit_transform_roundtrip(config, cohort):
    pre = VitalsPreprocessor(config).fit(cohort.observations)
    cleaned = pre.transform(cohort.observations)
    assert "__obs" in cleaned.columns[3] or any(c.endswith("__obs") for c in cleaned.columns)
    # standardised columns have ~zero mean on the fitted data
    hr = cleaned["heart_rate"].dropna()
    assert abs(hr.mean()) < 0.5
    # MAP derived where missing
    if "mean_arterial_pressure" in cleaned.columns:
        assert cleaned["mean_arterial_pressure"].notna().mean() > 0.5


def test_preprocessor_survives_serialisation(config, cohort, tmp_path):
    pre = VitalsPreprocessor(config).fit(cohort.observations)
    path = tmp_path / "pre.json"
    pre.save(path)
    pre2 = VitalsPreprocessor.load(path, config)
    a = pre.transform(cohort.observations.head(200))
    b = pre2.transform(cohort.observations.head(200))
    pd.testing.assert_frame_equal(a, b)


def test_label_engine_labels_and_exclusions(config, cohort):
    table = build_label_table(cohort.observations, cohort.outcomes, config)
    assert len(table.samples) > 0
    assert set(table.samples["label"].unique()) <= {0, 1}
    # no prediction times at/after onset when exclusion is on
    if config.labels.exclude_after_onset:
        onset = table.encounters.set_index("encounter_id")["sepsis_onset_timestamp"]
        merged = table.samples.merge(
            onset.rename("onset"), left_on="encounter_id", right_index=True, how="left"
        )
        viol = merged[merged["onset"].notna() & (merged["t"] >= merged["onset"])]
        assert len(viol) == 0
    # horizon consistency: positives must have onset within (t, t+h]
    horizon = pd.Timedelta(hours=config.labels.horizon_hours)
    pos = table.samples.merge(
        table.encounters[["encounter_id", "sepsis_onset_timestamp"]], on="encounter_id"
    )
    pos = pos[pos["label"] == 1]
    assert ((pos["sepsis_onset_timestamp"] > pos["t"]) &
            (pos["sepsis_onset_timestamp"] <= pos["t"] + horizon)).all()
