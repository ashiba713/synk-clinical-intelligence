#!/usr/bin/env python
"""SYNK CLI.

Subcommands:
    generate-data   Generate the synthetic demo cohort
    train           Train structured / text / multimodal models (full pipeline)
    evaluate        Re-evaluate a registered bundle on the stored test split
    predict         Run inference for one encounter and print the payload
    serve           Start the FastAPI backend
    dashboard       Start the Streamlit UI

Examples:
    python -m scripts.cli generate-data --n-patients 60
    python -m scripts.cli train --set training.max_epochs=2
    python -m scripts.cli predict --encounter SYN-E-00001 --explain
    python -m scripts.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from synk.config.settings import load_config, merge_configs, parse_dotlist

logger = logging.getLogger("synk.cli")


def _cfg(args) -> "object":
    return load_config(override_paths=[Path(args.config)] if getattr(args, "config", None) else None)


# ---------------------------------------------------------------------------
# generate-data
# ---------------------------------------------------------------------------
def cmd_generate_data(args) -> int:
    from synk.data.synthetic import generate_cohort
    from synk.utils.common import ensure_dir

    cfg = _cfg(args)
    if args.set:
        cfg = merge_configs(cfg, parse_dotlist(args.set))
    if args.n_patients:
        cfg = merge_configs(cfg, {"synthetic": {"n_patients": args.n_patients}})
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    cohort = generate_cohort(cfg, seed=args.seed or cfg.seed)
    out = Path(args.output) if args.output else Path(cfg.synthetic.output_dir)
    cohort.to_disk(out)
    print(json.dumps({k: v for k, v in cohort.manifest.items() if k != "config"}, indent=2))
    print(f"\n[SYNTHETIC DATA] written to {out.resolve()} - not real patient data")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args) -> int:
    from synk.pipeline import run_pipeline

    logging.basicConfig(level=logging.INFO)
    results = run_pipeline(
        config_path=Path(args.config) if args.config else None,
        overrides=args.set or [],
        skip_training=args.skip_training,
    )
    print(json.dumps(results, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args) -> int:
    import pandas as pd

    from synk.config.settings import load_config
    from synk.evaluation.evaluator import evaluate_model, save_report
    from synk.features.dataset import SampleSet
    from synk.inference.engine import load_bundle
    from synk.models.multimodal import load_trained_model
    from synk.utils.torch_utils import get_device

    cfg = load_config(override_paths=[Path(args.config)] if args.config else None)
    bundle = load_bundle(Path(args.model_dir), cfg)
    if bundle is None:
        print("No bundle found at", args.model_dir, file=sys.stderr)
        return 1
    sample_set = SampleSet.load(Path(cfg.data.processed_dir) / "samples")
    outcomes = pd.read_csv(Path(cfg.synthetic.output_dir) / "outcomes.csv")
    model, _payload = load_trained_model(Path(args.model_dir) / "model.pt", bundle.config)
    splits = json.loads((Path(cfg.data.processed_dir) / "samples" / "splits.json").read_text())
    enc_ids = splits.get(args.split, splits["test"])

    report = evaluate_model(
        model, sample_set, enc_ids, bundle.config, outcomes,
        calibrator=bundle.calibrator, include_ablation=args.ablation,
        device=get_device(),
    )
    out = Path(args.output) if args.output else Path(cfg.paths.evaluation_dir) / "reeval.json"
    save_report(report, out)
    print(f"Evaluation report written to {out}")
    print(json.dumps({k: report[k] for k in ("auroc", "auprc", "brier_score")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def cmd_predict(args) -> int:
    import pandas as pd

    from synk.data.synthetic import generate_cohort
    from synk.inference.engine import InferenceService

    cfg = _cfg(args)
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))

    syn_dir = Path(cfg.synthetic.output_dir)
    if not (syn_dir / "observations.csv").exists():
        logger.info("Synthetic data missing; generating first")
        generate_cohort(cfg, seed=cfg.seed).to_disk(syn_dir)
    patients = pd.read_csv(syn_dir / "patients.csv")
    observations = pd.read_csv(syn_dir / "observations.csv")
    notes = pd.read_csv(syn_dir / "notes.csv")
    outcomes = pd.read_csv(syn_dir / "outcomes.csv")

    service = InferenceService(cfg)
    engine_id = service.ensure_ready(observations, notes, outcomes, patients)
    logger.info("Engine: %s", engine_id)

    frame = service.predict_encounters(observations, notes, outcomes, patients,
                                       encounter_ids=[args.encounter] if args.encounter else None)
    if frame.empty:
        print("No predictions produced for the requested filter.", file=sys.stderr)
        return 1
    last = frame.sort_values("t").iloc[-1]

    sample_set, _ = service.featurizer.build(observations, notes, outcomes, patients)
    idx = int(sample_set.samples[(sample_set.samples["encounter_id"] == last["encounter_id"])].index[-1]) \
        if args.explain else -1
    payload = {
        "engine": service.engine_id,
        "prediction": {
            "patient_id": str(last["patient_id"]),
            "encounter_id": str(last["encounter_id"]),
            "prediction_time": str(last["t"]),
            "risk_probability": float(last["risk_probability"]),
            "risk_category": str(last["risk_category"]),
            "prediction_horizon_hours": float(last["prediction_horizon_hours"]),
            "model_version": str(last["model_version"]),
        },
    }
    if args.explain and idx >= 0:
        payload["explanation"] = service.explain_sample(sample_set, idx)
    payload["research_disclaimer"] = (
        "Research prototype - not for clinical diagnosis or treatment."
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
def cmd_serve(args) -> int:
    import uvicorn

    from synk.api.main import create_app
    from synk.config.settings import load_config

    cfg = load_config(override_paths=[Path(args.config)] if args.config else None)
    app = create_app(cfg)
    uvicorn.run(app, host=args.host or cfg.api.host, port=args.port or cfg.api.port, log_level="info")
    return 0


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
def cmd_dashboard(args) -> int:
    import subprocess
    import sys as _sys

    app_path = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"
    cmd = [_sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.port", str(args.port), "--browser.gatherUsageStats", "false"]
    if args.headless:
        cmd.append("--server.headless")
        cmd.append("true")
    logger.info("Starting Streamlit dashboard: %s", " ".join(cmd))
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synk", description="SYNK research CLI")
    parser.add_argument("--config", help="Path to a YAML config override")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_set_option(p: argparse.ArgumentParser) -> None:
        p.add_argument("--set", nargs="*", default=None,
                       help="Dot-list config overrides, e.g. --set training.lr=0.001 model.modality=both")

    p = sub.add_parser("generate-data", help="Generate the synthetic demo cohort")
    p.add_argument("--n-patients", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    _add_set_option(p)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("train", help="Run the full training + evaluation pipeline")
    p.add_argument("--skip-training", action="store_true", help="Baselines only (fast smoke mode)")
    _add_set_option(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Re-evaluate a registered model bundle")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ablation", action="store_true", help="Include modality ablation")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="Predict for one encounter (or the latest)")
    p.add_argument("--encounter", default=None)
    p.add_argument("--explain", action="store_true", help="Include the explanation payload")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("serve", help="Start the FastAPI backend")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("dashboard", help="Start the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--headless", action="store_true")
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
