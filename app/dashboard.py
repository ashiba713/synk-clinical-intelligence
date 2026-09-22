"""SYNK research dashboard (Streamlit).

Run:  streamlit run app/dashboard.py
      python -m scripts.cli dashboard

Sections: Overview · Patients · Patient Analysis · Model Performance ·
Experiments · Explainability · Reports · System Status · About
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from theme import (
    BG,
    CARD,
    DANGER,
    DISCLAIMER,
    PRIMARY,
    RISK_COLORS,
    SUCCESS,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
    apply_theme,
    disclaimer_banner,
    metric_card,
    plotly_layout,
    risk_pill,
    section_header,
    status_dot,
)

apply_theme()

import app.data_service as ds  # noqa: E402  (after theme for fast failure mode)

PAGES = [
    "Overview",
    "Patients",
    "Patient Analysis",
    "Model Performance",
    "Experiments",
    "Explainability",
    "Reports",
    "System Status",
    "About",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.4rem 0 1.2rem 0;">
              <div style="font-size:1.7rem; font-weight:700; letter-spacing:0.1em;">SYNK</div>
              <div style="color:{TEXT_SECONDARY}; font-size:0.82rem; margin-top:0.15rem;">
                Clinical Intelligence Before Critical Moments.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", PAGES, label_opacity=0)
        st.divider()

        try:
            info = ds.model_info()
            eid = info.get("engine", "?")
            ver = info.get("model_version", "?")
            st.markdown(
                f'<div class="synk-status">{status_dot(True, f"Engine: {eid}")}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Model version: `{ver}`")
            st.caption(f"Dataset: `{ds._get_config().data.dataset_id}` (synthetic)")
        except Exception as exc:  # noqa: BLE001
            st.markdown(status_dot(False, "Backend initialising"), unsafe_allow_html=True)
            st.caption(str(exc)[:120])

        st.divider()
        st.markdown(f'<div style="color:{TEXT_SECONDARY}; font-size:0.75rem;">⚠ {DISCLAIMER}</div>',
                    unsafe_allow_html=True)
    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_landing() -> None:
    st.markdown(
        f"""
        <div class="synk-hero">
          <h1>SYNK</h1>
          <div class="synk-tagline">Clinical Intelligence Before Critical Moments.</div>
          <div class="synk-sub">
            An explainable multimodal AI system for early sepsis risk stratification
            using physiological time-series and clinical documentation.
          </div>
          <div>{status_dot(True, "Research environment operational · synthetic data")}</div>
          <div class="synk-disclaimer">⚠ {DISCLAIMER}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        if st.button("Open Clinical Intelligence →", use_container_width=True):
            st.session_state["nav_override"] = "Overview"
            st.rerun()
    st.markdown("")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.markdown(metric_card("Modalities", "2", "physiology + clinical text"))
    with l2:
        st.markdown(metric_card("Fusion", "gated / cross-attn", "configurable"))
    with l3:
        st.markdown(metric_card("Explainability", "IG + evidence", "per prediction"))
    with l4:
        st.markdown(metric_card("Uncertainty", "MC dropout", "calibrated"))

    st.markdown("")
    st.markdown(
        f'<div class="synk-note">🧪 <b>Research context.</b> SYNK investigates whether multimodal fusion of ICU '
        f'physiological time-series and clinical text can improve early sepsis risk stratification versus unimodal '
        f'approaches. The hypothesis is under active study; this dashboard reports measured results only.</div>',
        unsafe_allow_html=True,
    )


def page_overview() -> None:
    section_header("Overview", "Cohort status and model snapshot (synthetic demonstration data)")
    try:
        preds = ds.predictions_frame()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Backend unavailable: {exc}")
        return
    preds_sorted = preds.sort_values(["encounter_id", "t"]).copy()
    preds_sorted["prev_prob"] = preds_sorted.groupby("encounter_id")["risk_probability"].shift(1)
    latest = preds_sorted.groupby("encounter_id", as_index=False).tail(1).copy()
    counts = latest["risk_category"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("Patients monitored", f"{latest['patient_id'].nunique():,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("High risk", f"{int(counts.get('HIGH', 0) + counts.get('CRITICAL', 0))}",
                                "HIGH + CRITICAL", color=WARNING), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Moderate risk", f"{int(counts.get('MODERATE', 0))}", color=WARNING), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Low risk", f"{int(counts.get('LOW', 0))}", color=SUCCESS), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("Mean risk", f"{latest['risk_probability'].mean():.3f}"), unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Risk distribution")
        import plotly.graph_objects as go

        order = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        vals = [int(counts.get(k, 0)) for k in order]
        colors = [RISK_COLORS[k] for k in order]
        fig = go.Figure(go.Bar(x=order, y=vals, marker_color=colors, hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_layout(**plotly_layout(), height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Recent alerts")
        alerts = latest[latest["risk_category"].isin(["HIGH", "CRITICAL"])].sort_values("t", ascending=False).head(8)
        if alerts.empty:
            st.caption("No HIGH/CRITICAL encounters in the current cohort.")
        for _, r in alerts.iterrows():
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:0.45rem 0; '
                f'border-bottom:1px solid {BORDER};">'
                f'<span style="font-family:JetBrains Mono, monospace;">{r["encounter_id"]}</span>'
                f'{risk_pill(r["risk_category"])} <span style="font-family:JetBrains Mono, monospace; '
                f'color:{TEXT_SECONDARY};">{r["risk_probability"]:.2f}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Model snapshot")
    info = ds.model_info()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Engine", info.get("engine", "?")), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Version", str(info.get("model_version", "?"))), unsafe_allow_html=True)
    mets = info.get("metrics") or {}
    auroc = mets.get("auroc")
    with m3:
        st.markdown(metric_card("Test AUROC", f"{auroc:.3f}" if isinstance(auroc, (int, float)) else "—"),
                    unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Calibration", str(info.get("calibration", "—"))), unsafe_allow_html=True)
    st.caption("Metrics are computed on the held-out test split of the configured dataset; on synthetic data they "
               "characterise the pipeline, not clinical performance.")


BORDER = "rgba(255,255,255,0.06)"


def page_patients() -> None:
    section_header("Patients", "Search, filter and sort the monitored cohort")
    preds = ds.predictions_frame()
    preds_sorted = preds.sort_values(["encounter_id", "t"]).copy()
    preds_sorted["prev_prob"] = preds_sorted.groupby("encounter_id")["risk_probability"].shift(1)
    latest = preds_sorted.groupby("encounter_id", as_index=False).tail(1).copy()
    latest["risk_change"] = latest["risk_probability"] - latest["prev_prob"]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search", placeholder="Patient or encounter id…", label_visibility="collapsed")
    with c2:
        cat = st.selectbox("Risk filter", ["ALL", "LOW", "MODERATE", "HIGH", "CRITICAL"], label_visibility="collapsed")
    with c3:
        sort_by = st.selectbox("Sort", ["risk ↓", "risk ↑", "change ↓"], label_visibility="collapsed")

    view = latest
    if q:
        ql = q.lower()
        view = view[view["patient_id"].str.lower().str.contains(ql) | view["encounter_id"].str.lower().str.contains(ql)]
    if cat != "ALL":
        view = view[view["risk_category"] == cat]
    if sort_by == "risk ↓":
        view = view.sort_values("risk_probability", ascending=False)
    elif sort_by == "risk ↑":
        view = view.sort_values("risk_probability")
    else:
        view = view.sort_values("risk_change", ascending=False, na_position="last")

    patients_meta = ds.load_frames()[0].set_index("patient_id")
    rows = []
    for _, r in view.head(80).iterrows():
        age = patients_meta.loc[r["patient_id"], "age"] if r["patient_id"] in patients_meta.index else np.nan
        rows.append({
            "Patient": r["patient_id"],
            "Encounter": r["encounter_id"],
            "Age": f"{age:.0f}" if age == age else "—",
            "Risk": r["risk_probability"],
            "Category": r["risk_category"],
            "Δ Risk": r["risk_change"],
            "Horizon (h)": r["prediction_horizon_hours"],
            "Last update": str(r["t"])[:16],
        })
    table = pd.DataFrame(rows)

    st.dataframe(
        table.style.format({"Risk": "{:.3f}", "Δ Risk": "{:+.3f}"})
        .background_gradient(subset=["Risk"], cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            "Encounter": st.column_config.TextColumn("Encounter"),
        },
    )
    st.caption("Click a row's encounter id, then open **Patient Analysis** and select it there.")

    first_enc = view.iloc[0]["encounter_id"] if len(view) else None
    if first_enc and st.button(f"Analyse {first_enc} →"):
        st.session_state["selected_encounter"] = first_enc
        st.session_state["nav_override"] = "Patient Analysis"
        st.rerun()


def page_patient_analysis() -> None:
    section_header("Patient Analysis", "Full multimodal evidence for one encounter")
    preds = ds.predictions_frame()
    encounters = sorted(preds["encounter_id"].unique())
    default = st.session_state.get("selected_encounter", encounters[0] if encounters else None)
    enc_id = st.selectbox("Encounter", encounters, index=encounters.index(default) if default in encounters else 0)

    if not enc_id:
        st.info("No encounters available.")
        return

    enc_preds = preds[preds["encounter_id"] == enc_id].sort_values("t")
    latest = enc_preds.tail(1).iloc[0]
    patients_meta, observations, notes, outcomes = ds.load_frames()
    obs = observations[observations["encounter_id"] == enc_id].sort_values("timestamp")
    enc_notes = notes[notes["encounter_id"] == enc_id].sort_values("timestamp") if notes is not None else pd.DataFrame()
    outcome = outcomes[outcomes["encounter_id"] == enc_id]

    # ---- header ----------------------------------------------------------
    h1, h2, h3, h4 = st.columns([2, 1.2, 1.2, 1.4])
    with h1:
        st.markdown(f"#### `{enc_id}`")
        st.caption(f"Patient `{latest['patient_id']}` · horizon {latest['prediction_horizon_hours']:.0f} h · "
                   f"updated {str(latest['t'])[:16]}")
    with h2:
        st.markdown(metric_card("Current risk", f"{latest['risk_probability']:.2f}"), unsafe_allow_html=True)
    with h3:
        st.markdown(risk_pill(latest["risk_category"]), unsafe_allow_html=True)
    with h4:
        st.caption(f"Engine `{latest['engine']}` · version `{latest['model_version']}`")

    st.markdown("")

    # ---- risk over time ----------------------------------------------------
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### Risk trajectory")
        import plotly.graph_objects as go

        layout = plotly_layout()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(enc_preds["t"]), y=enc_preds["risk_probability"],
            mode="lines+markers", line={{"color": PRIMARY, "width": 2}}, name="Risk",
            hovertemplate="%{x|%m-%d %H:%M} · %{y:.3f}<extra></extra>",
        ))
        onset = outcome["sepsis_onset_timestamp"].iloc[0] if len(outcome) else None
        if onset is not None and pd.notna(onset):
            fig.add_vline(x=pd.Timestamp(onset), line={"color": DANGER, "dash": "dot", "width": 1},
                          annotation_text="sepsis onset (research label)", annotation_position="top")
        for thr, name in zip(ds._get_config().evaluation.risk_thresholds, ["moderate", "high", "critical"]):
            fig.add_hline(y=thr, line={"color": BORDER, "width": 1}, opacity=0.6)
        fig.update_layout(**layout, height=300, yaxis={"range": [0, 1.02], **layout["yaxis"]})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("##### Why SYNK flags this patient")
        try:
            expl = ds.explain(enc_id)
            vars_ = expl.get("structured_attribution", {}).get("variables", [])[:4]
            if vars_:
                for v in vars_:
                    st.markdown(
                        f'<div class="synk-note" style="margin-bottom:0.4rem;">📈 <b>'
                        f'{v["name"].replace("_", " ").title()}</b> — attribution {v["score"]:.3f}</div>',
                        unsafe_allow_html=True,
                    )
            te = expl.get("text_evidence") or []
            phrases = [s for n in te for s in n.get("evidence", {}).get("spans", []) if not s.get("negated")][:3]
            for p in phrases:
                st.markdown(f'<div class="synk-note" style="margin-bottom:0.4rem;">📝 note evidence: '
                            f'“{p["text"]}” <span style="color:{TEXT_SECONDARY}">({p["category"]})</span></div>',
                            unsafe_allow_html=True)
            if not vars_ and not phrases:
                st.caption("No dominant evidence for this window.")
            unc = expl.get("uncertainty") or {}
            if unc.get("available"):
                st.caption(f"MC-dropout uncertainty: σ={unc['std_deviation']:.3f}, "
                           f"95% CI [{unc['ci95_low']:.3f}, {unc['ci95_high']:.3f}]")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Explanation unavailable: {exc}")

    # ---- physiological charts ----------------------------------------------
    st.markdown("##### Physiological signals")
    from synk.data.schemas import VITAL_VARIABLES, LAB_VARIABLES

    display_vars = st.multiselect(
        "Variables",
        [v for v in VITAL_VARIABLES + LAB_VARIABLES if v in obs.columns],
        default=[v for v in ("heart_rate", "respiratory_rate", "mean_arterial_pressure",
                             "temperature", "spo2", "lactate") if v in obs.columns],
    )
    if display_vars:
        import plotly.graph_objects as go

        n = len(display_vars)
        fig = go.Figure()
        palette = [PRIMARY, SUCCESS, WARNING, "#A78BFA", "#F97316", "#22D3EE", "#F472B6", "#94A3B8"]
        for i, var in enumerate(display_vars):
            s = obs[["timestamp", var]].dropna()
            fig.add_trace(go.Scatter(x=s["timestamp"], y=s[var], mode="lines", name=var,
                                     line={{"color": palette[i % len(palette)], "width": 1.8}}))
        if len(outcome) and pd.notna(outcome["sepsis_onset_timestamp"].iloc[0]):
            fig.add_vline(x=pd.Timestamp(outcome["sepsis_onset_timestamp"].iloc[0]),
                          line={"color": DANGER, "dash": "dot", "width": 1})
        fig.update_layout(**plotly_layout(), height=380)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Reference bands are intentionally omitted: ranges shown anywhere in SYNK are configurable defaults, "
               "not validated alert thresholds. Missing values are gaps, never zeros.")

    # ---- clinical notes -----------------------------------------------------
    st.markdown("##### Clinical documentation")
    if len(enc_notes):
        from synk.explainability.text_attribution import extract_text_evidence

        for _, note in enc_notes.tail(8).iloc[::-1].iterrows():
            ev = extract_text_evidence(str(note["note_text"]))
            chips = "".join(
                f'<span class="evidence-chip{" negated" if s["negated"] else ""}">{s["text"]}</span>'
                for s in ev["spans"][:6]
            )
            st.markdown(
                f'<div class="synk-card" style="margin-bottom:0.6rem;">'
                f'<div style="display:flex; justify-content:space-between;">'
                f'<span style="font-family:JetBrains Mono, monospace; font-size:0.78rem; color:{TEXT_SECONDARY};">'
                f'{str(note["timestamp"])[:16]} · {note.get("note_type", "")}</span><span>{chips}</span></div>'
                f'<div style="margin-top:0.45rem;">{note["note_text"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Struck-through chips are inside a negation scope (e.g. “no evidence of infection”) and are "
                   "not counted as positive evidence.")
    else:
        st.caption("No notes for this encounter (text modality absent — the model handles this transparently).")

    # ---- timeline -----------------------------------------------------------
    st.markdown("##### Timeline")
    timeline_df = pd.DataFrame({
        "time": pd.to_datetime(obs["timestamp"]),
        "risk": [np.nan] * len(obs),
    })
    risk_pts = enc_preds[["t", "risk_probability"]].rename(columns={"t": "time", "risk_probability": "risk"})
    merged = pd.concat([timeline_df, risk_pts.assign(time=pd.to_datetime(risk_pts["time"]))]).sort_values("time")
    merged["event"] = np.where(merged["risk"].notna(), "risk score", "")
    note_times = set(pd.to_datetime(enc_notes["timestamp"]).dt.floor("h")) if len(enc_notes) else set()
    merged["note"] = ["" if ts not in note_times else "note" for ts in merged["time"]]
    st.dataframe(merged.rename(columns={"time": "Time (h)", "risk": "Risk", "event": "Event", "note": "Note"}),
                 use_container_width=True, height=220, hide_index=True)
    st.caption("Hours are relative to stay start; the risk trajectory and note events are aligned on the same axis.")


def page_model_performance() -> None:
    section_header("Model Performance", "Measured metrics on the held-out test split")
    report = ds.latest_eval_report()
    if report is None:
        st.info("No evaluation report found yet. Run: `python -m scripts.cli train` "
                "(or `--skip-training` for baselines only).")
        return

    auroc, auprc = report["auroc"], report["auprc"]
    tm = report["threshold_metrics"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(metric_card("AUROC", f'{auroc["value"]:.3f}',
                                f'CI {auroc["ci_low"]:.3f}–{auroc["ci_high"]:.3f}'), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("AUPRC", f'{auprc["value"]:.3f}',
                                f'CI {auprc["ci_low"]:.3f}–{auprc["ci_high"]:.3f}'), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("F1", f'{tm["f1"]:.3f}'), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Sensitivity", f'{tm["sensitivity"]:.3f}'), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("Specificity", f'{tm["specificity"]:.3f}'), unsafe_allow_html=True)
    with c6:
        st.markdown(metric_card("Brier", f'{report["brier_score"]:.3f}'), unsafe_allow_html=True)

    left, mid, right = st.columns(3)
    import plotly.graph_objects as go

    with left:
        st.markdown("###### ROC")
        roc = report["roc_curve"]
        fpr = [p["fpr"] for p in roc]
        tpr = [p["tpr"] for p in roc]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line={{"color": PRIMARY}}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line={{"color": BORDER, "dash": "dash"}}, showlegend=False))
        fig.update_layout(**plotly_layout(), height=280, title=f"AUROC {auroc['value']:.3f}")
        st.plotly_chart(fig, use_container_width=True)
    with mid:
        st.markdown("###### Precision–Recall")
        pr = report["pr_curve"]
        fig = go.Figure(go.Scatter(x=[p["recall"] for p in pr], y=[p["precision"] for p in pr],
                                   mode="lines", line={{"color": SUCCESS}}))
        fig.update_layout(**plotly_layout(), height=280, title=f"AUPRC {auprc['value']:.3f}")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("###### Calibration")
        cal = report["calibration_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[p["mean_predicted"] for p in cal], y=[p["observed_frequency"] for p in cal],
                                 mode="lines+markers", line={{"color": WARNING}}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line={{"color": BORDER, "dash": "dash"}}, showlegend=False))
        fig.update_layout(**plotly_layout(), height=280, title=f"ECE {report['ece']:.3f}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("###### Confusion matrix & operating points")
    cm = tm["confusion_matrix"]
    cm1, cm2 = st.columns(2)
    with cm1:
        fig = go.Figure(go.Heatmap(
            z=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
            colorscale=[[0, CARD], [1, PRIMARY]], text=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            texttemplate="%{text}", showscale=False,
        ))
        fig.update_layout(**plotly_layout(), height=260)
        st.plotly_chart(fig, use_container_width=True)
    with cm2:
        ops = pd.DataFrame(report["operating_points"])
        st.dataframe(ops, use_container_width=True, hide_index=True, height=260)

    if "early_warning" in report and report["early_warning"]:
        ew = report["early_warning"]
        st.markdown("###### Early-warning metrics (computed, not assumed)")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.markdown(metric_card("Detection rate", f'{ew.get("detection_rate", float("nan")):.2f}'),
                        unsafe_allow_html=True)
        with e2:
            st.markdown(metric_card("Median lead time", f'{ew.get("lead_time_median_hours", float("nan")):.1f} h'),
                        unsafe_allow_html=True)
        with e3:
            st.markdown(metric_card("False alerts / encounter",
                                    f'{ew.get("false_alert_rate_per_encounter", float("nan")):.2f}'),
                        unsafe_allow_html=True)
        with e4:
            st.markdown(metric_card("Septic encounters", f'{ew.get("n_septic_encounters", 0)}'),
                        unsafe_allow_html=True)

    if "ablation" in report and report["ablation"]:
        st.markdown("###### Modality ablation (same multimodal model, one input zeroed)")
        st.caption(report["ablation"]["note"])
        ab = report["ablation"]
        st.dataframe(pd.DataFrame([
            {"configuration": "both active", **{k: ab["both_active"][k] for k in ("auroc", "f1", "sensitivity", "specificity")}},
            {"configuration": "text zeroed", **{k: ab["structured_zeroed"][k] for k in ("auroc", "f1", "sensitivity", "specificity")}},
            {"configuration": "structured zeroed", **{k: ab["text_zeroed"][k] for k in ("auroc", "f1", "sensitivity", "specificity")}},
        ]), use_container_width=True, hide_index=True)


def page_experiments() -> None:
    section_header("Experiments", "Unimodal vs multimodal comparison (append-only registry)")
    exp = ds.experiments_frame()
    if exp.empty:
        st.info("No experiments registered yet. Run `python -m scripts.cli train` to produce the comparison table.")
        return
    cols = [c for c in ["experiment_id", "modality", "model_kind", "dataset_id", "auroc", "auprc", "f1",
                        "sensitivity", "specificity", "brier", "detection_rate", "lead_time_median_hours",
                        "created_at"] if c in exp.columns]
    st.dataframe(exp[cols], use_container_width=True, height=420, hide_index=True)
    st.caption("Every row is produced by an actual training/evaluation run on the same split. Deep SYNK rows and "
               "sklearn baselines are labelled in `model_kind`.")


def page_explainability() -> None:
    section_header("Explainability", "Global and local model evidence")
    enc_id = st.selectbox("Encounter", sorted(ds.predictions_frame()["encounter_id"].unique()))
    try:
        expl = ds.explain(enc_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Explanation unavailable: {exc}")
        return
    if not expl:
        st.info("No explanation available for this encounter.")
        return

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Structured attribution (local)")
        attr = expl.get("structured_attribution", {})
        st.caption(f"Method: `{attr.get('method')}` — gradient-based attribution of the temporal branch.")
        variables = attr.get("variables", [])[:12]
        if variables:
            import plotly.graph_objects as go

            names = [v["name"] for v in variables][::-1]
            scores = [v["score"] for v in variables][::-1]
            fig = go.Figure(go.Bar(y=names, x=scores, orientation="h", marker_color=PRIMARY))
            fig.update_layout(**plotly_layout(), height=420)
            st.plotly_chart(fig, use_container_width=True)
        matrix = attr.get("attribution_matrix")
        if matrix:
            import plotly.graph_objects as go

            arr = np.asarray(matrix)
            fig = go.Figure(go.Heatmap(
                z=arr.T, colorscale="RdBu", zmid=0, showscale=False,
                y=attr.get("variables") and [v["name"] for v in attr["variables"][: arr.shape[1]]],
            ))
            fig.update_layout(**plotly_layout(), height=300, title="Attribution over time × variable")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Text evidence")
        for note_ev in expl.get("text_evidence", [])[:3]:
            ev = note_ev.get("evidence", {})
            st.markdown(f"**Note**: {note_ev['text'][:180]}…" if len(note_ev["text"]) > 180 else f"**Note**: {note_ev['text']}")
            chips = "".join(
                f'<span class="evidence-chip{" negated" if s["negated"] else ""}">{s["text"]}</span>'
                for s in ev.get("spans", [])[:10]
            )
            st.markdown(chips or "*no lexicon matches*", unsafe_allow_html=True)
            st.caption(f"Positive categories: {ev.get('positive_categories', [])}")

        st.markdown("#### Fusion contribution")
        fc = expl.get("fusion_contribution")
        if fc:
            import plotly.graph_objects as go

            fig = go.Figure(go.Pie(
                labels=["structured", "text"], values=[fc["structured"], fc["text"]],
                marker_colors=[PRIMARY, SUCCESS], hole=0.6,
            ))
            fig.update_layout(**plotly_layout(), height=240)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(fc["note"])
        else:
            st.caption("Fusion gates available for the gated multimodal fusion configuration.")

        st.markdown("#### Uncertainty")
        unc = expl.get("uncertainty") or {}
        if unc.get("available"):
            st.markdown(metric_card("σ (MC dropout)", f'{unc["std_deviation"]:.3f}',
                                    f'95% CI [{unc["ci95_low"]:.3f}, {unc["ci95_high"]:.3f}]'), unsafe_allow_html=True)
        else:
            st.caption(unc.get("note", "No uncertainty estimate available."))

    st.markdown(f'> {expl.get("research_disclaimer", DISCLAIMER)}')


def page_reports() -> None:
    section_header("Reports", "Research-oriented patient analysis reports")
    preds = ds.predictions_frame()
    enc_id = st.selectbox("Encounter", sorted(preds["encounter_id"].unique()))
    fmt = st.multiselect("Formats", ["json", "md", "csv"], default=["json", "md"])
    if st.button("Generate report") and enc_id and fmt:
        from synk.reporting.report import build_patient_report, save_report

        service = ds.get_service()
        frame = service.predict_encounters(*_frames(), encounter_ids=[enc_id]).sort_values("t")
        latest = frame.tail(1).iloc[0]
        patients, observations, notes, outcomes = _frames()
        ss = service.featurizer.build(observations, notes, outcomes, patients)[0]
        sub = ss.samples[ss.samples["encounter_id"] == enc_id]
        expl = service.explain_sample(ss, int(sub.index[-1]))
        report = build_patient_report(
            patient_id=str(latest["patient_id"]), encounter_id=enc_id, prediction_row=latest,
            explanation=expl, config=ds._get_config(), observations=observations, notes=notes,
        )
        saved = save_report(report, Path(ds._get_config().paths.report_dir), formats=tuple(fmt))
        st.success(f"Report `{report['report_id']}` generated")
        if "md" in saved:
            st.download_button("Download Markdown", Path(saved["md"]).read_text(encoding="utf-8"),
                               file_name=f"{report['report_id']}.md")
        if "json" in saved:
            st.download_button("Download JSON", Path(saved["json"]).read_text(encoding="utf-8"),
                               file_name=f"{report['report_id']}.json")
        st.markdown(Path(saved["md"]).read_text(encoding="utf-8") if "md" in saved else
                    f"```json\n{report['risk_assessment']}\n```")


def _frames():
    return ds.load_frames()


def page_status() -> None:
    section_header("System Status", "Environment, engine and artifact state")
    import torch

    from synk.utils.torch_utils import device_summary

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Runtime")
        st.markdown(metric_card("Device", device_summary()), unsafe_allow_html=True)
        info = ds.model_info()
        st.markdown(metric_card("Engine", str(info.get("engine"))), unsafe_allow_html=True)
        st.markdown(metric_card("Model version", str(info.get("model_version"))), unsafe_allow_html=True)
    with c2:
        st.markdown("#### Artifacts")
        cfg = ds._get_config()
        for label, p in [
            ("Synthetic data", Path(cfg.synthetic.output_dir)),
            ("Processed samples", Path(cfg.data.processed_dir) / "samples"),
            ("Checkpoints", Path(cfg.paths.checkpoint_dir)),
            ("Model registry", Path(cfg.paths.model_registry_dir)),
            ("Evaluation", Path(cfg.paths.evaluation_dir)),
        ]:
            exists = Path(p).exists() and any(Path(p).iterdir())
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:0.3rem 0;">'
                f'<span>{label}</span><span style="color:{SUCCESS if exists else TEXT_SECONDARY};">'
                f'{"present" if exists else "not generated"}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.markdown(f'<div class="synk-note">🧪 All serving data is synthetic. {DISCLAIMER}</div>', unsafe_allow_html=True)


def page_about() -> None:
    section_header("About SYNK")
    st.markdown(
        """
**SYNK** is an explainable multimodal AI platform for studying early sepsis risk stratification
from ICU physiological time-series and clinical documentation.

**Research question.** Can multimodal fusion of ICU physiological time-series data and clinical
text improve early sepsis risk stratification compared with unimodal approaches?

**Hypothesis (under study, not proven).** A multimodal model that combines temporal physiological
information with contextual clinical language may capture complementary evidence and provide
improved early risk stratification compared with either modality alone.

**What SYNK does**
- Encodes structured ICU sequences with a TFT-style temporal encoder (variable selection, attention)
- Encodes clinical notes with a configurable HF transformer or an offline TF-IDF backend
- Fuses both representations (gated / concat / cross-attention, configurable)
- Reports AUROC/AUPRC, calibration, early-warning metrics, modality ablation
- Explains every prediction via gradient attribution + deterministic text evidence + MC-dropout uncertainty

**What SYNK is not**
- Not a medical device; not for clinical diagnosis or treatment
- Not clinically validated; thresholds are study parameters
- Not a chatbot and not an LLM wrapper; the models are trained supervised predictors
        """
    )
    st.markdown(f"**Version** `{__import__('synk').__version__}` · "
                f"engine `{ds.engine_id()}`")
    disclaimer_banner()


# ---------------------------------------------------------------------------
def main() -> None:
    page = st.session_state.pop("nav_override", None) or sidebar()

    pages = {
        "Overview": page_overview,
        "Patients": page_patients,
        "Patient Analysis": page_patient_analysis,
        "Model Performance": page_model_performance,
        "Experiments": page_experiments,
        "Explainability": page_explainability,
        "Reports": page_reports,
        "System Status": page_status,
        "About": page_about,
    }
    # The landing hero is shown for first-time users via Overview.
    if page == "Overview" and not st.session_state.get("seen_overview"):
        st.session_state["seen_overview"] = True

    pages.get(page, page_overview)()
    st.markdown("")
    st.markdown(
        f'<div style="text-align:center; color:{TEXT_SECONDARY}; font-size:0.75rem; padding:1rem 0;">'
        f'SYNK · {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
else:
    # streamlit run executes this file as a script (no __main__).
    main()
