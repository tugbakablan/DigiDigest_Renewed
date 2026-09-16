from __future__ import annotations

import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from digest_and_rank import annotate_and_rank, build_ranked_candidates, staged_digest
from explainability import classifier_explanation_card, explain_classifier, reference_feature_vector
from history_store import delete_analysis, list_analyses, load_analysis, save_analysis
from train_baselines import feature_matrix
from ui_helpers import (
    format_ace_probability,
    format_ace_status,
    normalize_protein_sequence,
    split_single_fasta,
)


ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets" / "digidigest_symbol.png"
EXAMPLE_FASTA = ROOT / "examples" / "P02863_alpha_beta_gliadin.fa"
HISTORY_DB = ROOT / "user_data" / "analysis_history.sqlite3"
CLASSIFIER_BUNDLE = joblib.load(ROOT / "models" / "ace_classifier.joblib")
REGRESSOR_BUNDLE = joblib.load(ROOT / "models" / "pic50_regressor.joblib")
CLASSIFIER_MIN = int(CLASSIFIER_BUNDLE["minimum_length"])
CLASSIFIER_MAX = int(CLASSIFIER_BUNDLE["maximum_length"])
CLASSIFIER_REFERENCE = reference_feature_vector(pd.read_csv(ROOT / "output" / "classifier_dataset.csv"))
POSITIVE_CATALOG = pd.read_csv(ROOT / "output" / "positive_evidence_catalog.csv")
NEGATIVE_AUDIT = pd.read_csv(ROOT / "output" / "ai4aceip_label_audit.csv")


def length_support(length: int) -> tuple[int, int]:
    positive_count = int(
        POSITIVE_CATALOG[
            POSITIVE_CATALOG["standard_sequence"].astype(bool)
            & (POSITIVE_CATALOG["length"] == length)
        ]["sequence"].nunique()
    )
    negative_count = int(
        NEGATIVE_AUDIT[
            (NEGATIVE_AUDIT["ai4aceip_label"] == 0)
            & (~NEGATIVE_AUDIT["label_conflict"].astype(bool))
            & (NEGATIVE_AUDIT["standard_sequence"].astype(bool))
            & (NEGATIVE_AUDIT["length"] == length)
        ]["sequence"].nunique()
    )
    return positive_count, negative_count


LENGTH_3_SUPPORT = length_support(3)
LENGTH_4_SUPPORT = length_support(4)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#0A1F44; --teal:#00B4B3; --aqua:#66D6D3; --mist:#F6F8FB; --slate:#6B7280; }
        .stApp { background: linear-gradient(180deg,#F8FBFD 0%,#FFFFFF 34%); color:var(--navy); }
        .block-container { max-width:1320px; padding-top:1.6rem; }
        [data-testid="stSidebar"] { background:#F3F7FA; border-right:1px solid #DCE6ED; }
        .dd-header { display:flex; align-items:center; gap:18px; padding:12px 0 26px; border-bottom:1px solid #DCE6ED; margin-bottom:24px; }
        .dd-header img { width:74px; height:58px; object-fit:contain; }
        .dd-name { font-size:2.35rem; line-height:1; font-weight:700; letter-spacing:-1px; color:var(--navy); }
        .dd-name span { color:var(--teal); }
        .dd-tag { margin-top:7px; font-size:.72rem; letter-spacing:.30em; color:#53647A; font-weight:650; }
        .dd-kicker { color:var(--teal); font-size:.74rem; letter-spacing:.16em; font-weight:700; text-transform:uppercase; }
        .dd-card { background:#fff; border:1px solid #DEE8EE; border-radius:14px; padding:18px 20px; box-shadow:0 8px 26px rgba(10,31,68,.045); }
        div[data-testid="stMetric"] { background:#fff; border:1px solid #DEE8EE; border-radius:12px; padding:13px 16px; }
        div[data-testid="stMetricValue"] { color:var(--navy); }
        .stButton > button, .stDownloadButton > button { border-radius:9px; border:1px solid var(--navy); }
        .stButton > button[kind="primary"] { background:var(--navy); color:#fff; }
        .stButton > button[kind="primary"]:hover { background:#123566; border-color:#123566; }
        h1,h2,h3 { color:var(--navy); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    encoded = base64.b64encode(ASSET.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="dd-header">
          <img src="data:image/png;base64,{encoded}" alt="DigiDigest symbol">
          <div><div class="dd-name">Digi<span>Digest</span></div>
          <div class="dd-tag">DISCOVER · PRIORITIZE · IMPACT</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(protein_id: str, sequence_text: str, start: int, end: int, target_stage: str):
    protein_id = protein_id.strip()
    if not protein_id:
        raise ValueError("Protein title cannot be empty.")
    sequence = normalize_protein_sequence(sequence_text)
    if start < 1 or end < start or end > len(sequence):
        raise ValueError(f"The selected region must be within 1-{len(sequence)}.")
    fragments = staged_digest(protein_id, sequence[start - 1 : end], coordinate_start=start)
    all_fragments = annotate_and_rank(fragments, ROOT / "models", ROOT / "output", target_stage)
    ranked = build_ranked_candidates(all_fragments, target_stage)
    return protein_id, sequence, all_fragments, ranked


def score_input_sequence(sequence: str) -> dict:
    features = feature_matrix([sequence])
    classifier_in_scope = CLASSIFIER_MIN <= len(sequence) <= CLASSIFIER_MAX
    regression_min = int(REGRESSOR_BUNDLE["minimum_length"])
    regression_max = int(REGRESSOR_BUNDLE["maximum_length"])
    regression_in_scope = regression_min <= len(sequence) <= regression_max
    probability = (
        float(CLASSIFIER_BUNDLE["model"].predict_proba(features)[0, 1])
        if classifier_in_scope else None
    )
    predicted_pic50_raw = (
        float(REGRESSOR_BUNDLE["model"].predict(features)[0])
        if regression_in_scope else None
    )
    known_ace = sequence in set(POSITIVE_CATALOG["sequence"].astype(str))
    classifier_positive = probability is not None and probability >= 0.5
    predicted_pic50 = predicted_pic50_raw if classifier_positive else None
    return {
        "probability": probability,
        "predicted_pic50": predicted_pic50,
        "predicted_ic50": 10 ** (6.0 - predicted_pic50) if predicted_pic50 is not None else None,
        "classifier_in_scope": classifier_in_scope,
        "regression_in_scope": predicted_pic50 is not None,
        "known_ace": known_ace,
        "classifier_positive": classifier_positive,
        "ace_supported": known_ace or classifier_positive,
    }


st.set_page_config(page_title="DigiDigest", page_icon=None, layout="wide")
inject_style()
header()

st.markdown('<div class="dd-kicker">ACE Peptide Discovery Workspace</div>', unsafe_allow_html=True)
st.title("Protein digestion and candidate prioritization")
st.caption("Generates theoretical digestion products, matches ACE evidence, and ranks every candidate for laboratory follow-up.")

default_name, default_sequence = split_single_fasta(EXAMPLE_FASTA.read_text(encoding="utf-8"))
if "protein_name_input" not in st.session_state:
    st.session_state["protein_name_input"] = default_name
if "protein_sequence_input" not in st.session_state:
    st.session_state["protein_sequence_input"] = default_sequence
if "pending_history_input" in st.session_state:
    pending_name, pending_sequence = st.session_state.pop("pending_history_input")
    st.session_state["protein_name_input"] = pending_name
    st.session_state["protein_sequence_input"] = pending_sequence

input_tab, settings_tab, history_tab, model_tab = st.tabs(
    ["Protein input", "Analysis settings", "Analysis history", "Model & data status"]
)
with input_tab:
    uploaded = st.file_uploader("Upload a FASTA file", type=["fa", "fasta", "faa", "txt"])
    if uploaded:
        upload_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("processed_upload") != upload_key:
            try:
                uploaded_name, uploaded_sequence = split_single_fasta(uploaded.getvalue().decode("utf-8"))
                st.session_state["protein_name_input"] = uploaded_name
                st.session_state["protein_sequence_input"] = uploaded_sequence
                st.session_state["processed_upload"] = upload_key
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(str(exc))
    protein_name = st.text_input("Protein title", key="protein_name_input")
    sequence_text = st.text_area(
        "Amino-acid sequence", key="protein_sequence_input", height=210,
        help="Enter only the amino-acid sequence here, without the > FASTA header.",
    )

try:
    preview_length = len(normalize_protein_sequence(sequence_text))
except ValueError:
    preview_length = 1

with settings_tab:
    col1, col2, col3 = st.columns(3)
    with col1:
        target_label = st.selectbox("Target digestion stage", ["Intestinal", "Gastric"])
    with col2:
        region_start = st.number_input("Start position", min_value=1, max_value=max(preview_length, 1), value=min(21, preview_length), step=1)
    with col3:
        region_end = st.number_input("End position", min_value=1, max_value=max(preview_length, 1), value=max(preview_length, 1), step=1)
    st.caption("For the P02863 example, UniProt annotates residues 1–20 as the signal peptide and 21–286 as the mature chain.")

with history_tab:
    history = list_analyses(HISTORY_DB)
    if history.empty:
        st.info("No analyses have been saved yet.")
    else:
        history_display = history.copy()
        history_display["target_stage"] = history_display["target_stage"].map({"intestinal": "Intestinal", "gastric": "Gastric"})
        st.dataframe(history_display.rename(columns={
            "id": "Record", "created_at": "Date", "protein_name": "Protein",
            "protein_length": "Length", "region_start": "Start",
            "region_end": "End", "target_stage": "Stage",
        }), use_container_width=True, hide_index=True)
        selected_history_id = st.selectbox(
            "Select a record", history["id"].tolist(),
            format_func=lambda value: f"#{value} · {history.loc[history['id'] == value, 'protein_name'].iloc[0]}",
        )
        history_load, history_delete = st.columns(2)
        if history_load.button("Open selected analysis", use_container_width=True):
            saved = load_analysis(HISTORY_DB, int(selected_history_id))
            st.session_state["analysis_result"] = (saved["protein_name"], saved["sequence"], saved["fragments"], saved["ranked"])
            st.session_state["target_label"] = "Intestinal" if saved["target_stage"] == "intestinal" else "Gastric"
            st.session_state["analysis_region"] = (int(saved["region_start"]), int(saved["region_end"]))
            st.session_state["pending_history_input"] = (saved["protein_name"], saved["sequence"])
            st.session_state.pop("saved_analysis_id", None)
            st.rerun()
        if history_delete.button("Delete selected record", use_container_width=True):
            delete_analysis(HISTORY_DB, int(selected_history_id))
            st.rerun()

with model_tab:
    st.subheader("Model and evidence boundaries")
    st.caption(
        "This page distinguishes evidence, model predictions and unresolved data gaps. "
        "A missing database match is never treated as proof of inactivity."
    )
    status_cols = st.columns(4)
    status_cols[0].metric("ACE-positive catalogue", f"{POSITIVE_CATALOG['sequence'].nunique():,}")
    candidate_negatives = NEGATIVE_AUDIT[
        (NEGATIVE_AUDIT["ai4aceip_label"] == 0)
        & (~NEGATIVE_AUDIT["label_conflict"].astype(bool))
        & (NEGATIVE_AUDIT["standard_sequence"].astype(bool))
    ]
    status_cols[1].metric("Candidate negatives", f"{candidate_negatives['sequence'].nunique():,}")
    status_cols[2].metric("Classifier scope", f"{CLASSIFIER_MIN}–{CLASSIFIER_MAX} aa")
    status_cols[3].metric("Exact label conflicts", f"{int(NEGATIVE_AUDIT['label_conflict'].astype(bool).sum()):,}")

    short_scope = pd.DataFrame(
        [
            {
                "Length": length,
                "ACE-positive sequences": support[0],
                "Candidate-negative sequences": support[1],
                "Classifier decision": "Not trained — negative evidence missing",
            }
            for length, support in [(3, LENGTH_3_SUPPORT), (4, LENGTH_4_SUPPORT)]
        ]
    )
    st.markdown("#### Unresolved short-peptide gap")
    st.dataframe(short_scope, use_container_width=True, hide_index=True)
    st.info(
        "Unknown 3–4 aa peptides are not classified and receive no predicted IC50. "
        "Other bioactivity labels, such as antioxidant activity, are not valid ACE-negative labels because a peptide may have multiple activities."
    )
    st.markdown("#### Evidence labels used by DigiDigest")
    st.dataframe(
        pd.DataFrame(
            [
                ["Known ACE — experimental", "Exact ACE-positive database match", "Measured IC50 may be shown when available"],
                ["Predicted ACE candidate", "Positive classifier result within validated length scope", "Predicted IC50 may be shown and is labelled as predicted"],
                ["Classification negative", "Classifier result below the decision threshold", "No predicted IC50"],
                ["ACE assay required", "Classifier not applicable and no experimental ACE match", "No predicted IC50"],
            ],
            columns=["Status", "Basis", "IC50 rule"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "AI4ACEIP negative labels remain candidate negatives until their experimental or curation provenance is verified. "
        "All model outputs are research-prioritization results, not clinical efficacy or absorption claims."
    )

if st.button("Run digestion and rank candidates", type="primary", use_container_width=True):
    try:
        result = run_analysis(
            protein_name,
            sequence_text,
            int(region_start),
            int(region_end),
            "intestinal" if target_label == "Intestinal" else "gastric",
        )
        st.session_state["analysis_result"] = result
        st.session_state["target_label"] = target_label
        st.session_state["analysis_region"] = (int(region_start), int(region_end))
        protein_id, sequence, all_fragments, ranked = result
        st.session_state["saved_analysis_id"] = save_analysis(
            HISTORY_DB, protein_id, sequence, int(region_start), int(region_end),
            "intestinal" if target_label == "Intestinal" else "gastric", all_fragments, ranked,
        )
    except Exception as exc:
        st.error(str(exc))

if "analysis_result" in st.session_state:
    protein_id, sequence, all_fragments, ranked = st.session_state["analysis_result"]
    target_stage = "intestinal" if st.session_state["target_label"] == "Intestinal" else "gastric"
    st.divider()
    st.subheader("Analysis summary")
    metrics = st.columns(5)
    metrics[0].metric("Protein length", len(sequence))
    metrics[1].metric("Gastric fragments", int((all_fragments["stage"] == "gastric").sum()))
    metrics[2].metric("Intestinal fragments", int((all_fragments["stage"] == "intestinal").sum()))
    metrics[3].metric("Unique candidates", len(ranked))
    metrics[4].metric("Known ACE matches", int(ranked["known_ace_positive_exact_match"].sum()))

    analyzed_start, analyzed_end = st.session_state.get("analysis_region", (1, len(sequence)))
    analyzed_sequence = sequence[analyzed_start - 1 : analyzed_end]
    parent_score = score_input_sequence(analyzed_sequence)
    st.markdown("#### Input sequence model assessment")
    parent_cols = st.columns(4)
    parent_cols[0].metric("Analyzed sequence length", f"{len(analyzed_sequence)} aa")
    parent_cols[1].metric(
        "ACE probability",
        f"{100 * parent_score['probability']:.1f}%" if parent_score["probability"] is not None else "Unavailable",
    )
    parent_cols[2].metric(
        "ACE assessment",
        "Known ACE" if parent_score["known_ace"] else "Predicted ACE candidate" if parent_score["classifier_positive"] else "ACE assay required" if parent_score["probability"] is None else "Classification negative",
    )
    parent_cols[3].metric(
        "Predicted IC50",
        f"{parent_score['predicted_ic50']:.1f} µM" if parent_score["predicted_ic50"] is not None else "Not estimated",
    )
    if parent_score["classifier_in_scope"]:
        with st.expander("Explain the parent ACE probability"):
            parent_explanation = pd.DataFrame(
                explain_classifier(CLASSIFIER_BUNDLE, analyzed_sequence, CLASSIFIER_REFERENCE)
            )
            st.bar_chart(parent_explanation.set_index("Feature group"))
            st.dataframe(parent_explanation, use_container_width=True, hide_index=True)
            st.caption("Positive values support the ACE classification; negative values oppose it. This is model behavior, not a causal mechanism.")
    if not parent_score["ace_supported"]:
        st.caption(
            "IC50 is not estimated because ACE activity has not been supported by an exact experimental match or a positive ACE classification."
        )

    st.warning("The safety layer is not complete. ACE ranking does not establish absence of celiac epitopes, allergenicity, toxicity, absorption, or efficacy in humans.")
    if "saved_analysis_id" in st.session_state:
        st.success(f"Analysis saved to history (record #{st.session_state['saved_analysis_id']}).")
    st.subheader("Candidate ranking")
    ranking_view = st.radio(
        "Ranking view",
        ["Novel discovery candidates", "Overall evidence ranking", "Positive controls"],
        horizontal=True,
        help="Known ACE peptides are separated as controls; peptides without exact prior ACE evidence are treated as novel candidates.",
    )
    f1, f2, f3 = st.columns([1, 1, 1.4])
    min_len = int(ranked["length"].min())
    max_len = int(ranked["length"].max())
    with f1:
        if min_len == max_len:
            st.caption(f"Peptide length: {min_len} aa")
            length_range = (min_len, max_len)
        else:
            length_range = st.slider("Peptide length", min_len, max_len, (min_len, max_len))
    with f2:
        evidence_filter = st.selectbox("Evidence filter", ["All", "Known ACE positive", "Novel candidates"])
    with f3:
        model_scope_only = st.checkbox("Only ACE-supported candidates")

    with st.expander("When is IC50 estimated?"):
        st.write(
            "Predicted IC50 is shown only after ACE activity is supported by an exact experimental ACE match or by a positive "
            f"ACE classification (probability >= 50%). The classifier currently supports {CLASSIFIER_MIN}–{CLASSIFIER_MAX} aa. "
            f"At length 3 the catalog contains {LENGTH_3_SUPPORT[0]} positives and {LENGTH_3_SUPPORT[1]} candidate-negatives; "
            f"at length 4 it contains {LENGTH_4_SUPPORT[0]} positives and {LENGTH_4_SUPPORT[1]} candidate-negatives. "
            "Unknown 3–4 aa peptides therefore receive 'ACE assay required' and no IC50 estimate."
        )

    shown = ranked[ranked["length"].between(*length_range)].copy()
    active_rank_column = "priority_rank"
    if ranking_view == "Novel discovery candidates":
        shown = shown[shown["candidate_role"] == "novel_discovery_candidate"]
        active_rank_column = "novel_candidate_rank"
    elif ranking_view == "Positive controls":
        shown = shown[shown["candidate_role"] == "positive_control_candidate"]
        active_rank_column = "control_rank"
    if evidence_filter == "Known ACE positive":
        shown = shown[shown["known_ace_positive_exact_match"]]
    elif evidence_filter == "Novel candidates":
        shown = shown[~shown["known_ace_positive_exact_match"]]
    if model_scope_only:
        shown = shown[shown["ace_supported"]]

    shown = shown.sort_values(active_rank_column)
    shown["ace_probability_display"] = shown["ace_probability_exploratory"].map(format_ace_probability)
    shown["ace_status_display"] = shown.apply(
        lambda item: format_ace_status(
            bool(item["known_ace_positive_exact_match"]),
            bool(item["classifier_in_scope"]),
            bool(item["ace_prediction_positive"]),
        ),
        axis=1,
    )
    display_columns = {
        active_rank_column: "Rank", "relative_lab_priority_percentile": "Relative lab priority %",
        "sequence": "Peptide", "length": "Length", "known_ace_positive_exact_match": "Known ACE",
        "measured_median_pIC50": "Measured pIC50", "measured_median_IC50_uM": "Measured IC50 µM",
        "ace_probability_display": "ACE probability", "ace_status_display": "ACE status",
        "predicted_pIC50_exploratory": "Predicted pIC50",
        "predicted_IC50_uM_exploratory": "Predicted IC50 µM", "systemic_size_category": "Size category",
        "target_stage_occurrences": "Occurrences", "protein_locations": "Protein location",
        "ranking_reason": "Ranking basis", "ace_assessment_status": "ACE assessment",
    }
    display_table = shown[list(display_columns)].rename(columns=display_columns)
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relative lab priority %": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.1f"),
        },
    )
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download full ranking (CSV)", ranked.to_csv(index=False).encode("utf-8"), "ranked_ace_candidates.csv", "text/csv", use_container_width=True)
    dl2.download_button("Download all digestion fragments (CSV)", all_fragments.to_csv(index=False).encode("utf-8"), "all_staged_fragments.csv", "text/csv", use_container_width=True)

    st.subheader("Candidate detail and XAI")
    detail_source = shown if not shown.empty else ranked
    chosen = st.selectbox("Select peptide", detail_source["sequence"].tolist())
    row = ranked[ranked["sequence"] == chosen].iloc[0]
    detail1, detail2, detail3, detail4 = st.columns(4)
    role_is_control = row["candidate_role"] == "positive_control_candidate"
    role_rank = row["control_rank"] if role_is_control else row["novel_candidate_rank"]
    detail1.metric("Control rank" if role_is_control else "Novel candidate rank", int(role_rank))
    classifier_text = (
        f"{100 * row['ace_probability_exploratory']:.1f}%"
        if pd.notna(row["ace_probability_exploratory"])
        else "Not applicable — experimental ACE evidence"
        if bool(row["known_ace_positive_exact_match"])
        else "Unavailable — ACE assay required"
    )
    measured_text = "No measurement" if pd.isna(row["measured_median_IC50_uM"]) else f"{row['measured_median_IC50_uM']:.1f} µM"
    predicted_text = "Outside regression scope" if pd.isna(row["predicted_IC50_uM_exploratory"]) else f"{row['predicted_IC50_uM_exploratory']:.1f} µM"
    detail2.metric("ACE probability", classifier_text)
    detail3.metric("Measured median IC50", measured_text)
    detail4.metric("Predicted IC50", predicted_text if pd.notna(row["predicted_IC50_uM_exploratory"]) else "Not estimated")
    priority_col, evidence_col, occurrence_col = st.columns(3)
    priority_col.metric("Relative lab priority", f"{row['relative_lab_priority_percentile']:.1f}%")
    evidence_col.metric("Experimental ACE match", "Yes" if bool(row["known_ace_positive_exact_match"]) else "No")
    occurrence_col.metric("Digestion occurrences", int(row["target_stage_occurrences"]))
    role_label = "Positive control candidate" if role_is_control else "Novel discovery candidate"
    st.markdown(
        f"**Role:** {role_label}  \n"
        f"**Overall evidence rank:** {int(row['priority_rank'])}  \n"
        f"**ACE assessment:** {row['ace_assessment_status']}  \n"
        f"**Location:** {row['protein_locations']}  \n"
        f"**Ranking basis:** {row['ranking_reason']}  \n"
        f"**Size assessment:** {row['systemic_size_category']}"
    )
    st.caption(
        "Relative lab priority is a within-run ranking percentile, not the probability of oral absorption or clinical efficacy."
    )

    st.markdown("#### Local model explanation")
    if bool(row["classifier_in_scope"]):
        explanation_rows = explain_classifier(CLASSIFIER_BUNDLE, chosen, CLASSIFIER_REFERENCE)
        explanation = pd.DataFrame(explanation_rows)
        predicted_ic50_for_summary = (
            float(row["predicted_IC50_uM_exploratory"])
            if pd.notna(row["predicted_IC50_uM_exploratory"])
            else None
        )
        xai_card = classifier_explanation_card(
            float(row["ace_probability_exploratory"]),
            explanation_rows,
            predicted_ic50=predicted_ic50_for_summary,
        )
        xai_left, xai_middle, xai_right = st.columns(3)
        xai_left.metric("Decision", xai_card["Decision"])
        xai_middle.metric("ACE probability", xai_card["Probability"], xai_card["Threshold margin"])
        xai_right.metric("Predicted potency", xai_card["Predicted potency"])
        st.dataframe(
            pd.DataFrame(
                [
                    ["Positive contribution", xai_card["Main support"]],
                    ["Negative contribution", xai_card["Main opposition"]],
                    ["Laboratory interpretation", xai_card["Laboratory interpretation"]],
                    ["Potency interpretation", xai_card["Potency interpretation"]],
                ],
                columns=["Assessment", "Interpretation"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Technical XAI details"):
            st.caption(
                "Influence is the probability change obtained by replacing one feature group with the training-set reference."
            )
            st.bar_chart(explanation.set_index("Feature group"))
            st.dataframe(explanation, use_container_width=True, hide_index=True)
    elif bool(row["known_ace_positive_exact_match"]):
        st.info(
            "An exact experimental ACE match exists. Only measured IC50 is shown; predicted IC50 and model XAI are not reported because the ACE classifier was not applied."
        )
    else:
        st.info("ACE activity is not established for this peptide. An ACE assay is required, so IC50 and potency XAI are not reported.")
    st.caption(
        "This is a local feature-group ablation explanation of model behavior. It is not a causal biochemical mechanism."
    )
else:
    st.info("The example P02863 gliadin FASTA is ready. Review the settings and run the analysis.")

st.divider()
st.caption("Research discovery prototype · All results require laboratory validation")
