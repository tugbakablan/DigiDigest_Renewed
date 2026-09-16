# ACE Peptide Data Preparation

The scientific scope, unresolved 3–4 aa data gap, evidence rules and release
gates are documented in `reports/professionalization_audit.md`.

This package prepares the first auditable data layer for an ACE-inhibitory
peptide discovery workflow. It keeps source observations separate from derived
labels and does not treat an unreported activity as a negative result.

## Interface update

- Protein title and amino-acid sequence use separate fields.
- Uploading one FASTA record fills both fields automatically.
- Successful analyses are saved locally in `user_data/analysis_history.sqlite3`.
- Saved analyses can be reopened or deleted from the history tab.
- A single candidate length no longer causes a Streamlit slider error.
- The full interface is presented in English.
- The analyzed parent sequence and digestion fragments are assessed separately.
- In-scope classifier predictions are displayed as percentages.
- Local feature-group ablation XAI shows which model feature groups support or
  oppose each prediction; these explanations are not causal mechanisms.
- A relative laboratory-priority percentile is shown as a within-run ranking
  aid. It is not an absorption or clinical-efficacy probability.

### Classifier length boundary

The classifier remains restricted to 5–18 aa. After the current BIOPEP-UWM
export was imported, the cleaned catalog contains 477 positive and 0
candidate-negative records at length 3, and 238 positive and 0
candidate-negative records at length 4. A 3–18 classifier therefore
cannot learn a supported positive/negative distinction for 3–4 aa peptides.
For unknown short peptides, IC50 is not estimated until the ACE classifier can
be applied and returns a positive result. A known short ACE peptide may show its
measured IC50, but it does not receive a predicted IC50 when classification is
unavailable. Unknown short peptides are marked `ACE assay required`.

The BIOPEP-UWM import added 181 new unique positive sequences. Its original IDs,
sequences, masses and activity labels are retained in `sources/`, while six
non-standard or modified sequences are kept in a separate audit file.

## Inputs

Place these files in one directory and pass that directory with `--input-dir`:

- `ahtpdb_small_peptides.txt`
- `ahtpdb_long_peptides.txt`
- `ahtpdb_ic50.txt`
- `step1_biopep_with_ic50.xlsx`
- `02_ACE_TOXICITY(1).xlsx`
- `03_ACE_WITH_DIGESTION_FRAGMENTS(2).xlsx`
- `pos.fa`
- `neg.fa`

The AI4ACEIP source code and pretrained pickle files are not required. The
repository does not contain an explicit software license, so this package does
not copy or execute that code.

## Run

```bash
python prepare_data.py --input-dir /path/to/input --output-dir output
python train_baselines.py --data-dir output --model-dir models --report-dir reports
python score_peptides.py --fasta candidates.fa --model-dir models --output candidate_scores.csv
python digest_and_rank.py --fasta protein.fa --data-dir output --model-dir models --output-dir results

# İsteğe bağlı: tam listenin yanında ilk 20 için ayrı çalışma listesi
python digest_and_rank.py --fasta protein.fa --data-dir output --model-dir models --output-dir results --selection-count 20

# UniProt'ta olgun zincir 21-286 olarak verilmişse yalnızca bu bölgeyi sindir
python digest_and_rank.py --fasta precursor.fa --sequence-start 21 --sequence-end 286 --data-dir output --model-dir models --output-dir results

## Research interface

```bash
pip install -r requirements.txt
streamlit run app.py
```

The interface accepts one uploaded or pasted FASTA record, allows a mature-chain
coordinate range and target digestion stage to be selected, runs the existing
models, displays the complete ranked candidate table, provides filters and
candidate detail, and exports both ranking and fragment CSV files. It uses the
DigiDigest symbol and brand palette without decorative emoji.

Known ACE matches are labelled as positive-control candidates and are separated
from novel discovery candidates. The interface shows overall evidence rank,
novel-candidate rank and control rank independently. Classifier non-application
is explicitly described as a peptide-length/model-scope issue, and measured IC50
is displayed separately from model-predicted IC50.
```

## Outputs

- `source_summary.csv`: record counts and basic quality checks by source.
- `positive_evidence_catalog.csv`: one row per peptide with positive evidence
  sources. This is an evidence catalogue, not a model split.
- `ic50_records_clean.csv`: record-level IC50 values with preserved raw values,
  normalized units, censoring flags and regression eligibility.
- `ai4aceip_label_audit.csv`: one row per AI4ACEIP sequence with conflict flags.
- `ai4aceip_negative_conflicts.csv`: sequences labelled negative by AI4ACEIP but
  present in AHTPDB or the earlier BIOPEP positive collection.
- `digestion_fragment_catalog.csv`: unique theoretical digestion fragments from
  the previous DigiDigest work, with parent counts and evidence status.
- `data_quality_report.md`: generated summary and next-step recommendations.

The baseline training command additionally creates:

- `classifier_dataset.csv`: length-matched exploratory classification data.
- `classifier_metrics.csv`: clustered cross-validation metrics, including a
  length-only shortcut baseline.
- `classifier_oof_predictions.csv`: out-of-fold predictions for error review.
- `regression_dataset.csv`: sequence-level median pIC50 targets with replicate
  counts and dispersion.
- `regression_metrics.csv`: clustered cross-validation regression metrics.
- `regression_oof_predictions.csv`: out-of-fold pIC50 predictions.
- `models/ace_classifier.joblib` and `models/pic50_regressor.joblib`: fitted
  baseline models with metadata.
- `reports/baseline_model_report.md`: scope, metrics and scientific limits.

The staged digestion command creates:

- `all_staged_fragments.csv`: gastric and intestinal fragments with one-based
  protein coordinates, enzyme profile, exact evidence matches and model scores.
- `ranked_ace_candidates.csv`: every unique target-stage candidate ranked from
  highest to lowest priority. No candidate is discarded by a top-N cutoff.
- `selected_first_N_candidates.csv`: optional convenience shortlist, written
  only when `--selection-count N` is supplied. The complete ranking is always
  retained.

## Included real-protein case

`examples/P02863_alpha_beta_gliadin.fa` contains the reviewed UniProt wheat
alpha/beta-gliadin precursor. `examples/P02863_mature_results/` contains the
complete ranking obtained after selecting its annotated mature chain (21-286),
plus `case_summary.md`. The case deliberately records the coeliac-epitope safety
issue revealed by the ranking; it is not presented as a ready functional-food
candidate list.

The transparent ranking gives precedence to an exact measured-IC50 match, then
an exact positive-evidence match, then an in-scope model prediction. Within a
tier, a transparent length window is applied before higher pIC50 (lower IC50):
4–12 residues is the preferred screening window, 2–3 is retained as small and
potentially absorbable, >12 is flagged for limited intact absorption, and a
single residue is not treated as a peptide-model candidate. Repeated sequences
are ranked once and their occurrence count is reported. This heuristic does not
combine absorption, toxicity or systemic efficacy into an unsupported number.

## Scientific boundaries

- AHTPDB and BIOPEP positives are not proof of systemic efficacy.
- AI4ACEIP negatives are candidate negatives, not experimentally confirmed
  inactive peptides.
- The classifier is restricted to lengths with at least ten positive and ten
  candidate-negative examples after exact-length matching. With the supplied
  data this is 5–18 amino acids. Peptides outside that scope must not receive a
  classifier probability from this baseline.

- BIOPEP digestion fragments are rule-based predictions, not INFOGEST/LC-MS/MS
  observations.
- `digest_and_rank.py` is likewise a deterministic, complete-cleavage hypothesis
  generator. Its gastric profile uses a broad pepsin pH > 2 approximation; its
  intestinal profile applies trypsin and high-specificity chymotrypsin cleavage
  to each gastric product. It does not simulate enzyme kinetics, food matrix,
  transit time, brush-border peptidases, uptake, metabolism or bioavailability.
- Exact IC50 values are retained at record level because assay conditions differ.
- `less_than`, `greater_than`, range, percent-inhibition and unparsed records are
  excluded from the initial regression subset but remain in the audit table.
- Model evaluation groups similar sequences to reduce train/test leakage. It is
  still exploratory until experimentally confirmed negatives and external
  validation data are available.
