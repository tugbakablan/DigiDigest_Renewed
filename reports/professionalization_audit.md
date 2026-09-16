# DigiDigest professionalization audit

## Product claim

DigiDigest is a research-prioritization system. It generates rule-based digestion
fragments, checks exact ACE evidence, applies models only inside documented
domains, and ranks candidates for laboratory follow-up. It does not establish
oral absorption, safety, antihypertensive efficacy, or clinical benefit.

## Current verified assets

| Component | Current asset | Permitted interpretation |
| --- | --- | --- |
| Positive evidence | 2,440 unique ACE-positive sequences | Exact match is prior ACE evidence |
| IC50 evidence | 3,087 eligible records; 1,177 unique sequences | Exact normalized measurements only |
| Candidate negatives | 1,734 non-conflicting AI4ACEIP-labelled sequences | Training candidates pending provenance review |
| Activity classifier | Length-matched, similarity-clustered 5–18 aa baseline | Exploratory ACE probability within scope |
| Potency regressor | Similarity-clustered 2–30 aa baseline | Predicted pIC50 only after ACE support gating |
| Digestion | Pepsin, then trypsin/chymotrypsin rule engine | Theoretical fragments, not observed yield |

## Non-negotiable evidence rules

1. Database absence is unknown activity, not a negative label.
2. A different bioactivity label (for example antioxidant activity) is not an
   ACE-negative label; multifunctional peptides are possible.
3. Censored, ranged, percent-inhibition and unparsed potency values are excluded
   from the first pIC50 regression target.
4. Measured and predicted IC50 values are never merged or displayed as if they
   had the same evidence level.
5. Predicted IC50 is hidden when ACE classification is unavailable or negative.
6. Rule-based digestion is not described as INFOGEST validation or LC-MS/MS
   detection.
7. Relative laboratory priority is a within-run ordering, not an absorption or
   efficacy probability.

## Blocking gap: 3–4 aa classification

The positive catalogue contains 477 unique 3 aa sequences and 238 unique 4 aa
sequences. The current candidate-negative source contains no 3 aa or 4 aa
sequences. A 3–18 aa binary classifier would therefore be structurally invalid:
lengths 3 and 4 would contain only one class.

Unknown 3–4 aa sequences must remain `ACE assay required` until the gate below
is passed.

### Data acceptance criteria

Each candidate negative must include:

- exact standardized sequence;
- explicit ACE assay outcome or an auditable source label;
- assay method and substrate when reported;
- tested concentration or activity threshold when reported;
- citation or stable database identifier;
- label type (`experimental_negative`, `weak_or_inactive_at_tested_dose`, or
  `curated_candidate_negative`);
- no exact conflict with the positive catalogue.

### Model-release gate

The 3–4 aa classifier scope may be enabled only after:

1. both classes are present at each length;
2. provenance review is complete and conflicts are quarantined;
3. train/test separation uses sequence-similarity groups;
4. performance is reported separately for 3 aa, 4 aa, and the full scope;
5. probability calibration and uncertainty are evaluated;
6. an untouched external set or laboratory set confirms acceptable behavior;
7. the model card and UI scope are updated together.

No fixed minimum sample count is declared in advance as proof of quality.
Learning curves and confidence intervals must show whether the collected sample
size is adequate.

## Prioritized development backlog

### P0 — Scientific correctness

- Audit AI4ACEIP candidate-negative provenance at record level.
- Build a dedicated short-peptide negative evidence table for 3 aa and 4 aa.
- Add duplicate, exact-conflict, non-standard residue, unit and censoring tests.
- Add probability calibration, confidence intervals and per-length evaluation.
- Preserve an untouched external validation set.

### P1 — Candidate evidence

- Add explicit digestion-confidence fields rather than one opaque total score.
- Add celiac epitope, allergenicity and toxicity evidence as separate auditable
  layers; do not call unvalidated heuristics safety clearance.
- Add LC-MS/MS observation status and assay results when laboratory data become
  available.
- Record source version, retrieval date and citation for every evidence row.

### P1 — User experience

- Show ACE probability and ACE status in separate columns.
- Keep measured and predicted potency in separate columns.
- Provide a per-candidate evidence trail and model explanation.
- Save analyses with model version, data version, parameters and timestamps.
- Provide exports containing the complete provenance and not only visible rows.

### P2 — Reproducibility and release

- Pin dependency versions and create a deterministic environment lock.
- Add continuous tests for data preparation, model training and UI contracts.
- Version models and datasets with checksums.
- Publish a model card, data sheet and release notes for each production build.
- Add access control and privacy rules before storing user sequences remotely.

## Immediate next dataset task

Create `short_peptide_negative_evidence.csv` with the acceptance fields above.
Populate it only from explicit ACE assay evidence or a provenance-audited
negative source. Until then, the production classifier remains 5–18 aa.
