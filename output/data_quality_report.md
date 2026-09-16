# ACE peptide data quality report

## Prepared collections

- Positive-evidence catalogue: 2,440 unique sequences
- Regression-eligible IC50 records: 3,087 records across 1,177 unique sequences
- AI4ACEIP candidate negatives after exact positive-conflict removal: 1,734 unique sequences
- AI4ACEIP exact negative/positive-evidence conflicts: 16 unique sequences
- Previous DigiDigest theoretical fragments: 612 unique sequences
- Theoretical fragments with positive ACE evidence: 160
- Theoretical fragments with unknown activity: 452

## Decisions

1. Do not treat database absence as a negative label.
2. Do not train the activity model on single-residue digestion products.
3. Keep repeated IC50 measurements at record level; assay context can change the value.
4. Exclude censored, ranged, percent-inhibition and unparsed values from the first IC50 regression model.
5. Treat AI4ACEIP negatives as candidate negatives until their evidence provenance is verified.
6. Match or reweight positive and negative samples by peptide length before classification.
7. Split model development data by sequence-similarity clusters, not random rows.
8. Treat previous BIOPEP digestion as rule-based reference output, not experimental validation.

## Next build step

Create two separately evaluated baselines:

- ACE activity classifier using positive evidence and a reviewed, length-matched negative subset.
- pIC50 regression model using only exact, normalized, positive IC50 records.

The digestion engine should remain a separate module. It will digest full FASTA proteins stage by stage and send the resulting peptides to the activity models.
