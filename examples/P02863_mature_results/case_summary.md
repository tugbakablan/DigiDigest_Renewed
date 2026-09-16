# P02863 Alpha/Beta-Gliadin: First Real-Protein Case

## Input and processing choice

- Source: UniProtKB reviewed entry P02863 (GDA0_WHEAT), *Triticum aestivum*.
- Precursor length: 286 amino acids.
- UniProt processing annotation: signal peptide 1-20; mature chain 21-286.
- Digested region: residues 21-286. Original UniProt coordinates are retained.
- Simulation: theoretical complete cleavage; gastric pepsin pH > 2 approximation,
  followed by trypsin plus high-specificity chymotrypsin.

## Output counts

- Gastric fragment occurrences: 34.
- Intestinal fragment occurrences: 38.
- Unique ranked intestinal candidates: 35.
- Exact ACE-positive evidence matches: 2.
- Candidates with measured median pIC50 in the local evidence data: 1.
- Candidates inside classifier scope: 16.
- Candidates inside pIC50 regression scope: 33.

## Highest-priority observations

1. `VR` (residues 21-22) is an exact ACE-positive match and has measured
   evidence in the assembled data; it ranks first because evidence takes
   precedence over an unvalidated prediction.
2. `AL` (259-260) is an exact ACE-positive match without a regression-eligible
   measured pIC50 in the current table.
3. `IPCMDVVL` (144-151) is the highest-ranked novel model candidate: classifier
   probability 0.588 and predicted pIC50 4.613 (predicted IC50 about 24.4 uM).
4. Other high-ranked novel hypotheses include `QQSTY`, `QQQQPSSQVSF`, `QQYPL`,
   `QTLPAMCNVY`, and `CTIAPF`.

## Critical safety interpretation

`QPFPQPQLPY` appears in the ranking because the current MVP scores ACE potential,
not coeliac immunogenicity or allergenicity. It overlaps a well-known
alpha-gliadin coeliac T-cell epitope region and must not be advanced as a
functional-food candidate without a dedicated safety exclusion workflow.
More generally, a high ACE score is not sufficient for candidate selection.

## Recommended next gate

Before ordering peptides, add an exclusion/flag layer for known coeliac epitopes,
allergenicity and toxicity. Then select a small panel containing: a known
positive control (`VR`), top novel candidates that pass the safety screen, and
one or more low-ranked controls. Confirm actual generation by INFOGEST plus
LC-MS/MS before interpreting ACE assay results as digestion-derived activity.
