# E1 editorial closure changelog

Date: 2026-09-06

This pass made no scientific, numerical, estimand, table, evidence, or Lockbox-v2 changes.

- Terminology was aligned to assigned-state language in manuscript prose, captions, figure source labels, Figure 4A, Figure 6A, and Figure S1; the historical taxonomy label `Active disappearance` remains explicitly identified.
- Figure 4A now labels first-departure survival and its caption states the weighted Kaplan--Meier first-departure semantics, four-cardinal-ray scope, and descriptive censoring interpretation.
- Main Results §4.5 now states Rank--Set contrasts in the canonical 16--32 minus 8--16 px direction. Supplement Table S20 retains its generated legacy tiny-minus-small direction with an explicit caption; no generated table value was edited.
- Lockbox-v1 narrative remains concise in Methods, absent from Results chronology, evidence-status-only in Discussion, and limited to one sentence in Limitations; Supplement chronology is retained.
- Defensive-language audit classified existing scope statements and did not add repeated disclaimers.

Verification: `E1_SIGN_CONVENTION_QA.json` PASS; source artifacts remain SHA-bound; figure scripts rebuilt Figures 4, 6, and S1; PDFs were rebuilt in the E1 build directory and copied to the manuscript tree only after successful compilation.
