# Evidence ledger schema

Every final evidence entry must contain the following keys, even when the value is `null` because a legacy artifact cannot prove it:

- `claim_id` and `evidence_id` (identical, unique);
- `claim`;
- `dataset`;
- `checkpoint_sha256`;
- `seed`;
- `unit`;
- `denominator`;
- `estimand`;
- `stress_contract`;
- `sampling`;
- `weighting`;
- `uncertainty`;
- `estimates` with explicitly labelled point and interval fields;
- `multiplicity_status`;
- `evidence_stage_status` (`instrument`, `discovery`, `confirmatory`, `exploratory`, or a clearly qualified subtype);
- `artifact` and `artifact_sha256`;
- `script_sha256`;
- `manifest_path` and `manifest_sha256`;
- `validity_notes`;
- `missing_reason`, mapping every unproved `null` field to a reason.

Smoke, failed, incomplete, or invalidated runs cannot be evidence entries. Their records remain in `RUN_REGISTRY.csv` and, when relevant, `excluded_from_evidence`.

The final ledger must pass `scripts/validate_current_evidence_ledger.py` before any manuscript number is populated from it.
