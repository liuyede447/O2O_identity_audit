# Canonical prospective lockbox command runbook

Status: **pre-freeze template; paths become immutable at instrument freeze**.

All commands stop on a nonzero exit code. No raw lockbox CSV may be opened manually. Replace only `<PROJECT_ROOT>` with the absolute current project root; the remaining names and parameters are frozen by the protocol.

## 1. Stage, source-only commit, and tag

```powershell
$PROJECT = '<PROJECT_ROOT>'
$FREEZE = Join-Path $PROJECT 'freeze_bundles\audit_instrument_v1.0'
D:\anaconda3\python.exe "$PROJECT\scripts\create_instrument_freeze_bundle.py" stage --output $FREEZE
git -C $FREEZE init
git -C $FREEZE config core.autocrlf false
git -C $FREEZE config user.name 'O2O audit instrument freeze'
git -C $FREEZE config user.email 'audit-freeze@example.invalid'
git -C $FREEZE add .
git -C $FREEZE commit -m 'Freeze audit instrument v1.0'
$COMMIT = (git -C $FREEZE rev-parse HEAD).Trim()
git -C $FREEZE tag audit_instrument_v1.0 $COMMIT
D:\anaconda3\python.exe "$FREEZE\scripts\create_instrument_freeze_bundle.py" finalize --output $FREEZE --source-commit $COMMIT --selection-seed 20260831
D:\anaconda3\python.exe "$FREEZE\scripts\create_instrument_freeze_bundle.py" validate --output $FREEZE
D:\anaconda3\python.exe "$FREEZE\scripts\create_instrument_freeze_bundle.py" seal-preregistration --output $FREEZE
D:\anaconda3\python.exe "$FREEZE\scripts\create_instrument_freeze_bundle.py" validate-preregistration-seal --output $FREEZE
```

Do not continue unless `audit_instrument_v1.0` resolves to the source-only commit, `audit_preregistration_v1.0` resolves to the clean final freeze HEAD, and the complete frozen-files validation passes.

## 2. Outcome-blind disjoint selection

```powershell
$SELECTION = Join-Path $PROJECT 'lockbox_runs\selection_v1'
D:\anaconda3\python.exe "$FREEZE\scripts\prepare_disjoint_audit_lockbox.py" `
  --data "$FREEZE\configs\aitod_v2_local_g.yaml" `
  --exclude-selected "$FREEZE\frozen_inputs\discovery_selected_images_exclusion.csv" `
  --output-dir $SELECTION `
  --seed 20260831 --sample-images 300 --imgsz 800 `
  --preregistration "$FREEZE\CONFIRMATORY_AUDIT_PLAN.json" `
  --instrument-manifest "$FREEZE\TOOL_FREEZE_MANIFEST.json"
```

The selector commits the three-file selection bundle and creates `audit_lockbox_selection_v1.0` before returning. Verify that its repository is clean. Never replace the selected sample.

## 3. Four sealed raw extractions

Common frozen inputs:

```powershell
$CHECKPOINT = 'E:\two_paper\beifen\archive_backup\artifacts\analysis_runs\detect\experiments\baseline_13zone\yolo26s_aitodv2_baseline_13zone_e300_s0\weights\best.pt'
$CHECKPOINT_SHA = '4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa'
$SELECTED = Join-Path $SELECTION 'selected_images.csv'
$SELECT_MANIFEST = Join-Path $SELECTION 'selection_manifest.json'
$PREREG = Join-Path $FREEZE 'CONFIRMATORY_AUDIT_PLAN.json'
$INSTRUMENT = Join-Path $FREEZE 'TOOL_FREEZE_MANIFEST.json'
$DATA = Join-Path $FREEZE 'configs\aitod_v2_local_g.yaml'
```

Fixed branch:

```powershell
$FIXED_BRANCH = Join-Path $PROJECT 'lockbox_runs\sealed_fixed_branch_v1'
D:\anaconda3\python.exe "$FREEZE\scripts\run_reviewer_killer_controls.py" --checkpoint $CHECKPOINT --expected-sha256 $CHECKPOINT_SHA --data $DATA --output-dir $FIXED_BRANCH --selected-images $SELECTED --selected-manifest $SELECT_MANIFEST --preregistration $PREREG --instrument-manifest $INSTRUMENT --stress-mode fixed_1px --normalization equivalent_side --kappa 0.0625 --detector-contract yolo26 --imgsz 800 --device 0 --workers 0 --bootstrap 5000 --normalized-only --sealed-extraction
```

Normalized branch:

```powershell
$NORMALIZED_BRANCH = Join-Path $PROJECT 'lockbox_runs\sealed_normalized_branch_v1'
D:\anaconda3\python.exe "$FREEZE\scripts\run_reviewer_killer_controls.py" --checkpoint $CHECKPOINT --expected-sha256 $CHECKPOINT_SHA --data $DATA --output-dir $NORMALIZED_BRANCH --selected-images $SELECTED --selected-manifest $SELECT_MANIFEST --preregistration $PREREG --instrument-manifest $INSTRUMENT --stress-mode normalized --normalization equivalent_side --kappa 0.0625 --detector-contract yolo26 --imgsz 800 --device 0 --workers 0 --bootstrap 5000 --normalized-only --sealed-extraction
```

Fixed anatomy:

```powershell
$FIXED_ANATOMY = Join-Path $PROJECT 'lockbox_runs\sealed_fixed_anatomy_v1'
D:\anaconda3\python.exe "$FREEZE\scripts\analyze_native_alignment_path_anatomy.py" --checkpoint $CHECKPOINT --expected-sha256 $CHECKPOINT_SHA --data $DATA --output-dir $FIXED_ANATOMY --selected-images $SELECTED --selected-manifest $SELECT_MANIFEST --preregistration $PREREG --instrument-manifest $INSTRUMENT --stress fixed_1px --kappa 0.0625 --imgsz 800 --device 0 --workers 0 --sealed-extraction
```

Normalized anatomy:

```powershell
$NORMALIZED_ANATOMY = Join-Path $PROJECT 'lockbox_runs\sealed_normalized_anatomy_v1'
D:\anaconda3\python.exe "$FREEZE\scripts\analyze_native_alignment_path_anatomy.py" --checkpoint $CHECKPOINT --expected-sha256 $CHECKPOINT_SHA --data $DATA --output-dir $NORMALIZED_ANATOMY --selected-images $SELECTED --selected-manifest $SELECT_MANIFEST --preregistration $PREREG --instrument-manifest $INSTRUMENT --stress equivalent_side --kappa 0.0625 --imgsz 800 --device 0 --workers 0 --sealed-extraction
```

Only progress, counts, errors, and hashes may be viewed during extraction.

## 4. Close the sealed bundle

```powershell
$READY = Join-Path $PROJECT 'lockbox_runs\SEALED_BUNDLE_READY.json'
D:\anaconda3\python.exe "$FREEZE\scripts\validate_sealed_lockbox_bundle.py" --fixed-branch $FIXED_BRANCH --normalized-branch $NORMALIZED_BRANCH --fixed-anatomy $FIXED_ANATOMY --normalized-anatomy $NORMALIZED_ANATOMY --selected-manifest $SELECT_MANIFEST --preregistration $PREREG --instrument-manifest $INSTRUMENT --output $READY
```

Both `SEALED_BUNDLE_READY.json` and `SEALED_BUNDLE_READY.json.sha256` must exist and validate.

## 5. Single outcome access

```powershell
$VERDICT = Join-Path $PROJECT 'lockbox_runs\confirmatory_verdict_v1'
D:\anaconda3\python.exe "$FREEZE\scripts\analyze_confirmatory_lockbox.py" --fixed-dir $FIXED_BRANCH --normalized-dir $NORMALIZED_BRANCH --fixed-anatomy $FIXED_ANATOMY --normalized-anatomy $NORMALIZED_ANATOMY --preregistration $PREREG --selected-manifest $SELECT_MANIFEST --instrument-manifest $INSTRUMENT --sealed-ready $READY --output-dir $VERDICT --reps 5000 --seed 20260831
```

This is the only permitted first interpretation of lockbox outcomes. It atomically creates `OUTCOME_ACCESS_RECEIPT.json` beside the ready artifact before reading the first raw CSV. If the analyzer later fails, the receipt remains and no replacement prospective run is permitted. Every H1--H5 result is retained; no failed gate permits resampling or parameter changes.
