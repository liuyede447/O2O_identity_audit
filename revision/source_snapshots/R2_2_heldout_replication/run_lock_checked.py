"""Run the frozen eligibility-lock replay with a selection pre-flight check before any inference.

The replay script (unchanged, hash-checked) draws its own stratified sample. This wrapper replaces
the SelectedDataset class in the script's namespace with a subclass that, when the data loader is
about to be built (after sampling, before the first forward pass and before the output directory is
created), compares the selected dataset indices, image ids and strata with the preregistered list
and aborts on any difference.

Usage: run_lock_checked.py --expected-selection LIST [--check-only] -- <replay script arguments>
--check-only stops right after a successful check: no forward pass, no output.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(r"E:\two_paper\beifen\publication_reproducible\scripts\run_eligibility_lock_counterfactual.py")
SCRIPT_SHA = "cfcb4bc8bc88670013acaaf113371608b028d906170071daf216169916141c5d"


class PreflightPassed(Exception):
    pass


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_lock_checked.py --expected-selection LIST [--check-only] -- <replay args>")
    own, replay = argv[: argv.index("--")], argv[argv.index("--") + 1:]
    check_only = "--check-only" in own
    expected = Path(own[own.index("--expected-selection") + 1])
    if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != SCRIPT_SHA:
        raise RuntimeError("replay script hash mismatch")
    exp = pd.read_csv(expected).sort_values("image_id").reset_index(drop=True)

    spec = importlib.util.spec_from_file_location("lock_replay", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = mod.SelectedDataset
    image_stratum = mod.image_stratum

    class CheckedSelection(base):
        def __init__(self, dataset, indices):
            ids = [Path(dataset.labels[i]["im_file"]).stem for i in indices]
            strata = [image_stratum(dataset.labels[i], 800) for i in indices]
            got = pd.DataFrame({"dataset_index": indices, "image_id": ids, "stratum": strata}).sort_values("image_id").reset_index(drop=True)
            same = (len(got) == len(exp)
                    and got["dataset_index"].astype(int).tolist() == exp["dataset_index"].astype(int).tolist()
                    and got["image_id"].astype(str).tolist() == exp["image_id"].astype(str).tolist()
                    and got["stratum"].astype(str).tolist() == exp["stratum"].astype(str).tolist())
            if not same:
                raise RuntimeError("PRE-FLIGHT FAILED: replay selection differs from the preregistered list; no inference was run")
            print(f"PRE-FLIGHT PASSED: {len(got)} images, indices/images/strata equal to {expected.name}", flush=True)
            if check_only:
                raise PreflightPassed
            super().__init__(dataset, indices)

    mod.SelectedDataset = CheckedSelection
    sys.argv = [str(SCRIPT)] + replay
    try:
        mod.main()
    except PreflightPassed:
        print("check-only: stopped before data loading and inference", flush=True)


if __name__ == "__main__":
    main()
