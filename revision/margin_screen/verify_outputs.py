"""Check five reproduced tables against the archived 5,000-draw outputs."""
import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
summary = json.loads((args.output / "summary.json").read_text("utf-8"))
if summary["reps"] != 5000:
    raise ValueError("A full 5,000-draw run is required.")
for table in sorted(Path(__file__).with_name("expected").glob("*.csv")):
    regenerated = pd.read_csv(args.output / table.name, float_precision="round_trip")
    expected = pd.read_csv(table, float_precision="round_trip")
    assert_frame_equal(regenerated, expected, check_exact=False, rtol=1e-10, atol=1e-12)
    print(f"{table.name}: {len(expected)} rows verified")
