"""Run the frozen random-stratified O2O audit at one alternate shift magnitude."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def pop_option(name: str) -> str:
    if name not in sys.argv:
        raise SystemExit(f"required option missing: {name}")
    index = sys.argv.index(name)
    try:
        value = sys.argv[index + 1]
    except IndexError as exc:
        raise SystemExit(f"value missing: {name}") from exc
    del sys.argv[index : index + 2]
    return value


def option_value(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def main() -> None:
    magnitude = float(pop_option("--perturbation-pixels"))
    if magnitude <= 0:
        raise SystemExit("perturbation magnitude must be positive")
    output_dir = Path(option_value("--output-dir"))

    import assignment_stability_random_stratified as audit

    audit.PERTURBATIONS = (
        (f"left_{magnitude:g}px", -magnitude, 0.0),
        (f"right_{magnitude:g}px", magnitude, 0.0),
        (f"up_{magnitude:g}px", 0.0, -magnitude),
        (f"down_{magnitude:g}px", 0.0, magnitude),
    )
    audit.main()
    manifest_path = output_dir / "sampling_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "protocol": "random_image_stratified_active_identity_magnitude_sensitivity_v1",
            "perturbation_pixels": magnitude,
            "primary_inference": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("MAGNITUDE_SENSITIVITY_PASS", magnitude, output_dir)


if __name__ == "__main__":
    main()
