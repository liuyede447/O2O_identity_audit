# Figure 2 final layout specification

Claim: observed scale response depends on the stress contract; the paired branch contrast retains its sign but has different branch-wise decomposition.

Canvas: 178 mm by 118 mm, white, 2 x 2 statistical panels. Arial only. Panel labels 8 pt bold; titles 7.8 pt; axes, legend, and ticks at least 7 pt. No grid except a 0.65 pt gray reference line.

## a. Primary stress-contract forest

- Six horizontal rows: Fixed O2O, O2M, paired; equivalent-area-side O2O, O2M, paired.
- X axis: `8-16 minus 16-32 px fragility contrast (pp)`.
- Point and 95% stratified image-cluster bootstrap CI.
- Fixed uses filled markers; equivalent-area-side uses open markers.
- Notes distinguish positive O2O decomposition under fixed stress from reverse O2M decomposition under equivalent-area-side stress.

## b. Kappa x dataset sensitivity

- Paired O2O-minus-O2M contrast at kappa 0.03125, 0.0625, and 0.125.
- AI-TOD-v2 circles; VisDrone squares; thin lines connect tested magnitudes only.
- No dose-response or interpolation claim.

## c. Physically archived stages

- Six source-driven stages; points and CIs only, no continuous curve.
- The gap between epochs 100 and 280 is marked `No intermediate archived checkpoints`.
- Epoch 0 is the first completed framework checkpoint, not random initialization.
- The final row is displayed as `Final` (source archive label `300`) because it is the archived `last.pt`, not the retained primary `best.pt`.

## d. Compatible native-contract replication

- Two grouped forests: YOLO26s and YOLOv10-S, each showing O2O, O2M, and paired.
- Title: `Compatible native-contract replication`.
- Required note: native assignment semantics differ; the comparison concerns the observed response signature, not architecture generalisation.

## Data and visual gates

- Every plotted number and CI is loaded from a SHA-verified frozen artifact.
- O2O is blue circle; O2M orange square; paired purple diamond.
- CI bars are never clipped and remain behind markers.
- Text is black/dark gray; no red-green status encoding.
- SVG text remains editable; PDF uses TrueType 42; PNG is 600 dpi at final physical size.
