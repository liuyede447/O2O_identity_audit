# Assignment spillover gated-pilot specification

Status: **pre-outcome gated extension**.

The spillover assay is not required for instrument freeze or the primary lockbox. It may proceed to a full 300-image discovery analysis only if the deterministic pilot establishes scientific interpretability.

## Frozen pilot design

- Source: the 300-image discovery manifest.
- Subset: 30 images ranked by ascending `SHA256("20260831|{image_id}")`, with image ID as deterministic tie-break.
- Stress: equivalent-side `kappa=0.0625`.
- Primary structural relation: shared native pre-conflict candidate claim in base or shifted state (`yes` versus `no`).
- Primary continuous descriptor: normalized GT-centre distance, without a cutoff.
- Secondary descriptive split only: shared-conflict / local-nonsharing / distant-nonsharing with the CLI-recorded threshold `2.0`; this threshold cannot affect a primary estimate.
- Uncertainty: 5,000 image-cluster bootstrap replicates within frozen sampling stratum.

## Gate to a full discovery run

A full run is scientifically warranted when both primary relationship strata are populated and at least one nonfocal loss-active identity change is observed. An event confined to nonshared pairs also passes the gate because it indicates unexpected nonlocal coupling that requires characterisation.

If the pilot contains no nonfocal identity change, the extension stops after reporting the pilot design, denominators, zero-event result, interval/upper-bound information available from the frozen bootstrap, and the prespecified stop rule. Zero events never imply proof that spillover is impossible.

Pilot outcomes cannot change the primary relation, subset, stress contract, or full-run estimand.
