# PROJECT_GOAL.md — north star

Durable statement of what this project is trying to do and how we'll know it
worked. Pairs with `CLAUDE.md` (conventions), `TODO.md` (task status),
`PLAN_monitor_features.md` (visual-feature ladder), and
`REVIEW_2026-06_decoding_gaps_and_tasks.md` (current gaps + task plan).
Keep this file short and stable; churn goes in TODO/notes.

---

## 1. The mission, in one sentence

Learn the **bidirectional map** between the visual scene a mouse sees while
navigating a VR world and the calcium activity of its visual-cortex
populations — well enough that we can (a) say *which cells, in what sequence,
represent an object at a given retinotopic location and motion*, and (b)
**re-drive that exact pattern with holographic 2p optogenetics** to test
whether it produces a perception / behavior change.

## 2. The key reframe — one problem, two directions

Decoding (neural → stimulus) and encoding (stimulus → neural) are the **same
map read two ways**, and the optogenetics endgame lives on the **encoding**
side:

- **Encoding model** `stimulus_features(t-window) → activity_n(t)` per neuron.
  This is literally "how monitor inputs are transformed into calcium." Its
  fitted tuning *is* the answer to "which cells fire for an object at location
  X moving direction Y" — i.e. the opto target. **This is the primary arm and
  it does not yet exist in the repo** (the current pipeline only decodes
  behavior *from* neural data).
- **Decoding model** `activity(t) → stimulus_feature(t)`. Validation +
  representation analysis: how much, and where (which area), is each feature
  legible in the population. This is what `VR_ca_cebra.py` does today.
- **Bridge to opto:** invert/condition the encoding model on a chosen
  stimulus event ("object enters right hemifield at 20°, looming") → predicted
  population activity pattern → intersect with the cells that are (i)
  well-predicted by that feature and (ii) extractable as a stable ensemble
  (`f_ensemble_extract`) and (iii) physically targetable → a ≤50-cell
  holographic stimulation list. This is the Walker/Sinz "inception loop" /
  Bashivan "neural population control" paradigm applied to your rig.

## 3. The three arms

> Status verified 2026-06-13 (repo-local): arm 1 (encoding) **confirmed absent**
> — no `stimulus→neural` model in the repo; tracked as TODO **#57** (primary).
> Arm 2 (decoding) confirmed present and its CV harness audited (train-only
> standardize + both-side embargo; CEBRA in-sample leak caveated). Arm 3
> (ensembles) present; cross-CV/session identity = TODO **#65**.

1. **Encoding arm (NEW, primary).** Per-neuron predictive model from visual +
   self-motion features to activity. Reports held-out **per-neuron
   correlation, normalized by a noise ceiling**. Output: a tuning map (which
   feature block / which retinotopic location-motion drives each cell) and the
   stimulus→pattern generator for the opto arm.
2. **Decoding / representation arm (EXISTS).** `VR_ca_cebra.py` +
   `f_decoding.py` + `f_cebra_helpers.py`. Honest blocked-CV decodability of
   each feature block, per recorded area. Key planned result: a
   **decodability gradient across areas** — object-center (`agg`) features
   should climb from V1 → higher visual areas (LM/AL/…); retinotopic
   edge/motion-energy features should dominate in V1. Treat this gradient as a
   *finding*, not a nuisance.
3. **Ensemble / opto arm (PARTLY EXISTS).** `VR_ca_dimred.py` +
   `f_ensembles.py` already extract ensembles with shuffle nulls + cross-seed
   stability. Needs: cross-validated (and ideally cross-session) ensemble
   identity, and the wiring to the encoding model so an ensemble can be tied
   to a stimulus cause and emitted as an opto target.

## 4. Success criteria (how we know it worked)

Stop chasing R² in the abstract; anchor every number to a ceiling and a null.

- **Encoding:** a clear fraction of cells are FDR-significant against a per-cell
  circular-shift null (autocorr-preserving), and their null-referenced effect
  size (z = (r − null_mean)/null_std) has a right tail well above 0 — used in
  place of a noise ceiling (no stimulus repeats in free foraging). Encoding is
  **higher for V1-style retinotopic motion-energy features than for object-center
  scalars** in V1 recordings.
- **Decoding:** for at least one feature block, blocked-CV R² (and real−null
  Δ) clears the `agg`/`pix` baseline on features that currently sit
  near/below chance, **with the OOF prediction-vs-truth slope ≈ 1** (genuine
  signal, not drift-misled — this is already your stated bar in
  `PLAN_monitor_features.md §4`).
- **Areal gradient:** object-center decodability is measurably ordered across
  recorded areas in the predicted direction.
- **Opto-readiness:** for a defined stimulus event, the model emits a ranked,
  stable, targetable ≤50-cell pattern that is reproducible across held-out
  time.

## 5. Things that will mislead us if ignored (standing confounds)

- **Input fidelity.** The encoding target is real V1 activity driven by the
  *actual displayed scene*; the model input is currently a **synthetic
  silhouette render** (`f_angles_to_movie_v2`, terrain optional/off). Any V1
  variance driven by scene content absent from that render is *unexplainable
  by construction*. Quantify this before blaming models. (See REVIEW P0-2.)
- **Locomotion / arousal.** Running speed and pupil drive large V1 variance in
  navigating mice. Self-motion must be a **predictor in the encoding model**,
  not just a decoding target, or object tuning will be confounded by the fact
  that the mouse moves differently near objects.
- **Calcium + nonstationarity.** GCaMP6s integrates ~1 s of history; slow
  drift across a session produces deep-negative blocked-CV R² independent of
  any real signal. Use deconvolved `S` (already done), high-pass where
  appropriate (#8/#10), and ceiling-normalize.
- **No stimulus repeats in free foraging.** A classic repeat-based noise
  ceiling isn't directly available; use object-approach/capture *event*
  alignment for a reliability estimate instead (see REVIEW P0-3).

## 6. Non-goals (for now)

- Not trying to reconstruct a photorealistic monitor image from neural data;
  the silhouette/feature reconstruction is a *diagnostic*, not a deliverable.
- CEBRA is a visualization / hypothesis tool here, **not** the production
  decoder (see REVIEW P0-1).
