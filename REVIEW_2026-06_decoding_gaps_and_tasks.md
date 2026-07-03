# REVIEW 2026-06 — why decoding is dying, and the gaps the current plan doesn't cover

Instruction file for the local (repo-aware) Claude. Companion to
`PROJECT_GOAL.md`. **Reconcile every task number below against `TODO.md`
before creating new ones** — assign real IDs there, don't invent a parallel
scheme. Append a dated rationale entry to `notes.txt` for each change, per
`CLAUDE.md` conventions.

## Provenance (what this review is based on)

Read in full: `CLAUDE.md`, `VR_ca_dimred.py`, `PLAN_monitor_features.md`, and
the first ~1000 lines of `VR_ca_cebra.py` (setup → feature build → CEBRA fit →
embedding scatter → first decode-bar cell). **Not** read (GitHub fetch
limits — the reviewer could not reach `functions/` or the back half of
`VR_ca_cebra.py`): all of `functions/*.py` (esp. `f_cebra_helpers.py`,
`f_decoding.py`, `f_feature_helpers.py`, `f_visual_features.py`,
`f_ensembles.py`), the rest of `VR_ca_cebra.py`, `VR_ca_analysis.py`,
`VR_video_recon.py`, `TODO.md`, `notes.txt`. Treat any claim about helper
internals as **a hypothesis to verify** (see §Verify checklist), not a fact.

## How this relates to what you already have

`PLAN_monitor_features.md` already scopes the visual-feature ladder correctly
(grid → flow → Gabor → motion-energy → CNN) with the right literature and
"filter→nonlinearity→pool→compress→decode" framing. **This review does not
touch that ladder** except to re-prioritize it. The gaps below sit *around*
that plan: input fidelity, decoder correctness, the missing encoding arm, the
noise ceiling, temporal model structure, the areal gradient, and the opto
bridge. The working diagnosis is that **building Tier 2/3/4 features will not
rescue decoding until P0 is resolved**, because P0 items can each produce
near-zero / negative R² no matter how good the features are.

---

## Central diagnosis — ranked causes of near-zero / negative R²
(Re-ranked 2026-06-13 after the scene-sparsity / dark-condition discussion;
input fidelity demoted.)

1. **Decoder correctness** (P0-1): CEBRA embeddings are fit in-sample on all
   frames *with labels* (your own comment flags this), then "decoded" — so
   CEBRA columns are optimistic and not comparable to the PCA-3 baseline; and
   `out_dim=3` is far too small a feature space to carry a moving retinotopic
   scene. Deep-negative R² additionally smells like nonstationarity baseline +
   R² aggregation over hour-long ~216k-frame sessions (P0-1c).
2. **Static decoder vs a temporally-structured stimulus** (P1-4 / P2-1): the
   same retinotopic location yields different activity depending on history
   (fresh sweep / looming vs long-static + adapted); an instantaneous
   location→activity map can't fit both, and pooling regimes hides the signal.
3. **Target definedness / object-in-FOV sparsity** (P1-2): `nan_absent=True`
   makes `lat/vert/dist` defined only when an object is in FOV; tiny, clustered
   CV folds → unstable, easily-negative R². Especially acute in the dark
   condition (object-absent ⇒ ~zero visual drive).
4. **No ceiling** (P0-3): an R² of 0.02 is meaningless without the explainable
   fraction; some "failures" may be at ceiling for a genuinely noisy cell.
5. **Input fidelity** (P0-2) — **DOWNGRADED**: the scene is sparse (grass +
   sky), ground flow is largely redundant with `self_mot`, and the dark
   condition makes objects-only ≈ the true stimulus. Add the horizon; otherwise
   neutralized.
6. **Feature geometry** (the `PLAN_monitor_features.md` ladder): real but
   *downstream* of 1–5.

---

## P0 — correctness & feasibility (do FIRST; cheap; gate everything else)

### P0-1 — Decoder correctness audit (`f_cebra_helpers.py`, `f_decoding.py`)
> **VERIFIED 2026-06-13 — mostly already in place.** (a) A leak-free neural-PC
> decoder already exists (the PCA-decoding sweep → `f_blocked_cv_r2`); CEBRA
> in-sample leak is now caveated, not used for headline numbers. (c)
> `f_blocked_cv_r2` already does train-only standardize + both-side embargo;
> baseline is test-fold mean (sklearn standard). Remaining = #62 (per-fold CEBRA
> refit, raise out_dim, surface pooled-OOF). The presence-in-R² degeneracy was
> fixed 2026-06-12. See `notes.txt`.
**Problem.** Quantitative decodability claims may be biased or artifactual.
**Fixes (each independently shippable):**
- **(a) Stop using CEBRA embeddings as the production decoder.** For every
  headline "how much does the population encode feature X" number, decode
  **directly from neural activity** (ridge / kernel-ridge from neural PCs or
  all cells) through `f_blocked_cv_r2`. This path has no embedding leak. Keep
  CEBRA strictly for visualization and qualitative hypothesis generation. If
  you want a CEBRA decodability number, it **must** use per-fold CEBRA refit
  (fit on train block only, transform test block) — implement as a nested-CV
  option or drop the claim.
- **(b) Raise the embedding bottleneck where CEBRA is used at all.** `out_dim=3`
  cannot carry a retinotopic moving scene. Use 3 only for scatter plots; use
  8–32 for any embedding that feeds a decoder.
- **(c) Audit R² aggregation in `f_blocked_cv_r2`.** Confirm: targets are
  standardized using **train-fold mean/std only**; R² baseline is the
  **train-fold mean** (not a global mean); report **both** per-fold-averaged
  R² and pooled-OOF R² (they diverge under drift and the divergence is
  diagnostic). Verify embargo actually purges both sides of each test block.
  (Hour-long 60 Hz sessions ≈ 216k frames with strong slow drift — bleaching,
  arousal — make train-only baselining and high-pass detrend non-optional, not
  a nicety.)
- **Verify it helped:** on a feature that "should" decode (e.g. `self_mot`
  running speed, or `beh` licking), ridge-from-neural-PCs should give clearly
  positive, stable R² with OOF slope ≈ 1. If even that is near zero, the
  problem is upstream (alignment/ceiling), not the model.

### P0-2 — Input fidelity: mostly handled by the dark condition; add only the horizon in the grass world (DOWNGRADED)
**Reevaluated (user, 2026-06-13).** The grass world is sparse — grass terrain +
hills + sky, objects the only salient features. Full-texture rendering is now
expected to be **low value** for two compounding reasons: (i) mouse acuity is
low (~0.5 cyc/deg) so fine ground texture is near-invisible; (ii) the main thing
the ground gives V1 is optic flow during running, which is **largely redundant
with the `self_mot` predictors already available** (running speed + yaw ≈ ground
flow). Rendering detailed grass and re-extracting flow mostly re-derives what
`self_mot` already carries — provided locomotion is a predictor in the model.
**What IS worth adding in the grass world (cheap):**
- The **horizon edge** (terrain→sky boundary) — the one high-contrast moving
  background feature, coupled to pitch/running. Add it to the render and/or as a
  scalar **'horizon elevation per monitor'** channel.
- The **global ground/sky luminance split**. Skip detailed grass texture.
**The dark condition is the real fix (user is building it).** With bright
objects on black, the **objects-only render ≈ the true stimulus**, so the
input-fidelity confound essentially disappears. Use a dark-condition session as
the **clean ground-truth dataset to debug everything else:**
- Run the P0-1 harness audit and the P1-4 transient/sustained analysis there
  first. If `self_mot`/`beh` decode cleanly but object location still doesn't,
  input fidelity was never the bottleneck — the culprit is the harness, the
  static-vs-transient confound, object-in-FOV sparsity (P1-2), or the ceiling.
- A black background means object-absent frames carry ~zero visual drive (V1 =
  locomotion + spontaneous), so all decodable visual signal concentrates in
  object-present and especially **object-transition** epochs. This promotes
  regime-stratification (P1-4), the event-weighted ceiling (P0-3), and
  presence-gating (P1-2) to the dominant issues — not scene content.
- **Cross-condition comparison is itself a result:** object-feature
  decodability similar in dark vs grass ⇒ objects dominate, background
  irrelevant (validates the current approach); dark ≫ grass ⇒ the grass
  background added interfering variance or those sessions differ in locomotion.
**Priority note:** P0-2 drops from "biggest cheap win" to "add the horizon;
otherwise let the dark condition carry it." Redirect that effort to P0-1, P1-4,
P0-3, P1-2.

### P0-3 — Noise ceiling / explainable variance (NEW; not in any current doc)
**Problem.** Free foraging has no repeated trajectories, so raw R²/correlation
is uninterpretable.
**Fix — event-aligned reliability:** the task *does* repeat **event types**
(object enters FOV, looming, approach, lick-triggered capture). Align activity
to object-approach / capture onsets, build the trial-averaged response per
cell, and estimate the noise ceiling from split-half (odd/even trial)
reliability of those event-locked responses. Report all encoding/decoding
performance for object-related features **as a fraction of this ceiling**.
**Static-period caveat (user, 2026-06-13):** objects can sit static for tens of
seconds, and those frames carry little stimulus-locked drive (largely adapted).
Weight the ceiling and the decoding evaluation toward **stimulus-transition
epochs** (FOV entry, motion onset, looming, capture), not all object-present
frames — otherwise long static holds dilute the ceiling and the scores.
**Files:** new `functions/f_noise_ceiling.py` (compute) + a cell in the
encoding script. Reuse `f_proc_lick_rew` event times for capture onsets and
`vec_data_*['obj_mon_idx']` transitions for FOV-entry onsets.

---

## P1 — the encoding arm + finishing the feature ladder

### P1-1 — Build the encoding model arm (NEW script, primary deliverable)
**Why.** This is the project's stated goal (monitor → calcium) and the source
of the opto target; it currently does not exist.
**Spec (mirror existing module conventions):**
- New script `VR_ca_encoding.py` (Spyder cells), helpers in new
  `functions/f_encoding.py` (compute) + `functions/f_encoding_plots.py`
  (visuals), matching the `f_decoding.py` / `f_ensembles.py` split.
- Model: per-neuron ridge readout from a **lagged** feature design matrix
  (FIR; see P2-1) built from `built_blocks` (visual + `self_mot` + `beh`).
  Start linear; this is the controlled baseline before any RNN.
- **Locomotion is a predictor**, not just a target (standing confound, GOAL §5).
- Score: held-out per-neuron correlation via the same blocked-CV machinery
  (`f_blocked_cv_r2` generalized to per-neuron), **ceiling-normalized** (P0-3).
- Output a **tuning table**: for each cell, which feature block and which
  retinotopic location/direction best predicts it. This table is the seed of
  the opto target generator (P3).
- Variance-partitioning: fit nested models (self-motion only; +object-center;
  +retinotopic motion-energy) and report each block's unique contribution per
  cell/area — this is what reveals the **areal gradient** (GOAL §4).

### P1-2 — Fix object-feature target definedness (#12 / `nan_absent`)
**Problem.** `lat/vert/dist` NaN-masked to in-FOV frames → sparse, clustered,
unstable CV.
**Fix.** Model presence and location **jointly** (hurdle/gating): classify
presence (you already score this with AUC — good), then regress location
*conditional on present*, evaluated on in-FOV epochs with proper temporal
blocking. Report the in-FOV fraction prominently next to every object-location
score (it bounds what's estimable). Cross-reference #17 (multi-object collapse)
— `salience_mean` smoothing is reasonable; keep `nearest` as a comparison.

### P1-3 — Re-prioritize the feature ladder (amendment to `PLAN_monitor_features.md`)
- **Flow (Tier 2) is expected to underperform on the synthetic render** and
  this likely explains "flow doesn't beat pix": Lucas–Kanade / Farnebäck need
  texture gradients; on near-binary silhouettes flow is defined only at object
  edges (aperture problem) and is ~0 elsewhere. Note this in the plan.
- Therefore **prioritize Tier 4a motion-energy (Adelson–Bergen) over Tier 2
  flow** for the V1 motion question: motion energy operates on raw luminance
  via oriented x–y–t quadrature filters and does not depend on trackable
  features, so it degrades gracefully on sparse input and is the principled
  "moving edges" model. Build order becomes: grid (done) → **gabor** →
  **motion-energy** → flow (keep as cheap interpretable comparison) → CNN
  (ceiling benchmark).
- Mouse acuity caveat from your plan stands (small 2–3 SF bank, 4 orientations).

### P1-4 — Separate transient vs sustained regimes (the stimulus's own nonstationarity)
**New constraint (user, 2026-06-13).** In an hour-long session an object may
sweep across in seconds or sit static for tens of seconds, and the mouse
alternates running and stationary. The *same* retinotopic object location
therefore produces very different activity depending on history (fresh
sweep/looming → strong transient; long-static → adapted, little drive). A
static location→activity map cannot fit both regimes, and pooling them hides
whatever signal exists.
**Fixes (cheaper than an RNN; do before P2-2):**
- Add **onset/transient channels** — object FOV-entry, motion onset, looming
  rate — alongside **sustained presence**; let both the encoding and decoding
  models use them. Much of V1's drive is transient, not tonic.
- Add an **adaptation / history term**: time-since-onset, or an exponential
  trace of recent local stimulus energy, so sustained-but-static periods aren't
  modeled as fresh onsets. This is the principled, low-cost form of your
  static→dynamic intuition.
- **Stratify evaluation by regime** (running vs stationary; object moving vs
  static) and report decodability per regime, never only pooled. The likely
  truth is "location decodes during sweeps, not during long static holds" — a
  result, not a failure.

---

## P2 — temporal structure (the static→dynamic fix you flagged)

### P2-1 — Lag/FIR embedding BEFORE any RNN
**Problem.** Decoder/encoder are instantaneous; stimulus + calcium are both
temporally extended.
**Fix (cheap, controlled).** Add a lag-embedding option: predict feature(t)
from neural activity over `[t-Δ, t+Δ]` (decoding) and activity_n(t) from
features over a causal window `[t-Δ, t]` (encoding), by augmenting the design
matrix with lagged copies (or a small temporal-conv). At 60 Hz with GCaMP6s,
sweep Δ over ~0.1–1.5 s. This quantifies *how much temporal context buys*
and is the baseline the RNN must beat.

### P2-2 — RNN encoder (your idea — position correctly)
**Do only after P2-1 establishes the linear-FIR ceiling.** A recurrent (or
temporal-conv) encoder `feature_sequence → activity_sequence` is the natural
nonlinear extension and is well-motivated for accumulating motion over time.
There is already `f_RNN_dred` / an `RNN_scripts` path on `sys.path` — reuse
that infrastructure. Compare to FIR-linear on the **same** ceiling-normalized
metric; report the lift, not just the absolute number.

---

## P3 — the ensemble → opto bridge (the endgame wiring)

### P3-1 — Cross-validated, targetable ensemble identity
`f_ensemble_extract` gives cross-seed *stability* (reliability). Add temporal
cross-validation of ensemble membership (do the same cells co-fire in held-out
time?) and, where possible, cross-session identity, since opto targets must be
re-identifiable on a new day.

### P3-2 — Stimulus → pattern generator
Wire the encoding model (P1-1) to emit, for a specified stimulus event
("object at +20° azimuth, looming, right monitor"), the predicted population
activity vector; rank cells by (predicted drive × encoding reliability ×
ensemble stability); intersect with physically targetable cells; output a
ranked ≤50-cell holographic stimulation list. This is the concrete artifact
the opto rig consumes and the formal version of "which cells fire in what
sequence for a moving object."

---

## Verify checklist — claims about code the reviewer could NOT read

Confirm or correct these in `functions/` before acting:
(VERIFIED 2026-06-13 by repo-local Claude — had read all of `functions/` + full
`VR_ca_cebra.py`. Details in `notes.txt` 2026-06-13.)
- [x] `f_blocked_cv_r2`: train-ONLY standardization ✓ (mu/sd from train rows).
      Embargo purges BOTH sides ✓. R² baseline = TEST-fold mean (sklearn
      `r2_score` standard) — NOT a global mean, so the P0-1c "global mean" worry
      is moot (it is local, just not train-mean). Per-fold mean±std returned;
      pooled-OOF available via `return_predictions`. Multi-output supported →
      per-neuron P1-1 feasible by adaptation.
- [x] `f_flow_features`: `lucas_kanade` confirmed (numpy-only; Farnebäck needs
      cv2, absent). Collapses to ~0 off object edges on the silhouette render —
      confirmed likely (LK needs intensity gradients; silhouettes have them only
      at edges). Supports P1-3 (prefer motion-energy over flow).
- [x] `build_feature_blocks` / `build_visual_blocks`: confirmed — the build cell
      prints `agg` NaN-fraction AND per-side any-obj-in-FOV / multi-obj stats.
- [x] `f_run_cebra`: `out_dim=3`, `max_iter=5000`, `conditional='time_delta'`,
      NO per-fold refit (embeddings fit once on all data → the in-sample leak;
      now caveated in the bar cell). out_dim=3 critique applies ONLY to CEBRA —
      the PCA-decoding sweep already decodes from up to 50 neural PCs leak-free.
- [x] Synthetic render ONLY — `f_angles_to_movie_v2` (+ optional terrain); no
      real-display capture anywhere. Confirms P0-2.
- [ ] Neurons / sessions / mice / areas — data-dependent, NOT determinable from
      code. Still open; pull from the loaded datasets.

## Conventions to honor (from `CLAUDE.md`)
Spyder `#%%` cells; `if 0:`/`if 1:` are intentional toggles; **preserve
commented-out reference code**; NumPy < 2.0 (`np.arctan2` etc.); Windows /
PowerShell paths; append dated rationale to `notes.txt`; keep `TODO.md`
authoritative for task status; new compute/plot code follows the
`f_x.py` (compute) + `f_x_plots.py` (visuals) split.
