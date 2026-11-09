# VR_analysis — summary of analyses, methods, focus and references (for grant writing)

Compiled 2026-10-13 from: `PROJECT_GOAL.md`, `notes.txt` (lab notebook, 2026-05-05 → 07-27),
`TODO.md`, `HANDOFF.md`, `REVIEW_2026-06_decoding_gaps_and_tasks.md`, `PLAN_monitor_features.md`,
`CLAUDE.md` (module log through 2026-08-05), `../reserrch_repo/` (master plan + sub-plans P0–P9),
and `gabor__model_galant_paper.txt`.

> **Status caveat (read first).** Most quantitative numbers in the project files are from
> synthetic validation tests or engineering benchmarks. The analysis infrastructure is
> extensive and validated on synthetic data; the newest components (encoding spine D1–D3,
> neural-scored feature search, FFT/quadrature/wavelet features) are **not yet run end-to-end
> on real sessions**. Real-data results so far are early decoding results (Section 5).
> Phrase the grant as "we have built and validated…", "preliminary…", not "we found…".

---

## 1. Scientific aim (one paragraph)

Mice navigate a two-monitor virtual-reality world containing moving objects while
two-photon calcium imaging records populations in visual cortex (V1 and higher visual areas).
The project learns the **bidirectional map between the visual scene and neural activity**:

- **Encoding (stimulus → neuron), primary arm:** a per-neuron predictive model from
  visual + self-motion features to activity. Its fitted weights are each cell's
  receptive field / tuning ("which cells fire for an object at retinotopic location X
  moving in direction Y").
- **Decoding (neuron → stimulus), validation arm:** how much and where each visual or
  behavioral feature can be read out of the population, and how this varies across areas
  (predicted gradient: object-level features rising from V1 → higher areas; retinotopic
  edge/motion-energy features dominant in V1).
- **Ensembles → optogenetics (endgame):** invert the encoding model for a chosen stimulus
  event, intersect with stable ensembles and targetable cells, and emit a ranked **≤50-cell
  holographic two-photon stimulation list** to test whether re-driving the pattern evokes a
  percept / behavioral response ("inception loop" / neural population control paradigm).

Research plan pillars (`reserrch_repo/PLAN_2026-07_master_overview.md`):
**Goal 1 — read out the code** (P0 data-regime accounting, P1 cortical localization,
P2 per-neuron encoding model = "the spine", P3 receptive-field identification,
P4 cortical functional maps, P5 population latent geometry) and
**Goal 2 — write to the code** (P8 opto target generation, P9 behavioral optogenetics).
Design rule: widefield retinotopy / area labels are **validation overlays only, never model
inputs**, so RF and map results are non-circular.

---

## 2. Data and stimulus reconstruction

- **Neural:** 2p calcium imaging (GCaMP6s, ~60 Hz noted), CaImAn-style estimates; deconvolved
  activity used as rate proxy; ~200 cells/session; hour-long sessions (~100k–216k frames).
  Multiple mice, many sessions per mouse (e.g. mouse L: 17 processed datasets).
- **Behavior:** VR position/heading, speed, yaw, licks, rewards, object captures, per-object
  positions; all aligned to the imaging clock with delay correction.
- **Stimulus reconstruction (key enabling step):** the free-foraging VR has no repeated
  stimulus, so the displayed monitor movie is **re-rendered from logged behavior** — object
  geometry, camera model matched to the Unity source (FOV 98.4° × 66.2°), heightmap terrain
  with depth-buffered occlusion, per-object visible-fraction. Gives a frame-accurate
  synthetic movie per monitor to feed visual feature models.
- **Cortical registration:** cell footprints → 4-point homography from 2p FOV to the
  widefield retinotopic map → per-cell cortical coordinates; scores (encoding R², tuning)
  projected onto the widefield map with blood-vessel overlays.

---

## 3. Analyses and methods

### 3a. Visual feature representations (stimulus side)
A ladder of visual-cortex-style representations, after Hubel–Wiesel → Adelson–Bergen →
Gallant-lab encoding models:
- **Object-level aggregate features** (position, angular size, distance, velocity, presence).
- **Pixel PCA** of the rendered movie (baseline).
- **Retinotopic grid features** (occupancy / contrast / edge per spatial cell).
- **Gabor wavelet pyramid** (orientation × spatial frequency × position), modelled on
  Kay et al. 2008: magnitude (contrast energy), phase, and **signed quadrature** (Re/Im — keeps
  phase in a linear-model-friendly form). Specified in **physical units** (cycles/deg),
  Nyquist-matched sampling, edge-energy guard.
- **Adelson–Bergen spatiotemporal motion energy** (direction/speed; temporal frequency in Hz,
  constant-cycle temporal windows; opponent and both-direction outputs).
- **Patch-FFT spectral features** binned onto the Gabor axes (shown to subsume the Gabor bank:
  a Gabor response equals one Fourier coefficient of the windowed patch), a global-FFT
  (position-free) contrast, and **2-D wavelets** (DWT / shift-invariant SWT).
- **Optical flow** (Lucas–Kanade) — deprioritized (degenerate on filled silhouettes).
- **Self-motion** decomposed into forward / lateral / yaw.
- Two-layer processing per filter / per cell (normalization, PCA, whitening) with
  variance-retained bookkeeping.
- Engineering: GPU (PyTorch) convolution with FFT kernels, streaming over time, quantized
  on-disk cache, resumable scan stores — makes wide parameter sweeps feasible on hour-long sessions.

### 3b. Encoding models (stimulus → neuron) — primary arm
- Per-neuron **ridge regression with a short causal FIR** (lag embedding), blocked
  cross-validation with embargo, out-of-fold Pearson r / R².
- **Significance by per-cell circular-shift null** (autocorrelation-preserving) + BH-FDR and a
  null-referenced z-score — used instead of a repeat-based noise ceiling (no stimulus repeats
  in free foraging). Relative ceiling r/r_full.
- **Receptive fields:** encoding weights → pixel spatiotemporal RFs; spike-triggered average
  (STA) baseline; **smoothness-regularized RFs** (generalized Tikhonov / graph-Laplacian over
  space and lag) with nested σ × λ selection (select on one split, report on held-out tail).
- **Banded ridge** (separate λ per feature space) and **variance partitioning** → per-cell
  visual-unique / behavior-unique / shared variance (Nunez-Elizalde 2019; Dupré la Tour 2022).
- Drop-one block importance; feature-**tuning scans** (reverse correlation over a wide
  orientation × SF × TF grid; per cell and per ensemble; shuffle-null significance).
- Cross-dataset per-neuron encoding scan (resumable store across mice × sessions).

### 3c. Decoding / representation (neuron → stimulus) — validation arm
- Unified **blocked-CV decoder** (train-only standardization, within-fold PCA, embargo),
  contiguous, interleaved and **leave-session-out** fold schemes.
- Decoders: ridge / RidgeCV, PLS, CCA, kNN, multi-output ridge; logistic + ROC-AUC for binary
  (presence) targets; neural lag-embedding (±300 ms FIR).
- Per-feature decode sweep vs number of neural PCs, with **circular-shift null** and
  real − null Δ; OOF prediction-slope diagnostic (slope ≈ 1 = genuine decoding).
- **CEBRA** (behavior-supervised contrastive embedding) — used for visualization/hypotheses,
  not as production decoder (known in-sample leak); Procrustes comparison of embeddings.
- Cross-dataset decoding comparison (per-dataset cache → dataset × feature tables).
- **Feature-selection by neural score:** a search scoring candidate feature parameterizations
  jointly by encode + decode performance on one dataset ("leanest feature set that maximizes
  encode + decode").
- **Stimulus-fidelity ceiling (“Chain-1”):** feature → movie reconstruction, neuron-free, against
  a mouse-acuity–lowpassed target; pooled across sessions with leave-session-out CV.

### 3d. Population structure and ensembles
- Unsupervised dim-red sweep: PCA, NMF (Frobenius / KL), sparse PCA; Hungarian-matched
  seed stability; Hoyer sparsity.
- Ensemble extraction (port of MATLAB pipeline): leave-neuron-out CV grid, shuffle-PCA
  dimensionality, threshold and correlation-cluster extraction, shuffle-based significance;
  behavior-clamped variants (behavior-residualized NMF, constrained NMF, ensembles in CEBRA space).
- Planned (P0/P5): neuron-count subsampling curves real vs shuffled (information-limiting
  correlations), dPCA / targeted dimensionality reduction, latent "map of visual space".

### 3e. Statistical safeguards (worth stating in the grant)
Blocked CV with embargo (temporal autocorrelation), train-only preprocessing, circular-shift
nulls with uniform shifts (unbiased), FDR control, nested hyperparameter selection, pooled OOF
R² (avoids flat-fold blowups), matched-dimensionality comparisons between feature sets,
effective-sample-size accounting, and non-circular validation against widefield maps.

---

## 4. Evolution of focus (timeline)

| Period | Focus |
|---|---|
| Early 2025 – Mar 2026 | Behavior analysis, calcium + monitor vectorization (`VR_analysis.py`, `VR_ca_analysis.py`) |
| May 2026 | Unsupervised dim-red + ensemble methods (NMF/KL, similarity metrics); first CEBRA decoding; renderer geometry matched to Unity; modular feature-block pipeline |
| Jun 2026 | Decoding infrastructure + audit (blocked CV, nulls, ROC-AUC); MATLAB ensemble port; **reframe to bidirectional map with encoding as primary arm (06-13)**; Gabor energy + motion-energy features; first per-neuron encoder; shuffle-test significance |
| Early Jul 2026 | Feature caching and batch generation; cross-dataset decoding; movie-reconstruction ceiling; **P2 encoding spine** (RFs, STA, smoothness priors, banded ridge, variance partition) |
| Mid Jul 2026 | GPU feature-tuning scans; physical-unit filter banks; cross-dataset encoding + tuning scans; 2p → widefield cell registration; neural-scored feature search |
| Late Jul – Aug 2026 (most recent) | **Choosing the visual representation:** Gabor magnitude vs quadrature vs FFT vs wavelet vs motion energy at matched dimensionality; GPU FFT convolution (8×), FFT features (18× speedup); edge guard; making the same feature config mean the same thing in search / decoding / encoding scripts |

**Current priorities (from HANDOFF/TODO):**
1. First real-session scorecard of the feature representations (Gabor mag/quad, FFT, motion energy).
2. Run the encoding spine (RFs → smoothness → visual/behavior variance partition) on real data.
3. Cross-dataset encoding pipeline + per-cell tuning table → feeds the opto target generator.
4. Area labels per cell (from widefield field-sign) → areal gradient analyses.

---

## 5. Results to date (be precise about which are real)

**Real data (preliminary, decoding):**
- First CEBRA / PCA decoding of object position gave negative or near-chance held-out R²;
  diagnosed as slow nonstationarity in pixel-PC targets and estimator issues → drove the
  blocked-CV / detrend / null harness redesign. (05-18, 06-06)
- Pipeline runs end-to-end on real sessions for tuning scans and registration (e.g. L/10_21_25;
  all 17 mouse-L datasets have registration landmarks; cells map inside the FOV).

**Method validation (synthetic / engineering):**
- Planted RF recovered by ridge (r = 0.99) and STA (r = 0.91); smoothness prior raised RF
  recovery 0.17 → 0.57 in low-data regime.
- Nested selection guard exposes optimism: selection r 0.24 vs held-out 0.10.
- Banded-ridge variance partition recovers planted visual / behavior unique variance.
- Ensemble module recovers 3 planted ensembles (CV r² ≈ 0.998).
- FFT-vs-Gabor equivalence: agreement up to 0.9998 on-lattice; quadrature recovers magnitude exactly.
- GPU: 15.5× (tuning scan), 8.1× (FFT conv), ~18× (FFT features) speedups — hour-long sessions
  processed in minutes per filter bank.

---

## 6. Notes marked important (verbatim excerpts)

- 06-13 — "the ENCODING arm … is the primary deliverable … and it DOES NOT YET EXIST."
  "Endgame is a ranked ≤50-cell holographic stimulation list (Walker/Sinz inception loop)."
- 06-12 — Methodology decision: "Shuffle null: min_shift REMOVED … Uniform random shifts over
  [0,T) are the unbiased circular-shift null."
- 07-01 — "Do NOT judge neural representation by movie-pixel reconstruction." "The working basis
  is MAGNITUDE (form) + MENG (motion); phase is a hypothesis to FALSIFY." Reconstruction target =
  mouse-acuity-smoothed movie.
- 07-03 — "P2 pivot: NO dark-vs-light split — all luminance pooled; visual/behavior read on
  DECOUPLED epochs." CEBRA single-fit leak unfixed → CEBRA figures not trustworthy until per-fold refit.
- 07-10 — "Chain-1 optimum ≠ neural optimum: it measures completeness not neural-relevance."
- 07-27 — "Dropping the pca to compare raw blocks would confound representation with
  dimensionality" (→ compare representations at matched 50 dims).
- Master plan — de Vries et al.: only ~10% of cells reliably driven and the fraction falls in
  higher areas → maps **must be conditioned on the reliably-driven fraction**.
- Master plan — "no neural pooling across sessions … pool at the results level"; effective
  sample size ≪ frame count.
- PROJECT_GOAL — standing confounds: input fidelity of the rendered scene, locomotion/arousal
  (self-motion must be a predictor), calcium nonstationarity, no stimulus repeats.

Open methodological issues: #8 nonstationarity of slow targets; #12 empty-frame encoding
ambiguity; #17 multi-object collapse; phase-gate numerical noise (#70/#71).

---

## 7. References cited in the project files

**Visual feature models / encoding**
- Marčelja S (1980). Mathematical description of the responses of simple cortical cells. *JOSA* 70(11):1297–1300.
- Daugman JG (1985). Uncertainty relation for resolution in space, spatial frequency, and orientation… *JOSA A* 2(7):1160–1169.
- Jones JP, Palmer LA (1987). An evaluation of the two-dimensional Gabor filter model of simple receptive fields in cat striate cortex. *J Neurophysiol* 58(6):1233–1258.
- Pollen DA, Ronner SF (1983). *IEEE Trans Syst Man Cybern* — volume/pages not recorded; verify.
- Adelson EH, Bergen JR (1985). Spatiotemporal energy models for the perception of motion. *JOSA A* 2(2):284–299.
- Watson AB, Ahumada AJ (1985). Model of human visual-motion sensing. *JOSA A* 2(2):322–341.
- Kay KN, Naselaris T, Prenger RJ, Gallant JL (2008). Identifying natural images from human brain activity. *Nature* 452:352–355. (Gabor wavelet pyramid model; supplementary methods saved in `gabor__model_galant_paper.txt`.)
- Nishimoto S, Vu AT, Naselaris T, Benjamini Y, Yu B, Gallant JL (2011). Reconstructing visual experiences from brain activity evoked by natural movies. *Curr Biol* 21(19):1641–1646.
- Naselaris T, Kay KN, Nishimoto S, Gallant JL (2011). Encoding and decoding in fMRI. *NeuroImage* 56:400–410.
- Wu MC, David SV, Gallant JL (2006). Complete functional characterization of sensory neurons by system identification. *Annu Rev Neurosci* 29:477–505.
- Theunissen FE, David SV, et al. (2001). Estimating spatiotemporal receptive fields of auditory and visual neurons from their responses to natural stimuli. *Network* 12:289–316.
- Nunez-Elizalde AO, Huth AG, Gallant JL (2019). Voxelwise encoding models with non-spherical multivariate normal priors. *NeuroImage* 197:482–492.
- Dupré la Tour T, Eickenberg M, Nunez-Elizalde AO, Gallant JL (2022). Feature-space selection with banded ridge regression. *NeuroImage* 267:119728.
- Lescroart MD, Gallant JL (2019). Human scene-selective areas represent 3D configurations of surfaces. *Neuron* 101:178–192.
- Sahani M, Linden JF (2003). Evidence optimization techniques for estimating stimulus-response functions. *NeurIPS* 15.
- Park M, Pillow JW (2011). Receptive field inference with localized priors. *PLoS Comput Biol* 7(10):e1002219.
- Duncker L, Ruda KM, Field GD, Pillow JW (2023). Scalable variational inference for low-rank spatiotemporal receptive fields. *Neural Comput* 35:995–.
- Meyer AF, Diepenbrock J-P, et al. (2014). Bias-corrected receptive-field estimation for natural stimuli with asymmetric intensity distributions. (journal not recorded; verify)

**Receptive fields (simple/complex)**
- Rust NC, Schwartz O, Movshon JA, Simoncelli EP (2005). Spatiotemporal elements of macaque V1 receptive fields. *Neuron* 46:945–956.
- Schwartz O, Pillow JW, Rust NC, Simoncelli EP (2006). Spike-triggered neural characterization. *J Vision* 6:484–507.
- Chen X, Han F, Poo M-M, Dan Y (2007). Excitatory and suppressive receptive field subunits in awake monkey V1. *PNAS* 104:19120–19125.
- Fournier J, Monier C, Levy M, Marre O, et al. (2014). Hidden complexity of synaptic receptive fields in cat V1. *J Neurosci* 34:5515–5528.

**Mouse visual cortex: areas, retinotopy, state**
- Marshel JH, Garrett ME, Nauhaus I, Callaway EM (2011). Functional specialization of seven mouse visual cortical areas. *Neuron* 72:1040–1054.
- Andermann ML, Kerlin AM, Roumis DK, Glickfeld LL, Reid RC (2011). Functional specialization of mouse higher visual cortical areas. *Neuron* 72:1025–1039.
- Glickfeld LL, Andermann ML, Bonin V, Reid RC (2013). Cortico-cortical projections in mouse visual cortex are functionally target specific. *Nat Neurosci*.
- Wang Q, Sporns O, Burkhalter A (2012). Network analysis of corticocortical connections reveals ventral and dorsal processing streams in mouse visual cortex. *J Neurosci*.
- de Vries SEJ, Lecoq JA, Buice MA, et al. (2020). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nat Neurosci* 23:138–151.
- Garrett ME, Nauhaus I, Marshel JH, Callaway EM (2014). Topography and areal organization of mouse visual cortex. *J Neurosci* 34:12587–12600.
- Kalatsky VA, Stryker MP (2003). New paradigm for optical imaging: temporally encoded maps of intrinsic signal. *Neuron* 38:529–545.
- Sereno MI, Dale AM, Reppas JB, et al. (1995). Borders of multiple visual areas in humans revealed by fMRI. *Science* 268:889–893.
- Zhuang J, Ng L, Williams D, et al. (2017). An extended retinotopic map of mouse cortex. *eLife* 6:e18372.
- Ronan L, et al. (2021). Functional parcellation of mouse visual cortex … response-dependent clustering. *PLoS Comput Biol* 17(2):e1008548.
- Niell CM, Stryker MP (2010). Modulation of visual responses by behavioral state in mouse visual cortex. *Neuron* 65:472–479.
- Pakan JM, Lowe SC, et al. (2016). Behavioral-state modulation of inhibition is context-dependent and cell type specific in mouse V1. *eLife* 5:e14985.
- Stringer C, Pachitariu M, Steinmetz N, Reddy CB, Carandini M, Harris KD (2019). Spontaneous behaviors drive multidimensional, brain-wide activity. *Science* 364.

**Population coding, dimensionality reduction, ensembles**
- Zohary E, Shadlen MN, Newsome WT (1994). Correlated neuronal discharge rate and its implications for psychophysical performance. *Nature* 370:140–143.
- Moreno-Bote R, Beck J, Kanitscheider I, Pitkow X, Latham P, Pouget A (2014). Information-limiting correlations. *Nat Neurosci* 17:1410–1417.
- Kanitscheider I, Coen-Cagli R, Kohn A, Pouget A (2015). Measuring Fisher information accurately in correlated neural populations. *PLoS Comput Biol*.
- Bartolo R, Saunders RC, Mitz AR, Averbeck BB (2020). Information-limiting correlations in large neural populations. *J Neurosci* 40:1668–1678.
- Cunningham JP, Yu BM (2014). Dimensionality reduction for large-scale neural recordings. *Nat Neurosci* 17:1500–1509.
- Gallego JA, Perich MG, Miller LE, Solla SA (2017). Neural manifolds for the control of movement. *Neuron* 94:978–984.
- Chung SY, Abbott LF (2021). Neural population geometry. *Curr Opin Neurobiol*.
- Chari T, Pachter L (2023). The specious art of single-cell genomics. *PLoS Comput Biol* 19(8):e1011288.
- Kobak D, Brendel W, Constantinidis C, et al. (2016). Demixed principal component analysis of neural population data. *eLife* 5:e10989.
- Mante V, Sussillo D, Shenoy KV, Newsome WT (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature* 503:78–84.
- Schneider S, Lee JH, Mathis MW (2023). Learnable latent embeddings for joint behavioural and neural analysis (CEBRA). *Nature* 617:360–368.
- Mackevicius EL, et al. (2019) — seqNMF (cited by name only in notes).
- Kotliar D, et al. (2019) — consensus NMF (cited by name only in notes).
- Fleet DJ, Jepson AD (1990) — phase-based optical flow (cited by name only).

**Holographic optogenetics / opto targeting**
- Carrillo-Reid L, Yang W, Kang Miller JE, Peterka DS, Yuste R (2019). Controlling visually guided behavior by holographic recalling of cortical ensembles. *Cell* 178:447–457.
- Marshel JH, Kim YS, Machado TA, et al. (2019). Cortical layer-specific critical dynamics triggering perception. *Science* 365:eaaw5202.
- Dalgleish HWP, Russell LE, Packer AM, et al. (2020). How many neurons are sufficient for perception of cortical activity? *eLife* 9:e58889.
- Chettih SN, Harvey CD (2019). Single-neuron perturbations reveal feature-specific competition in V1. *Nature* 567:334–340.
- Zhang Z, Russell LE, Packer AM, Gauld OM, Häusser M (2018). Closed-loop all-optical interrogation of neural circuits in vivo. *Nat Methods* 15:1037–1040.
- Gill JV, et al. (2020). Precise holographic manipulation of olfactory circuits reveals coding features determining perceptual detection. *Neuron* 108:382–393.
- Robinson NTM, et al. (2020). Targeted activation of hippocampal place cells drives memory-guided spatial behavior. *Cell* 183:1586–1599.
- Accanto N, et al. (2023). Ultrafast light targeting for high-throughput precise control of neuronal networks. *Nat Commun* 14:1888.
- Oldenburg IA, et al. (2024). Ensemble-specific 2P optogenetics (recurrent-circuit logic) — citation incomplete; verify.
- Deitch D, Rubin A, Ziv Y (2021). Representational drift in the mouse visual cortex. *Curr Biol* 31:4327–4339.
- Marks TD, Goard MJ (2021). Stimulus-dependent representational drift in primary visual cortex. *Nat Commun* 12:5169.

**Named in the framing but without a full citation in the project files** (verify before use):
Walker/Sinz "inception loops" (Walker EY et al. 2019, *Nat Neurosci*); Bashivan "neural
population control" (Bashivan P, Kar K, DiCarlo JJ 2019, *Science*); Yamins (goal-driven CNNs
predicting visual cortex, e.g. Yamins et al. 2014, *PNAS*).

**Software:** scikit-learn, SciPy, PyTorch/CUDA, CEBRA, Unity (VR environment), CaImAn-style
calcium extraction.
