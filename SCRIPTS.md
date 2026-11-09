# VR_analysis — script index

One line per script: what it is for. Compute lives in `functions/`; module details are in `CLAUDE.md`.
Each pipeline script also carries its PURPOSE line at the top of its docstring.

## Main pipeline: stimulus ↔ neural map

**Goal.** We want a two-way map between what the mouse sees in the virtual environment and what its
visual-cortex neurons do. The main deliverable is the encoding direction, and it should end in a
ranked list of up to 50 cells to target with holographic stimulation. VR_feature_search picks the
visual description of the monitor movie that best explains the neural data, scoring it by both encode
and decode so neither direction can game the result. The winning config goes into VR_ca_encoding,
which fits each neuron, tests it against a shuffle null, and splits its explained variance into
visual-only, behavior-only and shared parts. Those per-cell answers (receptive field, feature, latency,
significance) are the raw material for picking stimulation targets. VR_ca_cebra uses the same feature
config to confirm that the population carries that information in the decoding direction too. All
three read the same feature dict and the same saved convolutions, so a representation found in one
runs identically in the other two.

| Step | Script | Purpose |
|---|---|---|
| 1 | `VR_feature_search.py` | Tries many visual feature representations on one dataset (gabor magnitude/quadrature, FFT, motion energy, wavelet, raw pixels) and ranks them by how well they predict neurons (encode) and how well neurons predict them (decode). |
| 2 | `VR_ca_encoding.py` | Predicts each neuron's activity from the chosen visual features plus the mouse's own movement, then measures which cells are significantly driven, by what, at what lag, and with what receptive field. |
| 3 | `VR_ca_cebra.py` | Runs the reverse direction, decoding stimulus and behavior features from the neural population, as an independent check that the same features are represented. |

Shared knobs that must match across the three (else the conv_autosave store misses): `edge_min_frac`,
`conv_autosave_tag`, render 150×101, terrain off.

## Supporting the pipeline

| Script | Purpose |
|---|---|
| `VR_ensembles.py` | Builds one dataset's NMF ensembles from the shared ensemble config, plots them (cebra's ensemble plots), and saves them to the ensemble cache every script reloads. |
| `VR_feature_kernel_plot.py` | Draws the gabor/meng kernels to scale on a monitor frame to check filter sizes and sampling. |
| `VR_dataset_scan.py` | Fast across-dataset quality triage (cheap features, no render) to pick which sessions to run. |
| `VR_feature_gen.py` | Batch build + save feature caches / conv repos / PCA'd blocks across mice and datasets. |
| `VR_inspect_cache.py` | Load a feature cache and print its contents. |

## Feature analysis (no neural data or movie-only)

| Script | Purpose |
|---|---|
| `VR_feature_recon.py` | Reconstruct the monitor movie from feature blocks — each feature set's stimulus-coding ceiling. |
| `VR_feature_optim.py` | Sweep gabor/meng parameters to maximize movie-reconstruction R². |
| `VR_feature_viz_cache.py` | Visualize features from a block cache back in image space vs the original movie. |
| `VR_feature_viz_conv_repo.py` | Visualize / analyze a dense conv repo (per-filter activity, maps). |
| `VR_video_recon.py` | Synthetic monitor video rendering + terrain and all render-debugging cells. |

## Cross-dataset scans

| Script | Purpose |
|---|---|
| `VR_decode_analysis.py` | Compare saved decode results across datasets. |
| `VR_encode_scan.py` | Compute + save per-neuron encoding R² across mice × datasets × blocks × models. |
| `VR_encode_scan_plots.py` | Plot the encode scan: R² vs cortical location, distributions, comparisons. |
| `VR_tuning_scan.py` | Compute + save per-cell / per-ensemble feature tuning (STA) across datasets. |
| `VR_tuning_scan_plots.py` | Plot the tuning scan: population tuning, marginals, monitor maps. |
| `VR_feature_tuning.py` | Wide gabor/meng tuning scan per cell and ensemble on one dataset. |
| `VR_feature_tuning_progress.py` | Watch / score a running tuning scan from a separate process. |

## Neural-only / dim-red

| Script | Purpose |
|---|---|
| `VR_ca_analysis.py` | Calcium imaging + monitor vectorization (builds mov_data, vec_data). |
| `VR_ca_dimred.py` | Unsupervised neural dim-red (PCA + NMF sweep). |

## Registration

| Script | Purpose |
|---|---|
| `VR_register_2pfov.py` | Plots each dataset's 2p FOV outline on the widefield image from the VR_data.xlsx landmark pairs (tp_fov_locs ↔ wf_locs), per dataset and all datasets together. |
| `VR_register_2pfov_cells.py` | Project 2p cell ROIs onto the widefield map, colored by dataset or score. |

## Behavior / utilities / older

| Script | Purpose |
|---|---|
| `VR_analysis.py` | Behavior-only session analysis. |
| `VR_analysis_prelim.py` | Early Unity session exploration: loads one session's CSV + XML logs, plots DAQ voltage / distance / velocity and mouse position + rotation, and projects objects onto the camera FOV (predecessor of VR_analysis.py). |
| `VR_ball_calibration.py` | Calibrates the treadmill-ball motion sensors: loads the four sensor channels (S00–S11) + Unity movement data and compares their counts during known movement periods to derive sensor scaling/axes. |
| `retinotopic_mapping_analysis.py` | Widefield retinotopic mapping: aligns camera frames to stimulus trials via the voltage recording (photodiode/LED), builds per-direction trial-averaged baseline-subtracted movies (optional TIFF save), and takes the FFT amplitude/phase at the stimulus frequency for retinotopic maps. |
| `load_h5_data.py` | Quick look at a 2p movie .h5: loads the first N frames and optionally saves their mean image as a TIFF (unfinished plot cell). |
| `frame_data_xml_to_csv.py` | Converts each 2p acquisition XML in a mouse's preprocessing folder to a `_framedata.csv` of per-frame index / relative + absolute times / channel / page (skips existing). |
| `crop_videos_interactive.py` | Interactive video cropper (click two corners, crop whole video). |
