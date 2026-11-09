# f_encode_scan.py — cross-dataset per-neuron ENCODING sweep (created 2026-07-17).
#
# Loop many mice × datasets and, for EACH neuron, score how well it regresses onto each feature
# BLOCK (agg / self_mot / pix / gaborS+mengS ...) under each MODEL (ridge / cca ...). Results go into
# ONE tag-versioned, RESUMABLE joblib store (mirror of f_feature_scan): rewritten atomically after
# every combo, skip-done keyed by (mouse, dset, block, model) so a re-run only computes new combos.
# The store keeps ONLY neuron indices + per-combo scores + dataset-identity params (NO locations) —
# VR_encode_scan_plots.py loads locations (footprint centroids + widefield landmarks) separately and
# joins by dset_name + neuron id. Compute reuses f_encoding.f_encode_per_neuron (per-neuron OOF R²/r,
# model via f_decoding.make_decoder) and f_feature_source.f_resolve_features (build/load features).
#
# Split of labor: f_encode_prepare (build Y_enc + features + identity ONCE per dataset) →
# f_encode_one (one block × model fit) so the driver controls the save cadence. Location helpers
# (centroids / 4-corner homography / widefield coords) live here too, used by script 2.

import os
import re
import json

import numpy as np
import pandas as pd
import joblib

# FOV↔widefield registration + neuron-location + fov-scatter helpers live in f_fov_register (shared with
# VR_register_2pfov_cells.py); re-exported here so existing es.f_* call sites keep working.
from f_fov_register import (f_footprint_centroids, f_fov_homography, f_apply_homography,       # noqa: F401
                            f_load_fov_landmarks, f_neuron_widefield_coords, f_plot_neuron_scores_fov,
                            f_plot_neuron_field_fov, f_rot_for_plot, f_rot_xy)


# ─────────────────────────────────────────────────────────────────────────────
# Store — resumable, atomic, tag-versioned (mirror of f_feature_scan)
# ─────────────────────────────────────────────────────────────────────────────

def _safe(name):
    """Filesystem-safe token (matches f_feature_cache._safe)."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))


def f_encode_scan_path(root, tag):
    """Canonical store path — <root>/encode_scan/<tag>_encode_scan.joblib. `root` is a SHARED
    cross-mouse dir (e.g. feat_root without the mouse suffix); the encode_scan/ folder sits parallel
    to cache/ and conv_repo/. Caller makes the directory."""
    return os.path.join(str(root), 'encode_scan', f'{_safe(tag)}_encode_scan.joblib')


def _enc_key(mouse, dset_name, source, block, model):
    """Combo identity string (skip-done key). `source` = unit source ('cells' | 'ensembles')."""
    return f'{_safe(mouse)}|{_safe(dset_name)}|{source}|{block}|{model}'


def _dkey(mouse, dset_name):
    """Dataset identity string (key into store['datasets'])."""
    return f'{_safe(mouse)}|{_safe(dset_name)}'


def _save_atomic(path, state):
    """Write the whole store to <path>.tmp then os.replace — crash never leaves a partial file, a
    concurrent reader always sees a complete generation."""
    tmp = str(path) + '.tmp'
    joblib.dump(state, tmp)
    os.replace(tmp, path)


def f_new_store(config, target=()):
    """Fresh store dict. `config` = the run knobs (lags/models/blocks/…); `target` = the full planned
    list of (mouse, dset_name, block, model) tuples for progress."""
    return {'config': dict(config), 'results': {}, 'datasets': {}, 'target': [tuple(t) for t in target]}


def f_encode_scan_load(path):
    """Load the store (a single atomic snapshot — safe mid-run) or None if it doesn't exist yet."""
    return joblib.load(path) if os.path.exists(path) else None


def f_store_put(store, mouse, n_dset, dset_name, source, block, model, result, labels):
    """Insert one combo's result into the store (does NOT save). `result` = f_encode_one output;
    `source` = 'cells' | 'ensembles'; `labels` = unit id per row (cells → true est1 id, ensembles →
    ensemble index) — length matches r2/r/keep."""
    store['results'][_enc_key(mouse, dset_name, source, block, model)] = {
        'mouse': mouse, 'n_dset': int(n_dset), 'dset_name': dset_name, 'source': source,
        'block': block, 'model': model, 'labels': np.asarray(labels),
        'r2': np.asarray(result['r2'], np.float32),
        'r':  np.asarray(result['r'],  np.float32),
        'keep': np.asarray(result['keep'], bool)}


def f_ensemble_record(ens_table, n_ens):
    """Compact per-dataset ensemble record for the store (small — the store is rewritten each combo,
    so no full scores). Arrays indexed by ensemble id 0..n_ens-1: sig (bool), cv_r2, members (true
    cell-id list). Built from f_ensemble_stats' ens_table (which is sorted by cv_r2, not by id)."""
    sig = np.zeros(n_ens, bool); cvr2 = np.full(n_ens, np.nan); members = [[] for _ in range(n_ens)]
    for _, row in ens_table.iterrows():
        i = int(row['ensemble'])
        if 0 <= i < n_ens:
            sig[i] = bool(row.get('sig', False)); cvr2[i] = float(row.get('cv_r2', np.nan))
            members[i] = list(row.get('cells', []))
    return {'n_ens': int(n_ens), 'sig': sig, 'cv_r2': cvr2, 'members': members}


def f_store_set_dataset(store, rec):
    """Record a dataset's identity + neuron index (cell_ids) for verification/join in script 2."""
    store['datasets'][_dkey(rec['mouse'], rec['dset_name'])] = dict(rec)


def f_encode_scan_progress(path_or_store):
    """Progress snapshot (neural-free; safe mid-run). n_done / n_target / n_remaining + per-source /
    per-block / per-model combo counts + which datasets have any results. Accepts a store FILE PATH
    (loaded from disk — safe mid-run via the atomic writes) OR an in-memory store dict."""
    st = path_or_store if isinstance(path_or_store, dict) else f_encode_scan_load(path_or_store)
    if st is None:
        return {'exists': False, 'n_done': 0, 'n_target': 0, 'n_remaining': 0,
                'by_block': {}, 'by_model': {}, 'by_source': {}, 'datasets': [], 'keys': []}
    res = st['results']
    tgt = st.get('target', [])
    tkeys = {_enc_key(*t) for t in tgt}
    by_block, by_model, by_source = {}, {}, {}
    for r in res.values():
        by_block[r['block']] = by_block.get(r['block'], 0) + 1
        by_model[r['model']] = by_model.get(r['model'], 0) + 1
        _s = r.get('source', 'cells'); by_source[_s] = by_source.get(_s, 0) + 1
    return {'exists': True, 'n_done': len(res), 'n_target': len(tkeys),
            'n_remaining': len(tkeys - set(res)),
            'by_block': by_block, 'by_model': by_model, 'by_source': by_source,
            'datasets': sorted({(r['mouse'], r['dset_name']) for r in res.values()}),
            'keys': sorted(res)}


def f_encode_scan_table(store):
    """Tidy per-unit table from the store: one row per (dataset, source, unit, block, model). `unit` =
    the stored label (cells → true est1 id; ensembles → ensemble index). `sig` = ensemble significance
    (from the dataset's ensemble record; NaN for cells). r2/r NaN where the unit was dropped
    (keep=False). Script 2 joins the CELLS rows to footprint centroids by (dset_name, unit)."""
    rows = []
    for r in store['results'].values():
        rec = store['datasets'].get(_dkey(r['mouse'], r['dset_name'])) or {}
        source = r.get('source', 'cells')
        labels = r.get('labels')
        if labels is None:                               # legacy: cells labelled by cell_ids
            labels = (np.asarray(rec['cell_ids'], int) if rec.get('cell_ids') is not None
                      else np.arange(len(r['r2'])))
        ens = rec.get('ens') if source == 'ensembles' else None
        for i, lab in enumerate(labels):
            _sig = (bool(ens['sig'][int(lab)]) if ens is not None and int(lab) < ens['n_ens'] else np.nan)
            rows.append({'mouse': r['mouse'], 'n_dset': r['n_dset'], 'dset_name': r['dset_name'],
                         'source': source, 'unit': int(lab), 'block': r['block'], 'model': r['model'],
                         'r2': float(r['r2'][i]), 'r': float(r['r'][i]), 'keep': bool(r['keep'][i]),
                         'sig': _sig})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Compute — prepare once per dataset, then one fit per (block, model)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_KW = {'ridge': {'ridge_alpha': 1e3}, 'ridgecv': {}, 'cca': {'n_components': 20},
                    'pls': {'n_components': 20}, 'knn': {'k_nn': 15}}


def f_encode_prepare(session, blocks, *, sigma_frames=6, plan=None, build_cfg=None, src=None,
                     whiten=False, combine=None, verbose=True):
    """Build the encoder inputs for ONE session ONCE: the neural target Y_enc and the feature blocks.

    Neural proxy = drop-silent → smooth → peak-normalize, NO hclust so rows stay in est1['A'] order
    (matches f_dataset_scan). Features via f_resolve_features(only=blocks). Returns a dict:
      Y_enc       (T, n_active) float32   — encoder target (cells in est1-row order)
      built_blocks, frame_t, fs
      n_active, cell_ids (int, indices into est1 rows)
      rec         dataset identity + cell_ids (→ store['datasets'])
    Trims Y_enc / blocks to a common T if the clocks differ by a frame.

    `whiten` → load-time whiten of PCA blocks (÷√eigenvalue). `combine` = {new_name: {'blocks':[...],
    'n_pca':…, 'balance':bool, 'zscore':bool}} composes a joint block from ALREADY-LOADED per-input PCA
    caches (concat scores + one PCA, NO reconvolution) — use for 'gaborS+mengS' when its inputs are the
    loaded gaborS/mengS caches. Composite names are NOT resolved from disk; their source blocks are
    loaded instead."""
    from f_feature_source import (f_resolve_features, f_default_feature_plan, f_default_build_cfg,
                                  f_combine_loaded_pca)
    from f_utils import f_gauss_smooth
    import f_decoding as dec

    est1 = session['est1']
    mouse, n_dset = session['mouse_id'], session['n_dset']
    dset_name = est1['dset_name']

    from f_utils import f_neural_rate                                # the ONE spikes→rate recipe
    S = np.asarray(est1['S'], float)
    S_smn, active = f_neural_rate(S, sigma_frames=sigma_frames, peak_norm=True)
    cell_ids = np.flatnonzero(active)
    Y_enc = S_smn.T.astype(np.float32)                              # (T_img, n_active)

    plan = plan if plan is not None else f_default_feature_plan()
    build_cfg = build_cfg if build_cfg is not None else f_default_build_cfg()
    src = src or {}
    combine = combine or {}
    combine_srcs = {b for spec in combine.values() for b in spec['blocks']}
    # RESOLVE requested blocks MINUS composite names, PLUS the composites' source blocks (loaded, then
    # combined below) — so gaborS+mengS is composed from the caches, not rebuilt.
    need = sorted((set(blocks) - set(combine)) | combine_srcs)
    built_blocks, frame_t, side_tags, _ = f_resolve_features(
        session, plan, build_cfg, dset_name=dset_name, only=need,
        render_verbose=False, whiten=whiten, verbose=verbose, **src)
    if combine:
        f_combine_loaded_pca(built_blocks, combine, verbose=verbose)

    T = min(int(Y_enc.shape[0]), int(len(frame_t)))                # align clocks (defensive)
    Y_enc = Y_enc[:T]
    for b in built_blocks.values():
        if b['X'].shape[0] != T:
            b['X'] = b['X'][:T]
    frame_t = np.asarray(frame_t).ravel()[:T]
    fs = float(dec.f_imaging_fs(frame_t))

    rec = {'mouse': mouse, 'n_dset': int(n_dset), 'dset_name': dset_name,
           'n_cells_total': int(S.shape[0]), 'n_active': int(active.sum()),
           'cell_ids': cell_ids.astype(int), 'T': int(T), 'fs': fs,
           'feature_list': list(blocks), 'sigma_frames': int(sigma_frames)}
    return {'Y_enc': Y_enc, 'built_blocks': built_blocks, 'frame_t': frame_t, 'fs': fs,
            'n_active': int(active.sum()), 'cell_ids': cell_ids.astype(int), 'rec': rec}


def f_encode_one_cfg(built_blocks, block, Y_enc, enc_cfg, frame_t, model=None, model_kw=None):
    """One block × model per-unit encode with the SHARED encoder config (f_encoding.f_run_encoder,
    model = VR_ca_cebra's). `model` overrides enc_cfg['decoder']; `model_kw` = extra encoder keys
    for that model (e.g. {'n_components': 30} for cca). feature_block = [block], n_feature_pcs = None
    (blocks arrive already reduced by the feature plan). Same return as f_encode_one:
    {'block','model','r2' (n_units,),'r' (n_units,),'keep' (n_units,) bool}, NaN where dropped."""
    from f_encoding import f_run_encoder
    cfg = {**enc_cfg, 'feature_block': [block], 'n_feature_pcs': None, **(model_kw or {})}
    if model is not None:
        cfg['decoder'] = model
    n = int(Y_enc.shape[1])
    T = min(int(Y_enc.shape[0]), int(np.asarray(frame_t).size))
    out = f_run_encoder(cfg, built_blocks, np.asarray(Y_enc)[:T], np.asarray(frame_t).ravel()[:T])
    keep = np.asarray(out['res']['keep_cells'], bool)
    r2 = np.full(n, np.nan, np.float32); r2[keep] = out['res']['r2_per_cell']
    r = np.full(n, np.nan, np.float32); r[keep] = out['res']['r_per_cell']
    return {'block': block, 'model': cfg['decoder'], 'r2': r2, 'r': r, 'keep': keep}


def f_encode_one(built_blocks, block, Y_enc, lags, model, *, model_kw=None, n_folds=5,
                 embargo=0, fold_scheme='interleaved', standardize=True):
    """One block × model per-neuron encode fit. Runs f_encode_per_neuron with the single `block` as
    the design and scatters its kept-cell OOF R²/r back to full length (n_active), NaN where dropped.
    Returns {'block','model','r2' (n_active,),'r' (n_active,),'keep' (n_active,) bool}."""
    from f_encoding import f_encode_per_neuron
    n_active = int(Y_enc.shape[1])
    mk = dict(DEFAULT_MODEL_KW.get(model, {}))
    if model_kw:
        mk.update(model_kw)
    res = f_encode_per_neuron(built_blocks, [block], Y_enc, lags, decoder=model,
                              n_folds=n_folds, embargo=embargo, standardize=standardize,
                              fold_scheme=fold_scheme, **mk)
    keep = np.asarray(res['keep_cells'], bool)
    r2 = np.full(n_active, np.nan, np.float32); r2[keep] = res['r2_per_cell']
    r  = np.full(n_active, np.nan, np.float32); r[keep]  = res['r_per_cell']
    return {'block': block, 'model': model, 'r2': r2, 'r': r, 'keep': keep}


# Neuron footprint locations, the FOV↔widefield homography, and the fov score-scatter are re-exported
# from f_fov_register (imported at the top) — shared with VR_register_2pfov_cells.py.


# ─────────────────────────────────────────────────────────────────────────────
# Plots (score tables)
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_score_hist(table, *, block=None, model=None, dset_tag='', bins=40):
    """R² distribution over neurons, one step-histogram per (block, model) present in `table` (or the
    filtered subset). Returns None."""
    import matplotlib.pyplot as plt
    df = table.copy()
    if block is not None:
        df = df[df['block'] == block]
    if model is not None:
        df = df[df['model'] == model]
    fig, ax = plt.subplots(figsize=(6, 4))
    for (b, m), g in df.groupby(['block', 'model']):
        ax.hist(g['r2'].dropna(), bins=bins, histtype='step', density=True, label=f'{b}·{m}')
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel('R²'); ax.set_ylabel('density'); ax.legend(fontsize=7)
    ax.set_title(f'{dset_tag} — per-neuron R²', fontsize=9)
    fig.tight_layout()
    return None


def f_plot_block_model_compare(table, *, x=('block', 'agg'), y=('block', 'gaborS+mengS'), model='ridge',
                               dset_tag=''):
    """Scatter per-neuron R² of one (field,value) vs another for a fixed model — e.g. agg vs
    gaborS+mengS, or ridge vs cca. `x`/`y` = (column, value) selectors; if comparing models, set
    x=('model','ridge'), y=('model','cca') and pass block via `model` unused. Returns None."""
    import matplotlib.pyplot as plt
    fx, vx = x; fy, vy = y
    dfx = table[(table[fx] == vx) & ((table['model'] == model) if fx == 'block' else True)]
    dfy = table[(table[fy] == vy) & ((table['model'] == model) if fy == 'block' else True)]
    key = ['mouse', 'dset_name', 'source', 'unit']
    m = dfx.merge(dfy, on=key, suffixes=('_x', '_y'))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(m['r2_x'], m['r2_y'], s=8, alpha=0.4)
    lim = [np.nanmin([m['r2_x'].min(), m['r2_y'].min(), 0]),
           np.nanmax([m['r2_x'].max(), m['r2_y'].max(), 0.1])]
    ax.plot(lim, lim, 'k--', lw=0.6)
    ax.set_xlabel(f'{vx} R²'); ax.set_ylabel(f'{vy} R²')
    ax.set_title(f'{dset_tag} — {vx} vs {vy} ({model})', fontsize=9)
    fig.tight_layout()
    return None
