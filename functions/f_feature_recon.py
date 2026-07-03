# -*- coding: utf-8 -*-
"""
f_feature_recon.py — Chain-1: reconstruct the monitor MOVIE from feature blocks
(neuron-free groundwork for TODO #47/#66).

Measures each feature set's stimulus-coding FIDELITY: how much of the (perceptually
smoothed) monitor movie a linear decoder can rebuild from a feature block. No neural
data — this is the offline ceiling that grounds the neural decoding/encoding arms.

Pipeline: render movie → build a smoothed TARGET (mouse-acuity lowpass, so we don't
score features on sharp edges the animal can't resolve) → PCA the target → blocked-CV
ridge from each feature set (gabor magnitude / meng / mag+meng / pix) to the target PCs
→ out-of-fold variance-weighted R², overall and stratified by on-screen motion.

Why lowpass (not rank reduction) as the target: PCA drops low-VARIANCE directions,
not high-SPATIAL-FREQUENCY ones — a sharp MOVING edge has high variance so PCA keeps
it. A spatial Gaussian removes a defined SF band regardless of variance = the actual
"what the mouse can resolve" operation. Use 'rank' only with pix dropped as a
competitor (its ceiling is then 1 by construction).

Reuses f_blocked_cv_r2 (same CV machinery as the neural arms) and _pca_fit. Driven by
VR_feature_recon.py.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score

from f_visual_features import _pca_fit, f_apply_block_pca
from f_cebra_helpers import f_blocked_cv_r2


# ─────────────────────────────────────────────────────────────────────────────
# Frame subset — truncate a session so render+extract runs on N frames
# ─────────────────────────────────────────────────────────────────────────────

def f_truncate_session(session, n):
    """Shallow-copy `session` (f_load_session output) with every PER-FRAME array sliced
    to the first `n` imaging frames, so f_build_features renders + extracts only N frames.
    Needed because f_build_features forbids max_frames (it requires movie length == T_img)
    and the full movie (~96k×H×W float32) blows out RAM. Any ndarray whose first axis
    equals the session's T_img is truncated (mov_data/vec_data/lr_data dict arrays,
    clocks, terrain-phi vectors); everything else is passed through untouched."""
    T = session['T_img']
    n = int(min(n, T))

    def _tr(v):
        if isinstance(v, dict):
            return {k: _tr(x) for k, x in v.items()}
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            return v[:n]
        return v

    s = dict(session)
    for key in ('mov_data', 'lr_data', 'mov_data_eye', 'vec_data_l', 'vec_data_r'):
        if s.get(key) is not None:
            s[key] = _tr(s[key])
    for key in ('theta', 'mon_l_phi', 'mon_r_phi', 'beh_t', 'frame_t'):
        v = s.get(key)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            s[key] = v[:n]
    s['T_img'] = n
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Acuity → smoothing sigma from the monitor's angular geometry
# ─────────────────────────────────────────────────────────────────────────────

def f_deg_per_sample(cam_params, num_samp):
    """(deg_per_col, deg_per_row) for the rendered monitor — cols=lateral (hFOV),
    rows=vertical (vFOV). cam_params should carry hFOV_rad/vFOV_rad (f_load_session
    runs f_comp_FOV_adj); falls back to FOV_deg on the vertical axis if absent."""
    hfov = cam_params.get('hFOV_rad')
    vfov = cam_params.get('vFOV_rad')
    if hfov is None or vfov is None:
        vfov = np.deg2rad(cam_params.get('FOV_deg', 65.9))
        hfov = 2 * np.arctan2(np.sin(vfov / 2) * cam_params.get('aspect', 16 / 9),
                              np.cos(vfov / 2))
    return (np.degrees(hfov) / (num_samp - 1), np.degrees(vfov) / (num_samp - 1))


def f_acuity_sigma_px(cam_params, num_samp, cutoff_cpd=0.5):
    """Gaussian σ (row_px, col_px) that lowpasses at `cutoff_cpd` cycles/deg. A Gaussian's
    frequency response is exp(−2π²σ²f²); σ_deg = 1/(2π·f_c) puts the 1/e point at the
    cutoff. Converted to pixels per axis via the monitor's deg/sample (anisotropic — the
    render is ~0.66°/row vertical vs ~0.98°/col horizontal at num_samp=101, FOV 65.9°)."""
    deg_col, deg_row = f_deg_per_sample(cam_params, num_samp)
    sigma_deg = 1.0 / (2.0 * np.pi * cutoff_cpd)
    return (sigma_deg / deg_row, sigma_deg / deg_col)     # (row=vert, col=lat)


# ─────────────────────────────────────────────────────────────────────────────
# Target movie: lowpass / rank / raw  +  PCA for the regression target
# ─────────────────────────────────────────────────────────────────────────────

def f_lowpass_movie(movie, sigma_px, n_mon=2):
    """Per-frame spatial Gaussian lowpass, applied to EACH monitor separately (so the
    L|R seam of two_mon_frames doesn't bleed). `sigma_px` = scalar or (row, col).
    movie (T, H, W); W split into `n_mon` equal columns."""
    movie = np.asarray(movie, np.float32)
    if np.isscalar(sigma_px):
        sr = sc = float(sigma_px)
    else:
        sr, sc = float(sigma_px[0]), float(sigma_px[1])
    if sr <= 0 and sc <= 0:
        return movie.copy()
    T, H, W = movie.shape
    w = W // n_mon
    out = np.empty_like(movie)
    for m in range(n_mon):
        a, b = m * w, (m + 1) * w if m < n_mon - 1 else W
        out[:, :, a:b] = gaussian_filter(movie[:, :, a:b], sigma=(0, sr, sc),
                                         mode='nearest')
    return out


def f_make_target(movie, mode='lowpass', sigma_px=1.0, rank=50, n_target_pcs=50,
                  n_mon=2):
    """Build the reconstruction TARGET + its PCA.

    mode : 'lowpass' — acuity Gaussian (recommended; pix stays a fair competitor).
           'rank'    — pix-PCA reconstruction at `rank` comps (DROP pix as a competitor:
                       its R² is then 1 by construction). A weak proxy for smoothing —
                       keeps high-variance moving edges.
           'raw'     — full-resolution movie (reference).
    Then PCA the (flattened) target to `n_target_pcs` — the regression target and the
    inverse map for rendering reconstructed frames.

    Returns {'movie': (T,H,W) target frames, 'scores': (T, k), 'pca': PCA,
             'mode': str, 'hw': (H, W)}.
    """
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    if mode == 'lowpass':
        tgt = f_lowpass_movie(movie, sigma_px, n_mon=n_mon)
    elif mode == 'raw':
        tgt = movie.copy()
    elif mode == 'rank':
        flat = movie.reshape(T, -1)
        sc, pca = _pca_fit(flat, rank)
        # pca None (rank None/0/≥n_pixels) → target = raw movie.
        tgt = (movie.copy() if pca is None
               else pca.inverse_transform(sc).reshape(T, H, W).astype(np.float32))
    else:
        raise ValueError(f"target mode {mode!r} not in lowpass/rank/raw")
    scores, pca = _pca_fit(tgt.reshape(T, -1), n_target_pcs)
    return {'movie': tgt, 'scores': scores.astype(np.float32), 'pca': pca,
            'mode': mode, 'hw': (H, W)}


# ─────────────────────────────────────────────────────────────────────────────
# Feature-set assembly + per-frame motion score
# ─────────────────────────────────────────────────────────────────────────────

def f_assemble_features(built_blocks, spec):
    """Concatenate feature columns for one feature SET. `spec` = list of
    (block, prefix[, n_keep]): prefix None keeps the whole block, else channels whose
    name starts with it ('gabor_gabM' = magnitude PCs, 'gabor_gabP' = phase PCs). Optional
    n_keep keeps only the FIRST n_keep matched columns — nested PCA, so that's the exact
    rank-n_keep block (used by the n_pcs sweep). NaN→0. Returns (T, d)."""
    parts = []
    for entry in spec:
        blk, pref = entry[0], entry[1]
        n_keep = entry[2] if len(entry) > 2 else None
        if blk not in built_blocks:
            raise KeyError(f'{blk!r} not in built_blocks: {list(built_blocks)}')
        X, names = built_blocks[blk]['X'], built_blocks[blk]['names']
        if pref is None:
            cols = list(range(X.shape[1]))
        else:
            cols = [i for i, n in enumerate(names) if n.startswith(pref)]
            if not cols:
                raise ValueError(f'{blk!r}: no channels start with {pref!r} '
                                 f'(have e.g. {names[:3]})')
        if n_keep is not None:
            cols = cols[:n_keep]
        parts.append(X[:, cols])
    return np.nan_to_num(np.concatenate(parts, axis=1)).astype(np.float32)


def f_pc_sweep(built_blocks, block, prefix, target, cv_kw, k_list, lags=None,
               motion=None, verbose=True):
    """Reconstruction R² vs #PCs for ONE feature group — truncate (block, prefix) to the
    first k columns (nested PCA, no re-extraction), score each. k values above the built
    #PCs (the block's n_pca in visual_specs) are SKIPPED with a warning — raise n_pca and
    rebuild to sweep higher. Returns rows [{n_pcs, r2[, r2_hi, r2_lo]}]."""
    rows, skipped, n_avail = [], [], None
    for k in k_list:
        feat = f_assemble_features(built_blocks, [(block, prefix, k)])
        if feat.shape[1] < k:                    # ran out of built PCs
            n_avail = feat.shape[1]; skipped.append(k); continue
        r = f_recon_score(feat, target['scores'], cv_kw, lags=lags, motion=motion)
        row = {'n_pcs': feat.shape[1], 'r2': r['r2'], 'r2_pooled': r['r2_pooled']}
        if motion is not None:
            row['r2_hi'], row['r2_lo'] = r.get('r2_hi'), r.get('r2_lo')
        rows.append(row)
        if verbose:
            _s = (f"  hi={row['r2_hi']:+.3f} lo={row['r2_lo']:+.3f}"
                  if motion is not None else '')
            print(f"  n_pcs={row['n_pcs']:>3d}  R²(fold)={row['r2']:+.3f}  "
                  f"R²(pool)={row['r2_pooled']:+.3f}{_s}")
    if skipped and verbose:
        print(f"  ⚠ skipped n_pcs {skipped}: only {n_avail} PCs built for "
              f"{prefix or block!r} — raise its n_pca in visual_specs and REBUILD to go higher")
    return rows


def f_motion_score(built_blocks, block='pix'):
    """Per-frame on-screen motion = ‖Δ block['X']‖ over time (mirrors
    f_pick_motion_frames). Used to stratify reconstruction R² by motion."""
    X = np.nan_to_num(np.asarray(built_blocks[block]['X'], float))
    s = np.zeros(X.shape[0])
    s[1:] = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction scoring
# ─────────────────────────────────────────────────────────────────────────────

def _r2_vw(yt, yp):
    """Variance-weighted multi-output R² (chance ≈ 0; high-variance PCs dominate)."""
    if yt.shape[0] < 2:
        return np.nan
    return float(r2_score(yt, yp, multioutput='variance_weighted'))


def f_recon_score(feat_X, target_scores, cv_kw, lags=None, motion=None):
    """Blocked-CV ridge reconstruction of the target PCs from one feature set. Returns
    out-of-fold variance-weighted R² overall, and split by on-screen motion (median of
    `motion` over the OOF frames → hi/lo). `cv_kw` = f_blocked_cv_r2 kwargs
    (n_folds/embargo/decoder/ridge_alpha/standardize). Returns dict
    {r2, r2_hi, r2_lo, n_pts, res} (res = the full predictions dict for frame render)."""
    res = f_blocked_cv_r2(feat_X, target_scores, multioutput='variance_weighted',
                          return_predictions=True, lags=lags, **cv_kw)
    # r2 = mean of per-fold R² (the headline); r2_pooled = one R² over ALL OOF frames
    # (same estimator as the hi/lo split → a fair blend of them; immune to the per-fold
    # tiny-within-block-variance blowup that makes r2 go deeply negative on slow targets).
    out = {'r2': res['r2'], 'r2_pooled': _r2_vw(res['y_true'], res['y_pred']),
           'n_pts': int(res['idx'].size), 'res': res}
    # per-fold pooled R²: when fold_ids was used (leave-session-out), fold == session id,
    # so this is the per-session HELD-OUT R² (the cross-session generalization readout).
    _fold = res['fold']
    out['r2_per_fold'] = {int(g): _r2_vw(res['y_true'][_fold == g], res['y_pred'][_fold == g])
                          for g in np.unique(_fold) if (_fold == g).sum() > 1}
    if motion is not None:
        m = np.asarray(motion)[res['idx']]
        thr = np.median(m)
        hi, lo = m >= thr, m < thr
        out['r2_hi'] = _r2_vw(res['y_true'][hi], res['y_pred'][hi]) if hi.sum() > 1 else np.nan
        out['r2_lo'] = _r2_vw(res['y_true'][lo], res['y_pred'][lo]) if lo.sum() > 1 else np.nan
        out['motion_thresh'] = float(thr)
    return out


def f_recon_compare(built_blocks, feature_sets, target, cv_kw, lags=None,
                    motion=None, verbose=True):
    """Score every feature set (dict label → spec) against the target PCs. Returns
    {label: f_recon_score dict}. `target` = f_make_target output (uses target['scores'])."""
    results = {}
    for label, spec in feature_sets.items():
        feat_X = f_assemble_features(built_blocks, spec)
        r = f_recon_score(feat_X, target['scores'], cv_kw, lags=lags, motion=motion)
        r['dim'] = int(feat_X.shape[1])
        results[label] = r
        if verbose:
            _s = (f"  hi={r.get('r2_hi', np.nan):+.3f} lo={r.get('r2_lo', np.nan):+.3f}"
                  if motion is not None else '')
            print(f"  {label:10s}  dim={r['dim']:>4d}  R²(fold)={r['r2']:+.3f}  "
                  f"R²(pool)={r['r2_pooled']:+.3f}{_s}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-session pooling — one video decoder trained across datasets (Chain-1)
# ─────────────────────────────────────────────────────────────────────────────

def f_accumulate_session(session, cfg, n_frames, visual_specs):
    """RAW features + raw movie for ONE session (no PCA), for pooling across sessions.
    Truncates to n_frames, renders, extracts the requested visual blocks with n_pca=None
    (raw channels — aligned across sessions), returns the side-appropriate raw movie.
    Returns {raw_blocks: {block: build_visual_blocks entry (raw)}, movie: (T,H,W),
    side_tags, n}."""
    from f_pipeline import f_render_session
    from f_visual_features import build_visual_blocks
    s = f_truncate_session(session, n_frames)
    side = cfg.get('side', 'both')
    left, right, two = f_render_session(s, cfg)
    movies, side_tags = {'L': ([left], ['L']), 'R': ([right], ['R']),
                         'both': ([left, right], ['L', 'R'])}[side]
    movie = {'L': left, 'R': right, 'both': two}[side]
    raw_specs = {k: {**v, 'n_pca': None} for k, v in visual_specs.items()}
    raw_blocks = build_visual_blocks(movies, side_tags, s['beh_t'], s['frame_t'],
                                     s['pulse_delay'], raw_specs)
    return {'raw_blocks': raw_blocks, 'movie': movie, 'side_tags': side_tags,
            'n': int(s['frame_t'].size)}


def f_pool_sessions(accs, visual_specs, target_cfg, fs=1.0):
    """Pool per-session accumulations (f_accumulate_session outputs) into ONE dataset for a
    shared video decoder. Concatenates raw feature channels + raw movies across sessions,
    fits ONE shared feature PCA per block (grouped for gabor, via f_apply_block_pca) and ONE
    shared movie-target PCA (f_make_target on the concatenated movie). Verifies channel-name
    alignment across sessions. Returns {built_blocks (shared-PCA, same shape as single
    session), target, session_ids, motion, frame_t, side_tags, ns}."""
    side_tags = accs[0]['side_tags']
    blocks = list(accs[0]['raw_blocks'])
    for a in accs[1:]:
        if a['side_tags'] != side_tags:
            raise ValueError(f'side_tags differ across sessions: {a["side_tags"]} vs {side_tags}')
        for b in blocks:
            if a['raw_blocks'][b]['names'] != accs[0]['raw_blocks'][b]['names']:
                raise ValueError(f'raw channel names differ across sessions for block {b!r} '
                                 '— sessions must share grid/n_orient/wavelengths/stats/side')
    ns = [a['n'] for a in accs]
    session_ids = np.concatenate([np.full(n, i, int) for i, n in enumerate(ns)])

    pooled_movie = np.concatenate([a['movie'] for a in accs], axis=0)
    target = f_make_target(pooled_movie, n_mon=len(side_tags), **target_cfg)

    built = {}
    for b in blocks:
        X_raw = np.concatenate([a['raw_blocks'][b]['X'] for a in accs], axis=0)
        raw_names = accs[0]['raw_blocks'][b]['names']
        X, names, pca, pca_groups = f_apply_block_pca(X_raw, raw_names, b,
                                                      visual_specs[b].get('n_pca'))
        built[b] = {'X': X, 'names': names, 'pca': pca, 'raw_dim': X_raw.shape[1],
                    'raw_names': raw_names, 'pca_groups': pca_groups,
                    'params': accs[0]['raw_blocks'][b].get('params', {})}

    # motion from the shared movie-PC target, zeroed at session joins (Δ across a join is spurious)
    motion = np.zeros(session_ids.size)
    motion[1:] = np.linalg.norm(np.diff(target['scores'], axis=0), axis=1)
    for j in np.cumsum(ns)[:-1]:
        motion[j] = 0.0
    frame_t = np.arange(session_ids.size) / float(fs)     # synthetic (per-session fs shared)
    return {'built_blocks': built, 'target': target, 'session_ids': session_ids,
            'motion': motion, 'frame_t': frame_t, 'side_tags': side_tags, 'ns': ns}


def f_recon_diagnose(res, n_show=8, plot=True, title=''):
    """Break down ONE recon result's R² to explain a negative headline despite a good OOF
    trace: per-fold vs POOLED R², per-fold train vs test (overfit), and per-target-PC pooled
    R² (top by variance). The gap = slow target PCs have tiny WITHIN-block variance, so a
    small per-fold level offset → catastrophic per-fold R²; pooled over the full range is
    fine. Prints + optional bars. Returns the numbers."""
    yt, yp = res['y_true'], res['y_pred']
    folds = np.array(res['r2_folds'], float)
    folds_tr = np.array(res.get('r2_folds_train', [np.nan] * folds.size), float)
    pooled_vw = _r2_vw(yt, yp)
    pooled_ua = float(r2_score(yt, yp, multioutput='uniform_average'))
    per_pc = r2_score(yt, yp, multioutput='raw_values')
    var = yt.var(axis=0)
    order = np.argsort(var)[::-1]
    print(f'  headline (per-fold mean, var-wt): {np.nanmean(folds):+.3f}')
    print(f'  POOLED R²  var-wt: {pooled_vw:+.3f}   uniform: {pooled_ua:+.3f}')
    print(f'  per-fold test  R²: {np.round(folds, 2)}')
    print(f'  per-fold train R²: {np.round(folds_tr, 2)}')
    print('  top target PCs by variance — pooled R² each:')
    for c in order[:n_show]:
        print(f'    PC{c + 1:>2d}  var={var[c]:.3g}  poolR²={per_pc[c]:+.3f}')
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(range(folds.size), folds, color='C0', label='test')
        ax[0].plot(range(folds.size), folds_tr, 'o-', color='C1', label='train')
        ax[0].axhline(0, color='k', lw=0.6)
        ax[0].set_xlabel('fold'); ax[0].set_ylabel('R² (var-wt)')
        ax[0].legend(fontsize=8); ax[0].set_title('per-fold R²  (headline = mean)')
        top = order[:n_show]
        ax[1].bar(range(len(top)), per_pc[top], color='C2')
        ax[1].axhline(0, color='k', lw=0.6)
        ax[1].set_xticks(range(len(top)))
        ax[1].set_xticklabels([f'PC{c + 1}' for c in top], fontsize=7)
        ax[1].set_ylabel('pooled R²'); ax[1].set_title('per-PC pooled R²  (top by variance)')
        if title:
            fig.suptitle(title)
        fig.tight_layout()
    return {'per_fold': folds, 'per_fold_train': folds_tr, 'pooled_vw': pooled_vw,
            'pooled_ua': pooled_ua, 'per_pc': per_pc, 'var': var}


# ─────────────────────────────────────────────────────────────────────────────
# Rendering reconstructed frames (inverse target-PCA of the OOF predictions)
# ─────────────────────────────────────────────────────────────────────────────

def f_target_evr_curve(target_movie, max_pcs=150):
    """Cumulative explained-variance-ratio of the (smoothed) target-movie PCA vs #PCs — the
    fraction of movie info the first k target-PCs capture (= the low-rank reconstruction R²
    ceiling at rank k). Fit at `max_pcs` (> n_target_pcs) to see where the curve plateaus, so
    you can pick n_target_pcs. Returns (k array 1..K, cumulative_evr array)."""
    from sklearn.decomposition import PCA
    flat = np.asarray(target_movie).reshape(np.asarray(target_movie).shape[0], -1)
    k = int(min(max_pcs, flat.shape[0] - 1, flat.shape[1]))
    pca = PCA(n_components=k, random_state=42).fit(flat)
    return np.arange(1, k + 1), np.cumsum(pca.explained_variance_ratio_)


def f_plot_target_evr(k, cum_evr, marks=(0.90, 0.95, 0.99), n_target_pcs=None, suptitle=''):
    """Plot cumulative variance captured vs #target-PCs; annotate the #PCs reaching each
    `marks` fraction and the current n_target_pcs. Returns None."""
    import matplotlib.pyplot as plt
    k = np.asarray(k); cum_evr = np.asarray(cum_evr)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(k, cum_evr, '-', color='C0')
    for m in marks:
        j = int(np.searchsorted(cum_evr, m)) + 1
        if j <= k[-1]:
            ax.axhline(m, color='gray', lw=0.6, ls=':')
            ax.plot(j, cum_evr[j - 1], 'o', color='C3')
            ax.annotate(f'{m:.0%} @ {j}', (j, cum_evr[j - 1]), fontsize=8,
                        xytext=(4, -9), textcoords='offset points')
    if n_target_pcs is not None and n_target_pcs <= k[-1]:
        ax.axvline(n_target_pcs, color='C2', lw=1.0, ls='--',
                   label=f'n_target_pcs={n_target_pcs} ({cum_evr[n_target_pcs - 1]:.1%})')
        ax.legend(fontsize=8)
    ax.set_xlabel('# target PCs')
    ax.set_ylabel('cumulative variance captured (of smoothed movie)')
    ax.set_ylim(0, 1.01)
    if suptitle:
        ax.set_title(suptitle)
    fig.tight_layout()
    return None


def f_target_lowrank_at(target, frame_idx):
    """The LOW-RANK target the decoder actually aims at, at `frame_idx` = inverse-PCA of the
    target scores (the n_target_pcs reconstruction of the smoothed movie). Plotting it under
    target['movie'] shows the reconstruction CEILING — how much n_target_pcs itself drops."""
    H, W = target['hw']
    lr = target['pca'].inverse_transform(target['scores'][list(frame_idx)]).reshape(-1, H, W)
    return lr.astype(np.float32)


def f_recon_frames_at(res, target_pca, hw, frame_idx):
    """Reconstructed target frames at the requested imaging-frame indices. Picks the OOF
    rows in `res` whose original index is in `frame_idx`, inverse-transforms the target-PC
    predictions back to image space. Returns (frames (n,H,W), found_idx) — found_idx are
    the requested frames that were actually out-of-fold (others can't be shown OOF)."""
    H, W = hw
    idx = res['idx']
    pos = {int(f): i for i, f in enumerate(idx)}
    take, found = [], []
    for f in frame_idx:
        if int(f) in pos:
            take.append(pos[int(f)]); found.append(int(f))
    if not take:
        return np.zeros((0, H, W), np.float32), []
    recon = target_pca.inverse_transform(res['y_pred'][take]).reshape(-1, H, W)
    return recon.astype(np.float32), found


# ─────────────────────────────────────────────────────────────────────────────
# Plots (return None — Spyder-echo safe)
# ─────────────────────────────────────────────────────────────────────────────

def f_plot_recon_bars(results, suptitle='', motion=True):
    """Bar chart of reconstruction R² per feature set (overall, + hi/lo motion split
    when present). 'overall' uses the POOLED R² (global-normalized, one score over all
    OOF frames) — same estimator as hi/lo, so overall sits between them; avoids the
    per-fold-mean blowup on flat folds. Ordered as given. Returns None."""
    import matplotlib.pyplot as plt
    labels = list(results)
    r2 = [results[l].get('r2_pooled', results[l]['r2']) for l in labels]   # pooled headline
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 3, 4.5))
    has_m = motion and all('r2_hi' in results[l] for l in labels)
    if has_m:
        r2hi = [results[l]['r2_hi'] for l in labels]
        r2lo = [results[l]['r2_lo'] for l in labels]
        ax.bar(x - 0.25, r2, 0.25, label='overall (pooled)', color='C0')
        ax.bar(x, r2hi, 0.25, label='high motion', color='C3')
        ax.bar(x + 0.25, r2lo, 0.25, label='low motion', color='C2')
    else:
        ax.bar(x, r2, 0.5, color='C0')
    for xi, l in zip(x, labels):
        ax.annotate(f"{results[l]['dim']}d", (xi, 0), (0, -14), textcoords='offset points',
                    ha='center', va='top', fontsize=7, color='gray')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('OOF reconstruction R²  (feature → movie)')
    ax.legend(fontsize=8)
    if suptitle:
        ax.set_title(suptitle)
    fig.tight_layout()
    return None


def f_plot_pc_sweep(rows, series=('pooled', 'high', 'low'), suptitle=''):
    """Reconstruction R² vs #PCs (f_pc_sweep output). `series` picks which curves to draw:
    'pooled' (overall, global-normalized), 'per_fold' (overall, per-fold mean — can dip
    deeply negative on flat folds and wreck the y-scale, so it's OFF by default),
    'high'/'low' (motion split). e.g. series=('pooled','per_fold') or just ('pooled',).
    Returns None."""
    import matplotlib.pyplot as plt
    if not rows:
        print('no rows to plot'); return None
    spec = {'per_fold': ('r2',        'overall (per-fold)', 'C0', '-o'),
            'pooled':   ('r2_pooled', 'overall (pooled)',   'C4', '-s'),
            'high':     ('r2_hi',     'high motion',        'C3', '--o'),
            'low':      ('r2_lo',     'low motion',         'C2', '--o')}
    k = [r['n_pcs'] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s in series:
        if s not in spec:
            raise ValueError(f'unknown series {s!r}; pick from {list(spec)}')
        key, lbl, col, style = spec[s]
        if all(key in r for r in rows):
            ax.plot(k, [r[key] for r in rows], style, color=col, ms=4, label=lbl)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('# PCs kept'); ax.set_ylabel('OOF reconstruction R²  (feature → movie)')
    ax.legend(fontsize=8)
    if suptitle:
        ax.set_title(suptitle)
    fig.tight_layout()
    return None


def _bool_spans(mask):
    """Contiguous [start, stop) index spans where a boolean array is True."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        stops = stops + [m.size]
    return list(zip(starts, stops))


def f_plot_motion_traces(traces, motion, thresh=None, x=None, labels=None, suptitle=''):
    """Plot feature traces (e.g. pix PC1/PC2) with HIGH-motion frames shaded, so you see
    which frames the hi/lo split labels. `traces` = list of 1-D arrays (same length as
    `motion`); `thresh` = motion cutoff (None → median of `motion`, the split the scorer
    uses); `x` = x-axis (None → frame index). Returns None."""
    import matplotlib.pyplot as plt
    traces = [np.asarray(t) for t in traces]
    motion = np.asarray(motion)
    thr = float(np.median(motion)) if thresh is None else float(thresh)
    x = np.arange(motion.size) if x is None else np.asarray(x)
    labels = labels or [f'trace {i}' for i in range(len(traces))]
    spans = _bool_spans(motion >= thr)

    fig, axes = plt.subplots(len(traces) + 1, 1, figsize=(12, 1.8 * (len(traces) + 1)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, tr, lb in zip(axes, traces, labels):
        for a, b in spans:
            ax.axvspan(x[a], x[min(b, x.size - 1)], color='C3', alpha=0.15, lw=0)
        ax.plot(x, tr, color='C0', lw=0.8)
        ax.set_ylabel(lb, fontsize=8)
    for a, b in spans:
        axes[-1].axvspan(x[a], x[min(b, x.size - 1)], color='C3', alpha=0.15, lw=0)
    axes[-1].plot(x, motion, color='0.4', lw=0.7)
    axes[-1].axhline(thr, color='C3', ls='--', lw=0.8)
    axes[-1].set_ylabel('motion', fontsize=8); axes[-1].set_xlabel('frame')
    axes[0].plot([], [], color='C3', alpha=0.3, lw=6, label='high motion (shaded)')
    axes[0].legend(fontsize=8, loc='upper right')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return None


def f_plot_pred_overlay(res, cols=(0, 1, 2), labels=None, x=None, motion=None,
                        thresh=None, suptitle=''):
    """Overlay OOF prediction (red) on the true target (blue) for chosen target-PC columns,
    over time. `res` = a f_recon_score result's ['res'] (has y_true/y_pred/idx). `cols` =
    which target PCs to show. Optional `motion` shades high-motion OOF frames. `x` =
    per-frame x values indexed by res['idx'] (None → OOF frame index). Returns None."""
    import matplotlib.pyplot as plt
    yt, yp, idx = res['y_true'], res['y_pred'], res['idx']
    xt = idx if x is None else np.asarray(x)[idx]
    cols = [c for c in cols if c < yt.shape[1]]
    labels = labels or [f'target PC{c + 1}' for c in cols]
    spans = []
    if motion is not None:
        m = np.asarray(motion)[idx]
        thr = float(np.median(m)) if thresh is None else float(thresh)
        spans = _bool_spans(m >= thr)

    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 1.9 * len(cols)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, c, lb in zip(axes, cols, labels):
        for a, b in spans:
            ax.axvspan(xt[a], xt[min(b, xt.size - 1)], color='C3', alpha=0.12, lw=0)
        r = np.corrcoef(yt[:, c], yp[:, c])[0, 1]
        ax.plot(xt, yt[:, c], color='C0', lw=0.9, label='true')
        ax.plot(xt, yp[:, c], color='C3', lw=0.9, alpha=0.8, label='pred')
        ax.set_ylabel(f'{lb}\nr={r:+.2f}', fontsize=8)
    axes[0].legend(fontsize=8, loc='upper right', ncol=2)
    axes[-1].set_xlabel('OOF frame')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return None
