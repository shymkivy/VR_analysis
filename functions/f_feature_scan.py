# -*- coding: utf-8 -*-
"""
f_feature_scan.py — wide feature-parameter SCAN engine (generation) + pluggable SCORER.

For a WIDE gabor / motion-energy grid, render the movie ONCE (caller) then convolve filter-by-filter
(GPU, f_conv_gpu) and immediately reduce each to its WHOLE-MONITOR magnitude time series — the tuning
regressor — never persisting the hundreds-of-GB dense maps. The two halves are deliberately split so
the SAME generation can feed different scorers:

  GENERATION  f_scan_conv_activity(movies, side_tags, specs, ...) -> (X_stim (T,n_filt), filt_df)
  SCORING     f_score_scan(X_stim, filt_df, neural, scorer=..., ...) -> {source: result}

`scorer='tuning'` (v1) reverse-correlates each unit (cells / ensembles / ...) against the per-filter
regressors → preferred orientation/wavelength/tfreq (reuses f_tuning). `scorer='encode'|'decode'`
are documented extension points (stubbed) that will treat X_stim as a feature block and reuse
f_dataset_scan.f_score_representations / f_encoding.f_run_encoder / f_decoding.f_run_decode_sweep —
so encode/decode performance scans drop in WITHOUT changing generation.

Efficiency: gabor filters batch as conv2d out-channels (`filt_batch` = the GPU resource knob); meng
groups by (orientation, wavelength) and shares the spatial gabor across its temporal freqs. X_stim is
tiny (T × n_filt), so all filter columns live in RAM and STA/ridge run once at the end.

Created 2026-07-16.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

import f_conv_repo as cr


def _batches(seq, n):
    for i in range(0, len(seq), max(1, int(n))):
        yield seq[i:i + int(n)]


def _reduce_np(x, reduce):
    """Whole-monitor reduce of a dense (T,H,W) map → (T,) — CPU mirror of f_conv_gpu._reduce_space."""
    if reduce == 'mean':
        return x.mean(axis=(1, 2))
    if reduce == 'rms':
        return np.sqrt((x * x).mean(axis=(1, 2)))
    return np.abs(x).mean(axis=(1, 2))


def _transform(x, kind):
    """Per-regressor value map (meng ENERGY→amplitude etc.) — mirrors f_tuning._reg_transform."""
    if not kind:
        return x
    x = np.asarray(x, float)
    if kind == 'sqrt':
        return np.sqrt(np.clip(x, 0, None))
    if kind == 'signed_sqrt':
        return np.sign(x) * np.sqrt(np.abs(x))
    if kind == 'log1p':
        return np.log1p(np.clip(x, 0, None))
    if kind == 'signed_log1p':
        return np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(f"unknown transform {kind!r}")


def _combine(side_ts, mode):
    """Combine per-side time series (list of (T,) or (T,b)) → mean across sides | first side only."""
    return np.mean(side_ts, axis=0) if mode == 'mean' else side_ts[0]


def _gabor_activity_cpu(movie, specs, reduce, sigma_scale, gamma, n_workers=1, return_spatial=False):
    """CPU fallback: dense gabor magnitude per spec (f_conv_repo.f_gabor_map) → whole-monitor ts
    (T, len(specs)); with return_spatial also the per-filter time-mean map (len(specs), H, W).
    Optional thread pool (scipy FFT releases the GIL)."""
    def _one(sp):
        d = cr.f_gabor_map(movie, sp['theta'], sp['wavelength'], sigma_scale, gamma)
        return (_reduce_np(d, reduce), d.mean(0) if return_spatial else None)
    if n_workers and n_workers > 1 and len(specs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
            res = list(ex.map(_one, specs))
    else:
        res = [_one(sp) for sp in specs]
    ts = np.column_stack([r[0] for r in res])
    sp = np.stack([r[1] for r in res]) if return_spatial else None
    return ts, sp


def _meng_activity_cpu(movie, theta, wavelength, tfreqs, reduce, temporal_support, output,
                       sigma_scale, gamma, return_spatial=False):
    """CPU fallback: dense meng opponent per tfreq (f_conv_repo.f_meng_map) → whole-monitor ts dict;
    with return_spatial each value is (ts, spat (H, W)) = time-mean |opponent|. `temporal_support` =
    scalar (all tf) or {tf: support} (per-tf, constant-cycle)."""
    out = {}
    for tf in tfreqs:
        _sup = int(temporal_support[float(tf)] if isinstance(temporal_support, dict) else temporal_support)
        m = cr.f_meng_map(movie, theta, wavelength, tf, _sup, None,
                          sigma_scale, gamma, output)
        if output == 'opponent':
            ts = _reduce_np(m, reduce)
            out[float(tf)] = (ts, np.abs(m).mean(0)) if return_spatial else ts
        else:
            out[float(tf)] = (_reduce_np(m[0], reduce), _reduce_np(m[1], reduce))
    return out


def _g(v):
    return f'{v:g}'


def _is_nan_tf(tf):
    return tf is None or (isinstance(tf, float) and np.isnan(tf))


def _scan_key(row):
    """Canonical filter-identity string — the resume / dedup / skip key (independent of column order)."""
    k = f"{row['kind']}_o{int(row['odeg'])}_wl{_g(row['wavelength'])}"
    if row['kind'] == 'meng' and not _is_nan_tf(row.get('tfreq', np.nan)):
        k += f"_tf{_g(float(row['tfreq']))}"
    return k


def _row_from_spec(s):
    """Filter-identity row from an f_conv_filter_specs entry (theta→odeg, tfreq for meng)."""
    return {'kind': s['kind'], 'odeg': cr._odeg(s['theta']), 'wavelength': s['wavelength'],
            'tfreq': (float(s['tfreq']) if s['kind'] == 'meng' else np.nan)}


def _gabor_batch_cols(mov, side_tags, batch, backend, dev, reduce, combine_sides,
                      sigma_scale, gamma, chunk_t, n_workers, spatial=False):
    """Convolve+reduce a batch of gabor specs (both sides, combined) → list of (col (T,), row). With
    `spatial`, row['spatial'] = the per-filter time-mean map, sides hstacked L‖R (H, n_sides·W)."""
    ts_sides, sp_sides = [], []
    for side in side_tags:
        if backend == 'gpu':
            import f_conv_gpu as cg
            r = cg.f_gabor_activity_gpu(mov[side], batch, reduce=reduce, sigma_scale=sigma_scale,
                                        gamma=gamma, chunk_t=chunk_t, device=dev, return_spatial=spatial)
        else:
            r = _gabor_activity_cpu(mov[side], batch, reduce, sigma_scale, gamma, n_workers, spatial)
        ts, sp = r if spatial else (r if backend == 'gpu' else r[0], None)
        ts_sides.append(ts); sp_sides.append(sp)
    ts = _combine(ts_sides, combine_sides)
    out = []
    for j, sp_spec in enumerate(batch):
        row = {'kind': 'gabor', 'odeg': cr._odeg(sp_spec['theta']), 'wavelength': sp_spec['wavelength'],
               'tfreq': np.nan}
        for k in ('sf_cyc_deg', 'wl_deg'):           # physical units (if built via f_conv_filter_specs_phys)
            if k in sp_spec:
                row[k] = sp_spec[k]
        if spatial:
            row['spatial'] = np.hstack([ss[j] for ss in sp_sides]).astype(np.float32)  # L‖R
        out.append((np.array(ts[:, j], np.float32), row))    # COPY (not a view → batch ts frees)
    return out


def _meng_group_cols(mov, side_tags, th, wl, gspecs, out, backend, dev, reduce, combine_sides,
                     transform_meng, sigma_scale, gamma, chunk_t, spatial=False):
    """Convolve+reduce one (theta,wl) meng group over its tfreqs (spatial conv shared) → list of
    (col (T,), row), transform applied (energy→amplitude). `gspecs` = the group's specs (carry the
    physical sf_cyc_deg/wl_deg/tf_hz + per-tf temporal_support). PER-TF support (constant-cycle bank)
    is passed as {tf: support}; spatial conv is still shared. With `spatial`, row['spatial'] =
    time-mean |opponent| per tfreq, sides hstacked L‖R."""
    tfs = [float(s['tfreq']) for s in gspecs]
    tf2spec = {float(s['tfreq']): s for s in gspecs}
    tsup_map = {float(s['tfreq']): int(s.get('temporal_support', 11)) for s in gspecs}
    per_side = []
    for side in side_tags:
        if backend == 'gpu':
            import f_conv_gpu as cg
            d = cg.f_meng_activity_gpu(mov[side], th, wl, tfs, reduce=reduce, temporal_support=tsup_map,
                                       output=out, sigma_scale=sigma_scale, gamma=gamma,
                                       chunk_t=chunk_t, device=dev, return_spatial=spatial)
        else:
            d = _meng_activity_cpu(mov[side], th, wl, tfs, reduce, tsup_map, out, sigma_scale, gamma, spatial)
        per_side.append(d)
    cols = []
    for tf in tfs:
        _ts_sides = [(ps[tf][0] if spatial else ps[tf]) for ps in per_side]
        ts = _transform(_combine(_ts_sides, combine_sides), transform_meng)
        row = {'kind': 'meng', 'odeg': cr._odeg(th), 'wavelength': wl, 'tfreq': float(tf)}
        for k in ('sf_cyc_deg', 'wl_deg', 'tf_hz'):
            if k in tf2spec[tf]:
                row[k] = tf2spec[tf][k]
        if spatial:
            row['spatial'] = np.hstack([ps[tf][1] for ps in per_side]).astype(np.float32)  # L‖R
        cols.append((np.asarray(ts, np.float32), row))
    return cols


def _save_scan_atomic(scan_file, state):
    """Rewrite the whole scan bundle atomically (temp + os.replace) so a crash never corrupts it and
    a concurrent reader always sees a consistent snapshot."""
    tmp = str(scan_file) + '.tmp'
    joblib.dump(state, tmp)
    os.replace(tmp, scan_file)


def _assemble(state, specs, standardize):
    """Build (X_stim, filt_df) from a scan state — all stored columns (specs=None) or the subset
    matching `specs`. Columns sorted by (odeg, wavelength, tfreq); optional per-column z-score."""
    cd = state['cols']
    rows = (list(cd.values()) if specs is None
            else [cd[k] for k in (_scan_key(_row_from_spec(s)) for s in specs) if k in cd])
    if not rows:
        raise ValueError('f_scan: no stored columns for the requested specs')
    _keys = ['kind', 'odeg', 'wavelength', 'tfreq']
    _keys += [k for k in ('sf_cyc_deg', 'wl_deg', 'tf_hz') if any(k in r for r in rows)]
    filt_df = pd.DataFrame([{k: r.get(k, np.nan) for k in _keys} for r in rows])
    X = np.column_stack([r['col'] for r in rows]).astype(np.float32)
    order = filt_df.sort_values(['odeg', 'wavelength', 'tfreq'], na_position='last').index.to_numpy()
    X = X[:, order]
    filt_df = filt_df.iloc[order].reset_index(drop=True)
    if standardize:
        mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
        X = ((X - mu) / sd).astype(np.float32)
    return X, filt_df


def f_scan_path(features_dir, dset_name, gen_tag, lat_samp, vert_samp, sample_ds=None):
    """Canonical scan-file path — <features_dir>/scan/<dset>_<gen_tag>_<lat>x<vert>[_ds<..>]_scan.joblib
    (the `scan/` folder sits PARALLEL to conv_repo/). Both the scan script and the progress/analysis
    script build the path from the SAME gen_tag + resolution so they point at one file. Returns the
    path; the caller makes the dir."""
    ds = '' if sample_ds is None else f'_ds{sample_ds}'
    fn = f'{dset_name}_{gen_tag}_{int(lat_samp)}x{int(vert_samp)}{ds}_scan.joblib'
    return os.path.join(str(features_dir), 'scan', fn)


def f_nyquist_limits(cam_params, lat_samp, vert_samp, fs, sample_ds=None, temporal_support=11,
                     sf_ref=0.1, verbose=True):
    """Resolvable SPATIAL / TEMPORAL limits for a conv scan at the given render × downsample + imaging
    fs — what sf_cyc_deg (/ wavelength) and tf_hz you can use. Uses the EFFECTIVE deg/px (render ×
    sample_ds). Also the implied motion SPEED (deg/s = tf_hz / sf_cyc_deg) for a meng unit at the
    reference spatial freq `sf_ref` (a grating drifts at speed = tf/sf along the orientation normal).
    Returns a dict {deg_per_px, sf_max (Nyquist λ=2px), sf_practical (λ≥4px), sf_min (λ=full monitor),
    wl_min_deg_nyq/_practical, tf_max_hz (fs/2), tf_practical_hz, tf_min_hz (≥1 cycle in the
    temporal_support window), sf_ref, speed_min/max_deg_s (tf_min/max ÷ sf_ref)}. `verbose` prints."""
    dpp = cr.f_deg_per_px(cam_params, lat_samp, vert_samp, sample_ds)
    hfov = float(np.degrees(cam_params['hFOV_rad']))
    tf_min, tf_max = fs / temporal_support, fs / 2
    lim = {'deg_per_px': dpp, 'sf_max': 1 / (2 * dpp), 'sf_practical': 1 / (4 * dpp),
           'sf_min': 1 / hfov, 'wl_min_deg_nyq': 2 * dpp, 'wl_min_deg_practical': 4 * dpp,
           'tf_max_hz': tf_max, 'tf_practical_hz': 0.4 * fs, 'tf_min_hz': tf_min,
           'sf_ref': sf_ref, 'speed_min_deg_s': tf_min / sf_ref, 'speed_max_deg_s': tf_max / sf_ref}
    if verbose:
        print(f'--- Nyquist limits (render {lat_samp}x{vert_samp}, ds={sample_ds}, fs={fs:.1f}, '
              f'support={temporal_support}) ---')
        print(f'  eff deg/px = {dpp:.3f}  (hFOV {hfov:.1f} deg)')
        print(f'  SPATIAL:  max sf = {lim["sf_max"]:.3f} cyc/deg (λ=2px) | practical {lim["sf_practical"]:.3f} (λ≥4px)')
        print(f'            min λ = {lim["wl_min_deg_nyq"]:.2f} deg (2px) | {lim["wl_min_deg_practical"]:.2f} deg (4px); '
              f'min sf ≈ {lim["sf_min"]:.4f} (λ=full {hfov:.0f} deg)')
        print(f'  TEMPORAL: max tf_hz = {tf_max:.1f} (fs/2) | practical {lim["tf_practical_hz"]:.1f}; '
              f'min tf_hz ≈ {tf_min:.2f} (≥1 cycle/{temporal_support}-frame window)')
        print(f'  SPEED (deg/s = tf/sf) @ sf={sf_ref} cyc/deg: {lim["speed_min_deg_s"]:.0f}–'
              f'{lim["speed_max_deg_s"]:.0f}  (tf {tf_min:.1f}–{tf_max:.1f} Hz)')
    return lim


def f_scan_conv_activity(movies, side_tags, specs, *, backend='gpu', device='cuda', filt_batch=8,
                         reduce='absmean', combine_sides='mean', transform_meng='sqrt',
                         standardize=False, sigma_scale=0.5, gamma=1.0, chunk_t=2000,
                         sample_ds=None, spatial_mean=False, n_workers=1, scan_file=None, data=None,
                         store_neural=True, verbose=True):
    """Convolve a WIDE filter grid over a rendered movie, reduce each filter to its whole-monitor
    magnitude time series → the tuning regressor matrix. gabor filters batch `filt_batch` at a time
    as conv2d out-channels (GPU); meng groups by (theta, wavelength) sharing the spatial conv across
    tfreqs. `reduce`='absmean'; `combine_sides`='mean' L+R | 'first'; `transform_meng` maps meng
    energy→amplitude ('sqrt'); `standardize` z-scores each column (applied at assembly, not stored).
    `backend`='gpu' (f_conv_gpu) | 'cpu' (scipy, `n_workers` threads). `sample_ds` (int or
    (ds_h, ds_w)) POINT-SAMPLES the rendered movie before convolving — decouples the render
    (video-generation) pixelation from the SCAN resolution: render once, scan coarser/faster without
    re-rendering (None = full render; gabor wavelengths are then in the sampled-grid pixels).
    `spatial_mean=True` also records, per filter, the TIME-MEAN magnitude map (L‖R, in the scan file) —
    where each filter's activity concentrates on the monitor (read with f_scan_spatial). Per-filter
    magnitude distribution stats (mean/std/percentiles) are available post-hoc via f_scan_stats.

    RESUMABLE SINGLE-FILE STORE (`scan_file`): the whole bundle is rewritten ATOMICALLY after every
    batch, so a crash loses at most the in-flight batch; a re-run SKIPS filters already present
    (keyed by filter identity) and only convolves NEW ones — widen `specs` freely and old values are
    reused. `data` (dict) is embedded so the file is SELF-CONTAINED: with `store_neural=True`
    (default) it stores the cell traces (`Y_neu_raw`) + ensemble outputs (`ens_out`/`ens_table`) so
    another script can reload and score WITHOUT re-loading the session; with `store_neural=False`
    only the identity + `dimred_params`(+seed) are kept (regenerate the neural/ensembles
    deterministically). Read a running/finished file with f_scan_load / f_scan_load_data /
    f_scan_progress / f_plot_scan_coverage. Returns (X_stim (T, n_filt) float32, filt_df)."""
    mov = {s: np.asarray(m) for s, m in zip(side_tags, movies)}
    _ds_meta = None
    if sample_ds is not None:                                 # downsample the movie before conv
        nd = cr._norm_ds(sample_ds)
        ds_h, ds_w = nd if isinstance(nd, tuple) else (nd, nd)
        mov = {s: m[:, ::ds_h, ::ds_w] for s, m in mov.items()}
        _ds_meta = list(nd) if isinstance(nd, tuple) else int(nd)
    T = mov[side_tags[0]].shape[0]
    meta = {'reduce': reduce, 'combine_sides': combine_sides, 'transform_meng': transform_meng,
            'sample_ds': _ds_meta}
    dev = None
    if backend == 'gpu':
        import f_conv_gpu as cg
        dev = cg.pick_device(device)

    # ---- load or init the resumable state ----
    if scan_file and os.path.exists(scan_file):
        state = joblib.load(scan_file)
        if int(state.get('T', T)) != int(T):
            raise ValueError(f"scan_file T={state.get('T')} != movie T={T} (different render) — use a new scan_file")
        if state.get('meta') != meta:
            raise ValueError(f"scan_file params {state.get('meta')} != {meta} — use a new scan_file")
        if data is not None:
            for _key in ('dset_name', 'gen_tag'):             # identity checks against the existing file
                _have, _want = state.get('data', {}).get(_key), data.get(_key)
                if _have not in (None, _want):
                    raise ValueError(f"scan_file {_key} {_have!r} != {_want!r} — use a new scan_file / gen_tag")
        if verbose:
            print(f'scan: resuming {scan_file} ({len(state["cols"])} filters present)', flush=True)
    else:
        state = {'meta': meta, 'T': int(T), 'side_tags': list(side_tags),
                 'sigma_scale': sigma_scale, 'gamma': gamma,
                 'render': {'H': int(mov[side_tags[0]].shape[1]), 'W': int(mov[side_tags[0]].shape[2])},
                 'data': {}, 'cols': {}}
    # embed / refresh the self-contained data bundle
    if data is not None:
        d = {k: data.get(k) for k in ('mouse_id', 'n_dset', 'dset_name', 'dset_tag', 'gen_tag',
                                      'frame_t', 'fs', 'active_cells', 'cell_ids', 'dimred_params',
                                      'seed', 'ens_params')}
        if store_neural:
            d.update({k: data.get(k) for k in ('Y_neu_raw', 'ens_out', 'ens_table')})
        state['data'] = {k: v for k, v in d.items() if v is not None}

    # record the FULL requested grid (done + pending) so coverage/progress know what's outstanding
    state['target'] = [{**_row_from_spec(s),
                        **{k: s[k] for k in ('sf_cyc_deg', 'wl_deg', 'tf_hz') if k in s}} for s in specs]
    done = set(state['cols'])
    gab = [s for s in specs if s['kind'] == 'gabor' and _scan_key(_row_from_spec(s)) not in done]
    meng = [s for s in specs if s['kind'] == 'meng' and _scan_key(_row_from_spec(s)) not in done]
    if scan_file:                                             # persist data bundle + target even if no new filters
        _save_scan_atomic(scan_file, state)
    t_conv = t_save = 0.0                                      # timing (verbose) → convolve vs disk-save
    n_new = 0

    # ---- gabor: batch as conv2d out-channels; save after each batch ----
    for bi, batch in enumerate(_batches(gab, filt_batch)):
        _t0 = time.perf_counter()
        _cols = _gabor_batch_cols(mov, side_tags, batch, backend, dev, reduce, combine_sides,
                                  sigma_scale, gamma, chunk_t, n_workers, spatial=spatial_mean)
        for col, row in _cols:
            state['cols'][_scan_key(row)] = {'col': col, **row}
        _tc = time.perf_counter() - _t0; t_conv += _tc; n_new += len(batch)
        _t1 = time.perf_counter()
        if scan_file:
            _save_scan_atomic(scan_file, state)
        _ts = time.perf_counter() - _t1; t_save += _ts
        if verbose:
            print(f'  gabor batch {bi + 1}: +{len(batch)} in {_tc:.2f}s ({len(batch) / max(_tc, 1e-9):.1f} filt/s)'
                  f'  [save {_ts:.2f}s]  ({len(state["cols"])} total)', flush=True)

    # ---- meng: group by (theta, wavelength, output) so the spatial conv is shared across ALL tfreqs
    # (per-tf temporal_support handled inside — constant-cycle bank doesn't break spatial sharing) ----
    groups = defaultdict(list)
    for s in meng:
        groups[(round(float(s['theta']), 6), s['wavelength'], s.get('output', 'opponent'))].append(s)
    for gi, ((th, wl, out), gspecs) in enumerate(groups.items()):
        _t0 = time.perf_counter()
        _cols = _meng_group_cols(mov, side_tags, th, wl, gspecs, out, backend, dev,
                                 reduce, combine_sides, transform_meng, sigma_scale, gamma,
                                 chunk_t, spatial=spatial_mean)
        for col, row in _cols:
            state['cols'][_scan_key(row)] = {'col': col, **row}
        _tc = time.perf_counter() - _t0; t_conv += _tc; n_new += len(gspecs)
        _t1 = time.perf_counter()
        if scan_file:
            _save_scan_atomic(scan_file, state)
        _ts = time.perf_counter() - _t1; t_save += _ts
        if verbose:
            print(f'  meng group {gi + 1}: o{cr._odeg(th)} wl{wl} +{len(gspecs)} tf in {_tc:.2f}s'
                  f'  [save {_ts:.2f}s]  ({len(state["cols"])} total)', flush=True)

    if not state['cols']:
        raise ValueError('f_scan_conv_activity: no gabor/meng specs produced any columns')
    if verbose and n_new:
        print(f'scan timing: {n_new} new filters — convolve {t_conv:.1f}s '
              f'({n_new / max(t_conv, 1e-9):.1f} filt/s), disk-save {t_save:.1f}s, '
              f'total {t_conv + t_save:.1f}s', flush=True)
    return _assemble(state, specs, standardize)


def f_scan_load(scan_file, specs=None, standardize=False):
    """Load a scan file → (X_stim, filt_df) for `specs` (or ALL stored if None). Lets another script
    plot progress / run a scorer without recomputing. `standardize` z-scores each column."""
    return _assemble(joblib.load(scan_file), specs, standardize)


def f_scan_load_data(scan_file):
    """Load the self-contained data bundle embedded in a scan file (dataset identity, `frame_t`/`fs`,
    `active_cells`/`cell_ids`, `dimred_params`/`seed`, and — if store_neural was True — `Y_neu_raw`
    / `ens_out` / `ens_table`). Empty dict if none was embedded. Pair with f_scan_load to score from
    another script with NO session reload."""
    return dict(joblib.load(scan_file).get('data', {}))


def f_scan_stats(scan_file, percentiles=(5, 25, 50, 75, 95)):
    """Per-filter DISTRIBUTION stats of the whole-monitor magnitude regressor (RAW, pre-standardize) —
    the dataset's convolved-activity magnitudes at each freq/orientation (mirrors the conv_repo
    magnitude analysis). One row per filter: identity (kind/odeg/wavelength/tfreq[/sf_cyc_deg/tf_hz])
    + mean/std/min/max + the given percentiles (p5/p25/…). Returns a DataFrame sorted like filt_df."""
    cd = joblib.load(scan_file)['cols']
    rows = []
    for r in cd.values():
        c = np.asarray(r['col'], float)
        d = {k: r.get(k) for k in ('kind', 'odeg', 'wavelength', 'tfreq', 'sf_cyc_deg', 'tf_hz') if k in r}
        d.update({'mean': float(c.mean()), 'std': float(c.std()),
                  'min': float(c.min()), 'max': float(c.max())})
        for p in percentiles:
            d[f'p{p}'] = float(np.percentile(c, p))
        rows.append(d)
    df = pd.DataFrame(rows)
    return df.sort_values([c for c in ('kind', 'odeg', 'wavelength', 'tfreq') if c in df.columns],
                          na_position='last').reset_index(drop=True)


def f_scan_spatial(scan_file, specs=None):
    """Per-filter TIME-MEAN spatial activity maps (where each filter's activity concentrates on the
    monitor, L‖R hstacked) — only present if the scan ran with spatial_mean=True. Returns
    (maps, filt_df): `maps` a list of (H, n_sides·W) arrays aligned to `filt_df` rows (specs subset or
    all). Use to read tuning in the context of where/how often that feature appeared on screen."""
    cd = joblib.load(scan_file)['cols']
    rows = (list(cd.values()) if specs is None
            else [cd[k] for k in (_scan_key(_row_from_spec(s)) for s in specs) if k in cd])
    rows = [r for r in rows if 'spatial' in r]
    if not rows:
        raise ValueError('f_scan_spatial: no spatial maps stored (run the scan with spatial_mean=True)')
    _keys = ['kind', 'odeg', 'wavelength', 'tfreq']
    _keys += [k for k in ('sf_cyc_deg', 'tf_hz') if any(k in r for r in rows)]
    filt_df = pd.DataFrame([{k: r.get(k, np.nan) for k in _keys} for r in rows])
    order = filt_df.sort_values(['odeg', 'wavelength', 'tfreq'], na_position='last').index.to_numpy()
    maps = [np.asarray(rows[i]['spatial']) for i in order]
    return maps, filt_df.iloc[order].reset_index(drop=True)


def _sf_tf_cols(df):
    """Prefer the physical axis columns (sf_cyc_deg/tf_hz) if present, else the pixel ones."""
    return (('sf_cyc_deg' if 'sf_cyc_deg' in df.columns else 'wavelength'),
            ('tf_hz' if 'tf_hz' in df.columns else 'tfreq'))


def f_plot_scan_magnitude(scan_file, dset_tag='', logy=False):
    """How strongly the gabor/meng bank responds across the grid — the dataset's convolved-activity
    magnitude vs spatial freq / orientation / temporal freq (guides whether the filters are
    well-placed). 3 panels (vs SF, vs orientation, vs TF); each = mean whole-monitor magnitude (line)
    + mean p25–p75 band, aggregated over the other axes, per kind. Reads f_scan_stats. Returns None."""
    import matplotlib.pyplot as plt
    stats = f_scan_stats(scan_file)
    sf_col, tf_col = _sf_tf_cols(stats)

    def _agg(sub, col):
        g = sub.dropna(subset=[col]).groupby(col)
        x = np.array(sorted(sub[col].dropna().unique()))
        m = g['mean'].mean().reindex(x).to_numpy()
        lo = g['p25'].mean().reindex(x).to_numpy() if 'p25' in sub.columns else m
        hi = g['p75'].mean().reindex(x).to_numpy() if 'p75' in sub.columns else m
        return x, m, lo, hi
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axs, (sf_col, 'odeg', tf_col)):
        for kind, c in (('gabor', 'C0'), ('meng', 'C1')):
            sub = stats[stats['kind'] == kind]
            if col not in sub.columns or sub[col].dropna().empty:
                continue
            x, m, lo, hi = _agg(sub, col)
            ax.plot(x, m, '-o', color=c, ms=3, label=kind)
            ax.fill_between(x, lo, hi, color=c, alpha=0.2)
        ax.set_xlabel(col); ax.set_ylabel('whole-monitor magnitude')
        ax.set_title(f'vs {col}'); ax.legend(fontsize=7)
        if logy:
            ax.set_yscale('log')
    fig.suptitle(f'{dset_tag} — filter magnitude across params')
    fig.tight_layout()
    return None


def f_plot_scan_traces(scan_file, by='sf_cyc_deg', kind=None, max_points=2000, dset_tag=''):
    """Whole-monitor magnitude vs FRAME (conv_repo-style): group-mean trace per `by` value, aggregating
    over the other axes → temporal dynamics of the convolved activity. `by` falls back to the pixel
    column if the physical one is absent. Returns None."""
    import matplotlib.pyplot as plt
    cd = list(joblib.load(scan_file)['cols'].values())
    if kind:
        cd = [r for r in cd if r['kind'] == kind]
    if by not in (cd[0] if cd else {}):
        by = 'wavelength' if by == 'sf_cyc_deg' else ('tfreq' if by == 'tf_hz' else by)
    rows = [r for r in cd if r.get(by) is not None
            and not (isinstance(r.get(by), float) and np.isnan(r.get(by)))]
    if not rows:
        print(f'f_plot_scan_traces: no rows for by={by!r} kind={kind!r}'); return None
    groups = sorted(set(r[by] for r in rows))
    T = len(rows[0]['col']); step = max(1, T // max_points); xt = np.arange(0, T, step)
    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(groups))))
    for g, c in zip(groups, cmap):
        M = np.stack([r['col'] for r in rows if r[by] == g])
        ax.plot(xt, M.mean(0)[::step], color=c, lw=1.0, label=f'{by}={g:g}')
    ax.set_xlabel('frame'); ax.set_ylabel('whole-monitor magnitude'); ax.legend(fontsize=7)
    ax.set_title(f'{dset_tag} — magnitude vs frame by {by}' + (f' ({kind})' if kind else ''))
    fig.tight_layout()
    return None


def f_plot_scan_spatial(scan_file, group_by='odeg', kind=None, dset_tag='', max_panels=12):
    """Time-mean spatial activity maps (L‖R) — WHERE activity sits on the monitor. Panel 1 = OVERALL
    mean across filters (where activity is on average); then one map per `group_by` level (mean over
    that group). Needs spatial_mean=True at scan time. Returns None."""
    import matplotlib.pyplot as plt
    maps, mdf = f_scan_spatial(scan_file)
    if kind:
        m = (mdf['kind'] == kind).to_numpy()
        maps = [maps[i] for i in np.flatnonzero(m)]
        mdf = mdf[m].reset_index(drop=True)
    if not maps:
        print('f_plot_scan_spatial: no maps'); return None
    panels = [('overall', np.mean(maps, axis=0))]
    if group_by and group_by in mdf.columns:
        for g in sorted(mdf[group_by].dropna().unique()):
            idx = np.flatnonzero((mdf[group_by] == g).to_numpy())
            panels.append((f'{group_by}={g:g}', np.mean([maps[i] for i in idx], axis=0)))
    panels = panels[:max_panels]
    n = len(panels); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    axs = axs.ravel()
    for ax, (t, M) in zip(axs, panels):
        ax.imshow(M, aspect='auto', cmap='magma')
        ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[n:]:
        ax.axis('off')
    fig.suptitle(f'{dset_tag} — time-mean activity (L‖R)' + (f' {kind}' if kind else ''))
    fig.tight_layout()
    return None


def f_scan_progress(scan_file, target_specs=None):
    """Progress summary of a scan file (safe mid-scan — the writer replaces atomically): n_done,
    per-kind counts, done filter keys, embedded-data flags, and — if `target_specs` given — how many
    of that grid remain."""
    if not (scan_file and os.path.exists(scan_file)):
        return {'exists': False, 'n_done': 0}
    state = joblib.load(scan_file)
    cd = state['cols']
    dat = state.get('data', {})
    out = {'exists': True, 'n_done': len(cd), 'meta': state.get('meta'), 'T': state.get('T'),
           'render': state.get('render'), 'dset_name': dat.get('dset_name'),
           'gen_tag': dat.get('gen_tag'),
           'has_neural': 'Y_neu_raw' in dat, 'has_ensembles': 'ens_out' in dat,
           'by_kind': {k: sum(1 for c in cd.values() if c['kind'] == k) for k in ('gabor', 'meng')},
           'keys': sorted(cd)}
    tkeys = ({_scan_key(_row_from_spec(s)) for s in target_specs} if target_specs is not None
             else {_scan_key(r) for r in state.get('target', [])})   # recorded requested grid
    if tkeys:
        out['n_target'] = len(tkeys)
        out['n_remaining'] = len(tkeys - set(cd))
    return out


def _cov_cell(row):
    """(kind, tf, odeg, sf) identity for a coverage cell — physical (sf_cyc_deg/tf_hz) if present, else
    px wavelength / cyc-frame tfreq."""
    sf = float(row['sf_cyc_deg']) if row.get('sf_cyc_deg') is not None else float(row['wavelength'])
    _t = row.get('tf_hz', row.get('tfreq', np.nan))
    tf = None if _is_nan_tf(_t) else round(float(_t), 4)
    return (row['kind'], tf, float(row['odeg']), round(sf, 4))


def f_plot_scan_coverage(scan_file, target_specs=None, dset_tag=''):
    """Neural-free progress view: DONE vs PENDING filters on the orientation × spatial-freq grid (one
    panel per kind [× temporal-freq]). Green = done, blank = pending. The full requested grid comes
    from `target_specs` if given, else the `target` recorded in the scan file (so pending shows even
    without re-specifying). Axes are physical (cyc/deg + Hz) when present."""
    import matplotlib.pyplot as plt
    state = joblib.load(scan_file)
    cd = state['cols']
    phys = any(c.get('sf_cyc_deg') is not None for c in cd.values())
    done = {_cov_cell(c) for c in cd.values()}
    if target_specs is not None:
        tgt = {_cov_cell({**_row_from_spec(s),
                          **{k: s[k] for k in ('sf_cyc_deg', 'tf_hz') if k in s}}) for s in target_specs}
    else:
        tgt = {_cov_cell(r) for r in state.get('target', [])}   # recorded requested grid
    allc = done | tgt
    if not allc:
        print('scan coverage: nothing to plot'); return None
    sf_lab = 'spatial freq (cyc/deg)' if phys else 'wavelength (px)'
    tf_lab = 'Hz' if phys else 'cyc/frame'
    panels = sorted({(k, tf) for (k, tf, o, w) in allc}, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (k, tf) in zip(axs[0], panels):
        ods = sorted({o for (kk, t, o, w) in allc if kk == k and t == tf})
        sfs = sorted({w for (kk, t, o, w) in allc if kk == k and t == tf})
        M = np.zeros((len(sfs), len(ods)))
        for iw, w in enumerate(sfs):
            for io, o in enumerate(ods):
                M[iw, io] = 1.0 if (k, tf, o, w) in done else 0.0
        ax.imshow(M, aspect='auto', origin='lower', cmap='Greens', vmin=0, vmax=1,
                  extent=[-0.5, len(ods) - 0.5, -0.5, len(sfs) - 0.5])
        ax.set_xticks(range(len(ods))); ax.set_xticklabels([f'{o:g}' for o in ods])
        ax.set_yticks(range(len(sfs))); ax.set_yticklabels([f'{w:g}' for w in sfs])
        ax.set_xlabel('orientation (deg)'); ax.set_ylabel(sf_lab)
        _ttl = f'{k}' + (f' tf{tf:g}{tf_lab}' if tf is not None else '')
        ax.set_title(f'{_ttl}  ({int(M.sum())}/{M.size} done)')
    fig.suptitle(f'{dset_tag} scan coverage (green=done)'); fig.tight_layout()
    return fig


def _dense_one(movie, filt_row, backend, device, sigma_scale, gamma):
    """Dense (T,H,W) response for ONE filt_row (kind/odeg/wavelength/tfreq) — GPU (f_conv_gpu) or CPU
    (f_conv_repo). odeg→theta=radians. gabor→magnitude, meng→opponent."""
    theta = np.radians(float(filt_row['odeg']))
    wl = float(filt_row['wavelength'])
    kind = str(filt_row['kind'])
    if backend == 'gpu':
        import f_conv_gpu as cg
        if kind == 'gabor':
            return np.asarray(cg.f_gabor_map_gpu(movie, theta, wl, sigma_scale, gamma, device=device))
        tf = float(filt_row['tfreq'])
        return np.asarray(cg.f_meng_maps_gpu(movie, theta, wl, [tf], sigma_scale=sigma_scale,
                                             gamma=gamma, device=device)[tf])
    if kind == 'gabor':
        return cr.f_gabor_map(movie, theta, wl, sigma_scale, gamma)
    return cr.f_meng_map(movie, theta, wl, float(filt_row['tfreq']),
                         sigma_scale=sigma_scale, gamma=gamma)


def f_filter_activity_maps(movies, side_tags, filt_row, activity, *, backend='gpu', device='cuda',
                           sigma_scale=0.5, gamma=1.0):
    """Verify a unit's preferred filter WITHOUT a stored conv_repo: re-convolve `filt_row` over the
    movie (on the fly) and return per side {'location': (H,W), 'sta': (H,W)}. `location` = per-pixel
    Pearson corr of the filter's response with `activity` (signed RF blob); `sta` =
    rectified-activity-weighted mean response (the feature in monitor space). Mirror of
    f_tuning.f_tuning_location_map / f_tuning_feature_sta but movie-based (magnitude-only scan)."""
    a = np.asarray(activity, float)
    out = {}
    for side, mov in zip(side_tags, movies):
        dense = _dense_one(np.asarray(mov), filt_row, backend, device, sigma_scale, gamma)
        T = min(len(a), dense.shape[0])
        aa, X = a[:T], dense[:T]
        am = aa - aa.mean(); Xc = X - X.mean(0)
        denom = (np.sqrt((Xc ** 2).sum(0)) + 1e-12) * (np.sqrt((am ** 2).sum()) + 1e-12)
        corr = (Xc * am[:, None, None]).sum(0) / denom
        w = np.clip(aa - np.median(aa), 0, None)
        sta = (X * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        out[side] = {'location': corr, 'sta': sta}
    return out


def f_save_scan_results(path, tun_scan, params=None):
    """Persist scorer RESULTS for later reload + re-plot: the per-source readouts — including the
    fitted ridge/STA RESPONSE `W` (n_filter, n_lags, n_units), `filt_df`, `lags`, per-unit `table`,
    and marginal curves — plus the `params` used to produce them (scan reduce/combine/transform/
    standardize, tuning method/lags/sources, ensemble-extraction params, seed). Written atomically
    (joblib). Reload with f_load_scan_results → feed a readout straight to f_encoding_plots.f_plot_*
    with NO recomputation."""
    tmp = str(path) + '.tmp'
    joblib.dump({'tun_scan': tun_scan, 'params': dict(params or {})}, tmp)
    os.replace(tmp, path)


def f_load_scan_results(path):
    """Load saved scorer results → (tun_scan, params). `tun_scan[src]['readouts'][method]` is the full
    readout (W response, filt_df, tables, marginals) ready for f_encoding_plots.f_plot_tuning_*."""
    d = joblib.load(path)
    return d['tun_scan'], d.get('params', {})


def f_score_scan(X_stim, filt_df, neural, *, scorer='tuning', sources=('cells', 'ensembles'),
                 ens_out=None, ens_table=None, dimred_params=None, ens_which='sig', fs=None,
                 methods=('sta',), lags_ms=(0, 100, 200, 300, 400), split_kind=True, verbose=True):
    """Score the scanned regressors against neural units — PLUGGABLE scorer.
      scorer='tuning' (v1): for each source in `sources` ('cells'|'ensembles'|'ens_cells'|
        'reliable_cells') select the (T, n_units) activity (f_tuning.f_select_tuning_units),
        reverse-correlate against X_stim at `lags_ms` (each `methods` entry 'sta'|'ridge'), and read
        out preferred tuning (f_tuning.f_tuning_readout). `split_kind=True` (default) scores each
        FEATURE FAMILY separately (gabor = orientation×SF, meng = orientation×SF×TF) — a mixed grid
        would let meng's temporal axis hide the gabor filters in the readout/heatmap. Returns
        {'<source>:<feat_kind>': {'readouts': {method: readout}, 'labels': [...], 'kind': src_kind,
        'source': src, 'feat_kind': kind}} (key is just the source when split_kind=False).
      scorer='encode'|'decode': documented extension points (raise NotImplementedError) — will reuse
        f_dataset_scan.f_score_representations / f_encoding.f_run_encoder / f_decoding.f_run_decode_sweep
        with X_stim (or filt_df-defined subsets) as the feature block.
    `neural` = Y_neu_raw (T, n_active) in the extraction cell order; ensemble sources need ens_out
    (+ ens_table for 'sig'). `fs` = imaging Hz (for lag→ms + peak-lag ms)."""
    if scorer == 'tuning':
        import f_tuning as tun
        import f_encoding as enc
        _lags = enc.f_lags_from_ms(list(lags_ms), fs)
        fdf = filt_df.reset_index(drop=True)
        feat_kinds = list(dict.fromkeys(fdf['kind'].tolist())) if split_kind else [None]
        out = {}
        for src in sources:
            try:
                Yu, labels, kind = tun.f_select_tuning_units(
                    src, neural, ens_out=ens_out, ens_table=ens_table, dimred_params=dimred_params,
                    ens_src_which=ens_which, fs=fs)
            except Exception as e:
                if verbose:
                    print(f'  score: source {src!r} skipped ({type(e).__name__}: {e})')
                continue
            n = min(X_stim.shape[0], Yu.shape[0])
            Xs_all, Yu = np.asarray(X_stim)[:n], Yu[:n]
            for fk in feat_kinds:
                cmask = (np.ones(fdf.shape[0], bool) if fk is None
                         else (fdf['kind'] == fk).to_numpy())
                if not cmask.any():
                    continue
                key = src if fk is None else f'{src}:{fk}'
                Xs = Xs_all[:, cmask]; fdf_k = fdf.iloc[cmask].reset_index(drop=True)
                res = {}
                for m in methods:
                    fit = tun.f_reverse_corr_tuning(Xs, fdf_k, Yu, _lags, method=m)
                    res[m] = tun.f_tuning_readout(fit, fs=fs, labels=labels)
                out[key] = {'readouts': res, 'labels': labels, 'kind': kind,
                            'source': src, 'feat_kind': fk}
                if verbose:
                    print(f'  score[{key}]: {Yu.shape[1]} units × {int(cmask.sum())} filters '
                          f'({"/".join(methods)})')
        return out
    if scorer in ('encode', 'decode'):
        raise NotImplementedError(
            f"scorer={scorer!r} is a planned extension point (not built in v1). Intended wiring: "
            f"treat X_stim (or per-filter / per-group subsets selected via filt_df) as a feature "
            f"block and call f_dataset_scan.f_score_representations / f_encoding.f_run_encoder "
            f"('encode') or f_decoding.f_run_decode_sweep ('decode') to score each filter/group's "
            f"encode/decode performance per unit. The (X_stim, filt_df, neural, sources) signature is "
            f"chosen so these drop in without changing f_scan_conv_activity.")
    raise ValueError(f"unknown scorer {scorer!r} (use 'tuning' | 'encode' | 'decode')")
