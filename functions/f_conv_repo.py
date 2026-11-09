# -*- coding: utf-8 -*-
"""
f_conv_repo.py — dense per-filter CONVOLVED-feature repository (fast decoder optimization).

Feature-gen normally convolves the movie then immediately reduces (grid-pool / point-sample)
+ PCAs, saving only the reduced block — so every resolution/parameter sweep RE-CONVOLVES the
movie (the slow part). This module persists the DENSE per-filter convolved maps ONCE per dataset,
so downstream sampling / pooling / PCA / decoding just LOAD + subsample (no re-conv).

Repo layout  <features_dir>/conv_repo/<dset_name>_<GEN_TAG>/ :
    manifest.json          — dataset meta + per-filter entries (params + uint16 quant vmin/vmax)
    frame_t.npy            — imaging clock (align to neural later)
    <stat>_<side>_o{odeg}_sf{cyc/deg}[_tf{Hz}].npy — ONE uint16 (T,H,W) map per side × filter for
                             PHYSICAL specs (e.g. gabM_L_o0_sf0.13.npy / mengopp_L_o0_sf0.13_tf2.npy);
                             legacy pixel specs use _wl{px}[_tf{tftag}] instead. Filename is an on-disk
                             key + label only — selection/loading use the manifest fields.
                             (odeg = orientation in DEGREES, e.g. o0/o45/o90/o135; tftag = the
                              temporal-freq VALUE, lossless 3-decimal, e.g. tf0025 = 0.025 cyc/fr)

Per-filter files → filters can be ADDED INCREMENTALLY (new orientations / wavelengths / TFs)
without recomputing existing ones. Maps are uint16-quantized (per-filter min/max) — precision is
not critical and it halves size vs float32 (~1 GB/filter/side at num_samp=101 for a long session).
(A separate converter can later repackage these into HDF5 / TIFF for browsing if needed.)

Reuses the extractor internals from f_visual_features (_gabor_pair / _spatial_fftconv_chunked /
_temporal_quad) so the dense maps are byte-compatible with f_gabor_features / f_motion_energy_features.

Created 2026-07-11 (TODO #68, extends the sampled-feature method).
"""

import json
from pathlib import Path

import numpy as np

from f_visual_features import _gabor_pair, _spatial_fftconv_chunked, _temporal_quad, _sample_idx


# ─────────────────────────────────────────────────────────────────────────────
# uint16 quantization (per-filter linear min/max; handles signed meng opponent)
# ─────────────────────────────────────────────────────────────────────────────

def _quant_u16(x):
    """Linear-quantize a float array to uint16 over its own [min,max]. Returns (q, vmin, vmax)."""
    x = np.asarray(x, np.float32)
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(x.shape, np.uint16), vmin, vmin + 1.0
    q = np.round((x - vmin) / (vmax - vmin) * 65535.0)
    return np.clip(q, 0, 65535).astype(np.uint16), vmin, vmax


def _dequant_u16(q, vmin, vmax):
    """Inverse of _quant_u16 → float32."""
    return (vmin + np.asarray(q, np.float32) / 65535.0 * (vmax - vmin)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dense per-filter maps (one orientation × wavelength [× temporal-freq])
# ─────────────────────────────────────────────────────────────────────────────

def f_gabor_map(movie, theta, wavelength, sigma_scale=0.5, gamma=1.0, chunk_t=2000):
    """Dense gabor MAGNITUDE map |R|=√(even²+odd²) (T,H,W) for one (orientation × wavelength)."""
    movie = np.asarray(movie, np.float32)
    ev, od = _gabor_pair(theta, wavelength, sigma_scale * wavelength, gamma)
    re = _spatial_fftconv_chunked(movie, ev, chunk_t)
    ro = _spatial_fftconv_chunked(movie, od, chunk_t)
    return np.sqrt(re * re + ro * ro).astype(np.float32)


def f_meng_map(movie, theta, wavelength, tfreq, temporal_support=11, temporal_sigma=None,
               sigma_scale=0.5, gamma=1.0, output='opponent', chunk_t=1000):
    """Dense Adelson–Bergen motion energy (T,H,W) for one (orientation × wavelength × TF).
    output='opponent' → |R|²−|L|² (1 map); 'both' → (eR, eL). Halo-chunked in time; byte-
    compatible with f_motion_energy_features's inner loop."""
    from scipy.signal import fftconvolve
    movie = np.asarray(movie, np.float32)
    T, H, W = movie.shape
    halo = temporal_support // 2
    tsig = temporal_sigma if temporal_sigma else temporal_support / 4.0
    ev, od = _gabor_pair(theta, wavelength, sigma_scale * wavelength, gamma)
    te, to = _temporal_quad(tfreq, temporal_support, tsig)
    opp = np.empty((T, H, W), np.float32) if output == 'opponent' else None
    eRo = None if output == 'opponent' else np.empty((T, H, W), np.float32)
    eLo = None if output == 'opponent' else np.empty((T, H, W), np.float32)
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        lo, hi = max(0, s - halo), min(T, e + halo)
        sub = movie[lo:hi]
        vs, ve = s - lo, s - lo + (e - s)
        re = fftconvolve(sub, ev[None], mode='same', axes=(1, 2))
        ro = fftconvolve(sub, od[None], mode='same', axes=(1, 2))
        Cs = (re + 1j * ro).astype(np.complex64)
        R = fftconvolve(Cs, (te + 1j * to).reshape(-1, 1, 1), mode='same', axes=0)
        L = fftconvolve(Cs, (te - 1j * to).reshape(-1, 1, 1), mode='same', axes=0)
        eR = (R.real ** 2 + R.imag ** 2).astype(np.float32)[vs:ve]
        eL = (L.real ** 2 + L.imag ** 2).astype(np.float32)[vs:ve]
        if output == 'opponent':
            opp[s:e] = eR - eL
        else:
            eRo[s:e] = eR
            eLo[s:e] = eL
    return opp if output == 'opponent' else (eRo, eLo)


# ─────────────────────────────────────────────────────────────────────────────
# Repo I/O
# ─────────────────────────────────────────────────────────────────────────────

def f_conv_repo_dir(features_dir, dset_name, gen_tag):
    """Repo directory for one dataset: <features_dir>/conv_repo/<dset_name>_<gen_tag>."""
    return Path(features_dir) / 'conv_repo' / f'{dset_name}_{gen_tag}'


def f_conv_filter_specs(conv_specs):
    """Expand a conv_specs dict → a flat list of per-filter specs for f_build_conv_repo.
    conv_specs = {'gabor': {'n_orient', 'wavelengths'},
                  'meng':  {'n_orient', 'wavelengths', 'temporal_freqs', 'temporal_support',
                            'output'}}. Orientations are evenly spaced in [0,π); isf/itf index the
    wavelength / temporal-freq lists (→ the standard sfN/tfN channel tokens)."""
    def _num(x):                                       # Python int if whole, else float (JSON-safe)
        xf = float(x)
        return int(xf) if xf.is_integer() else xf
    specs = []
    g = conv_specs.get('gabor')
    if g:
        n_o = g['n_orient']
        wls = [_num(w) for w in np.atleast_1d(g['wavelengths'])]    # tolerate a scalar
        for io in range(n_o):
            th = io * np.pi / n_o
            for isf, wl in enumerate(wls):
                specs.append(dict(kind='gabor', io=io, theta=float(th), isf=isf, wavelength=wl))
    m = conv_specs.get('meng')
    if m:
        n_o = m['n_orient']
        wls = [_num(w) for w in np.atleast_1d(m['wavelengths'])]
        tfs = [_num(t) for t in np.atleast_1d(m['temporal_freqs'])]
        for io in range(n_o):
            th = io * np.pi / n_o
            for isf, wl in enumerate(wls):
                for itf, tf in enumerate(tfs):
                    specs.append(dict(kind='meng', io=io, theta=float(th), isf=isf, wavelength=wl,
                                      itf=itf, tfreq=tf,
                                      temporal_support=m.get('temporal_support', 11),
                                      output=m.get('output', 'opponent')))
    return specs


def f_deg_per_px(cam_params, lat_samp, vert_samp, sample_ds=None):
    """Effective degrees-of-visual-angle per pixel of the conv grid = mean of hFOV/lat_samp and
    vFOV/vert_samp (both in degrees), scaled by the point-sampling stride (fewer pixels → wider
    angular spacing). Scalar. For the angularly-isotropic render the two axes match, so the mean is
    exact; for anisotropic renders it's an average. Used to convert cyc/deg ↔ pixel wavelength."""
    hdeg, vdeg = np.degrees(cam_params['hFOV_rad']), np.degrees(cam_params['vFOV_rad'])
    dph, dpv = hdeg / lat_samp, vdeg / vert_samp
    if sample_ds is not None:
        nd = _norm_ds(sample_ds)
        dw, dh = nd if isinstance(nd, tuple) else (nd, nd)
        dph *= dw; dpv *= dh
    return float((dph + dpv) / 2.0)


def f_conv_filter_specs_phys(conv_specs_phys, cam_params, lat_samp, vert_samp, fs=None,
                             sample_ds=None):
    """Build conv filter specs from PHYSICAL, resolution/framerate-INDEPENDENT units: gabor spatial
    freq in CYCLES/DEGREE ('sf_cyc_deg', replaces 'wavelengths') and meng temporal freq in HZ
    ('tf_hz', replaces 'temporal_freqs'). Converts to the render's pixel/frame units:
        deg_per_px = f_deg_per_px(cam_params, lat_samp, vert_samp, sample_ds)
        wavelength_px = 1 / (sf_cyc_deg · deg_per_px)   ;   tfreq_cyc_per_frame = tf_hz / fs.
    Same shape as f_conv_filter_specs (kind/io/theta/isf/wavelength[/itf/tfreq/temporal_support/
    output]) PLUS the physical fields sf_cyc_deg/wl_deg (gabor) and tf_hz (meng) on every spec so
    downstream (scan filt_df, tuning readout) can label preferences in cyc/deg + Hz. `lat_samp`/
    `vert_samp` = RENDER resolution; pass the same `sample_ds` the scan uses so the pixel wavelength
    lands in the sampled conv grid.
    MAX: sf_cyc_deg < 1/(2·deg_per_px) (render Nyquist); tf_hz < fs/2 (temporal Nyquist)."""
    dpp = f_deg_per_px(cam_params, lat_samp, vert_samp, sample_ds)
    specs = []
    g = conv_specs_phys.get('gabor')
    if g:
        n_o = g['n_orient']
        sfs = [float(s) for s in np.atleast_1d(g['sf_cyc_deg'])]
        for io in range(n_o):
            th = io * np.pi / n_o
            for isf, sf in enumerate(sfs):
                wl_deg = 1.0 / sf
                specs.append(dict(kind='gabor', io=io, theta=float(th), isf=isf,
                                  wavelength=float(wl_deg / dpp),
                                  sf_cyc_deg=float(sf), wl_deg=float(wl_deg)))
    m = conv_specs_phys.get('meng')
    if m:
        if not fs:
            raise ValueError('f_conv_filter_specs_phys: meng tf_hz needs fs (imaging Hz)')
        n_o = m['n_orient']
        sfs = [float(s) for s in np.atleast_1d(m['sf_cyc_deg'])]
        hzs = [float(t) for t in np.atleast_1d(m['tf_hz'])]
        for io in range(n_o):
            th = io * np.pi / n_o
            _tsup = m.get('temporal_support', 11)      # int | 'auto' (constant-cycle, per-tf)
            _ncyc = m.get('n_cycles', 2.0); _smin = m.get('support_min', 7); _smax = m.get('support_max', 91)
            for isf, sf in enumerate(sfs):
                wl_deg = 1.0 / sf
                for itf, hz in enumerate(hzs):
                    sup = f_auto_support(hz, fs, _ncyc, _smin, _smax) if _tsup == 'auto' else int(_tsup)
                    specs.append(dict(kind='meng', io=io, theta=float(th), isf=isf,
                                      wavelength=float(wl_deg / dpp), tfreq=float(hz / fs),
                                      sf_cyc_deg=float(sf), wl_deg=float(wl_deg), tf_hz=float(hz),
                                      itf=itf, temporal_support=sup,
                                      output=m.get('output', 'opponent')))
    return specs


def f_auto_support(tf_hz, fs, n_cycles=2.0, support_min=7, support_max=91):
    """Constant-cycle temporal support (frames) for a meng filter at `tf_hz`: ~`n_cycles` temporal
    cycles → round(n_cycles·fs/tf_hz), clamped to [support_min, support_max], made ODD (centered
    kernel). Mirrors the spatial gabor's σ=sigma_scale·λ so every tf spans the same #cycles (uniform
    temporal bandwidth). Low tf → long window (capped by support_max)."""
    s = int(round(n_cycles * fs / float(tf_hz)))
    s = max(int(support_min), min(int(support_max), s))
    return s if s % 2 else s + 1


def _odeg(theta):
    """Orientation in whole degrees (the meaningful label, vs the arbitrary index)."""
    return int(round(np.degrees(theta))) % 180


def _tf_tag(tfreq):
    """Temporal-freq filename tag = the VALUE at fixed 3-decimal precision, period dropped:
    0.025 → '0025', 0.05 → '0050', 0.1 → '0100', 0.15 → '0150', 0.25 → '0250'. Fixed-width,
    lexically sortable, and LOSSLESS at this rig's 0.001 cyc/frame resolution. (The old 2-decimal
    tag rounded 0.025 → '003' = 0.03, misrepresenting the value and colliding nearby freqs.) Not
    the arbitrary index. Migrate old repos with f_conv_repo_retag."""
    return f'{float(tfreq):.3f}'.replace('.', '')


def _filter_fname(stat, side, odeg, wl, tfreq=None, sf_cyc_deg=None, tf_hz=None):
    """Per-filter .npy filename (also the display label). `odeg` = orientation in DEGREES.
    PHYSICAL units when available (f_conv_filter_specs_phys) → clean short tokens `sf{cyc/deg}`
    [+ `tf{Hz}`], e.g. 'gabM_L_o0_sf0.13.npy' / 'mengopp_L_o0_sf0.13_tf2.npy'. LEGACY (pixel) specs
    fall back to `wl{px}` [+ `tf{cyc-frame-tag}`] so old repos keep their names. The filename is only
    an on-disk key + label — selection/loading use the manifest fields, not the name."""
    if sf_cyc_deg is not None:                            # physical (cyc/deg + Hz), human-readable
        base = f'{stat}_{side}_o{odeg}_sf{float(sf_cyc_deg):g}'
        if tf_hz is not None:
            base += f'_tf{float(tf_hz):g}'
        return base + '.npy'
    base = f'{stat}_{side}_o{odeg}_wl{wl}'                # legacy pixel wavelength / cyc-frame tag
    if tfreq is not None:
        base += f'_tf{_tf_tag(tfreq)}'
    return base + '.npy'


def _chan_prefix(entry):
    """Channel prefix (…_r{iy}c{ix} appended downstream) for a filter entry — orientation in
    DEGREES: gabor → 'gabM_o{odeg}sf{isf}'; meng → '<stat>_o{odeg}sf{isf}tf{itf}'."""
    if entry['kind'] == 'gabor':
        return f"{entry['stat']}_o{entry['odeg']}sf{entry['isf']}"
    return f"{entry['stat']}_o{entry['odeg']}sf{entry['isf']}tf{entry['itf']}"


def _reindex_conv_filters(filters):
    """Reassign isf/itf for every filter from the CANONICAL sorted order of the distinct
    wavelength / tfreq VALUES present per kind, so the indices are unique + stable regardless of
    (incremental) build order. Incremental builds enumerate isf/itf per-call (f_conv_filter_specs),
    so a value added in a later pass reuses an index already taken → two tfreqs collapse to the same
    sf/tf channel token (…tf0). Keying off the VALUE (as filenames already do via _tf_tag) removes
    the collision. Returns a NEW list of copied entries; filenames are untouched (they carry the
    value tag)."""
    out = [dict(e) for e in filters]
    for kind in {e['kind'] for e in out}:
        ek = [e for e in out if e['kind'] == kind]
        wl_ix = {w: i for i, w in enumerate(sorted({e['wavelength'] for e in ek
                                                    if e.get('wavelength') is not None}))}
        tf_ix = {t: i for i, t in enumerate(sorted({e['tfreq'] for e in ek
                                                    if e.get('tfreq') is not None}))}
        for e in ek:
            if e.get('wavelength') is not None:
                e['isf'] = wl_ix[e['wavelength']]
            if e.get('tfreq') is not None:
                e['itf'] = tf_ix[e['tfreq']]
    return out


def _spec_maps(movie, spec, sigma_scale, gamma, gabor_chunk_t, meng_chunk_t):
    """Dense map(s) for one filter spec → list of (stat_token, (T,H,W) float32)."""
    if spec['kind'] == 'gabor':
        return [('gabM', f_gabor_map(movie, spec['theta'], spec['wavelength'],
                                     sigma_scale, gamma, gabor_chunk_t))]
    out = spec.get('output', 'opponent')
    res = f_meng_map(movie, spec['theta'], spec['wavelength'], spec['tfreq'],
                     spec.get('temporal_support', 11), spec.get('temporal_sigma'),
                     sigma_scale, gamma, out, meng_chunk_t)
    if out == 'opponent':
        return [('mengopp', res)]
    return [('mengd0', res[0]), ('mengd1', res[1])]


def f_build_conv_repo(movies, side_tags, filter_specs, repo_dir, frame_t, meta=None,
                      sigma_scale=0.5, gamma=1.0, gabor_chunk_t=2000, meng_chunk_t=1000,
                      overwrite=False, backend='cpu', device='cuda', verbose=True):
    """Build (or incrementally extend) the convolved-feature repo — one uint16 (T,H,W) .npy per
    side × filter + a manifest.json (params + quant vmin/vmax) + frame_t.npy. `movies` = per-side
    dense movie (aligned to side_tags). `filter_specs` = list of dicts {kind:'gabor'|'meng', io,
    theta, isf, wavelength[, itf, tfreq, output, temporal_support]}. Skips filter files already in
    the manifest (unless overwrite) → add new orientations/wavelengths/TFs without recomputing.
    `backend`='cpu' (scipy, default) | 'gpu' (torch via f_conv_gpu, `device`) — GPU produces the
    SAME dense maps (parity-tested) far faster. Returns (manifest, n_written, n_skipped)."""
    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    man_path = repo_dir / 'manifest.json'
    if man_path.exists():
        manifest = json.loads(man_path.read_text())
    else:
        meta = meta or {}
        _mh = np.asarray(movies[0]).shape[1:3]              # actual rendered (H, W) — may be non-square
        manifest = {'mouse': meta.get('mouse'), 'dset_name': meta.get('dset_name'),
                    'gen_tag': meta.get('gen_tag'), 'num_samp': meta.get('num_samp'),
                    'mon_hw': [int(_mh[0]), int(_mh[1])],
                    'side_tags': list(side_tags), 'filters': []}
    have = {f['file'] for f in manifest['filters']}
    if not (repo_dir / 'frame_t.npy').exists():
        np.save(repo_dir / 'frame_t.npy', np.asarray(frame_t, np.float64))
    mov = dict(zip(side_tags, movies))

    n_written = n_skipped = 0
    total = len(filter_specs) * len(side_tags)
    k = 0
    for spec in filter_specs:
        for side in side_tags:
            k += 1
            toks = (['gabM'] if spec['kind'] == 'gabor'
                    else (['mengopp'] if spec.get('output', 'opponent') == 'opponent'
                          else ['mengd0', 'mengd1']))
            _deg = _odeg(spec['theta'])
            fns = [_filter_fname(t, side, _deg, spec['wavelength'], spec.get('tfreq'),
                                 sf_cyc_deg=spec.get('sf_cyc_deg'), tf_hz=spec.get('tf_hz'))
                   for t in toks]
            _sf = (f"sf{spec['sf_cyc_deg']:g}cyc/deg" if spec.get('sf_cyc_deg') is not None
                   else f"wl{spec['wavelength']}px")
            _tf = ((f" tf{spec['tf_hz']:g}Hz" if spec.get('tf_hz') is not None
                    else f" tf{spec.get('tfreq')}") if spec['kind'] == 'meng' else '')
            _tag = f"{spec['kind']} {_deg}° {_sf}{_tf} [{side}]"
            if all(fn in have for fn in fns) and not overwrite:
                n_skipped += len(fns)
                if verbose:
                    print(f'    [{k}/{total}] skip {_tag} (present)')
                continue
            if verbose:
                print(f'    [{k}/{total}] convolving {_tag} ({backend}) ...', flush=True)
            if backend == 'gpu':
                from f_conv_gpu import _spec_maps_gpu
                _ck = gabor_chunk_t if spec['kind'] == 'gabor' else meng_chunk_t
                maps = _spec_maps_gpu(mov[side], spec, sigma_scale, gamma, chunk_t=_ck, device=device)
            else:
                maps = _spec_maps(mov[side], spec, sigma_scale, gamma, gabor_chunk_t, meng_chunk_t)
            for (tok, m), fn in zip(maps, fns):
                if fn in have and not overwrite:
                    n_skipped += 1
                    continue
                q, vmin, vmax = _quant_u16(m)
                np.save(repo_dir / fn, q)
                entry = {'file': fn, 'kind': spec['kind'], 'stat': tok, 'side': side,
                         'io': int(spec['io']), 'odeg': int(_deg), 'theta': float(spec['theta']),
                         'isf': int(spec['isf']), 'wavelength': spec['wavelength'],
                         'itf': (None if spec.get('itf') is None else int(spec['itf'])),
                         'tfreq': spec.get('tfreq'), 'vmin': vmin, 'vmax': vmax,
                         'shape': list(m.shape)}
                for _pk in ('sf_cyc_deg', 'wl_deg', 'tf_hz'):     # physical units (f_conv_filter_specs_phys)
                    if spec.get(_pk) is not None:
                        entry[_pk] = float(spec[_pk])
                manifest['filters'] = [f for f in manifest['filters'] if f['file'] != fn] + [entry]
                have.add(fn)
                n_written += 1
                if verbose:
                    print(f'      saved {fn}  [{vmin:.3g},{vmax:.3g}]')
    manifest['filters'] = _reindex_conv_filters(manifest['filters'])   # unique isf/itf by value
    man_path.write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f'  conv_repo {repo_dir.name}: +{n_written} maps, {n_skipped} skipped')
    return manifest, n_written, n_skipped


def f_list_conv_filters(repo_dir):
    """Manifest filter entries for a repo, with isf/itf canonicalized by VALUE (_reindex_conv_filters)
    so incremental-build index collisions can't collapse two wavelengths/tfreqs into one channel token.
    The single read chokepoint for all loaders/activity, so every path sees consistent indices."""
    return _reindex_conv_filters(
        json.loads((Path(repo_dir) / 'manifest.json').read_text())['filters'])


def f_conv_frame_t(repo_dir):
    """Imaging clock (frame_t) stored in the repo (frame_t.npy)."""
    return np.load(Path(repo_dir) / 'frame_t.npy')


def f_load_conv_map(repo_dir, entry, mmap=True):
    """Dequantized dense (T,H,W) float32 map for one filter entry (mmap the uint16 .npy)."""
    q = np.load(Path(repo_dir) / entry['file'], mmap_mode='r' if mmap else None)
    return _dequant_u16(np.asarray(q), entry['vmin'], entry['vmax'])


def _ram_str():
    """' [RAM x.x GB]' for the current process (via psutil), or '' if psutil isn't available."""
    try:
        import psutil, os
        return f' [RAM {psutil.Process(os.getpid()).memory_info().rss / 1e9:.1f} GB]'
    except Exception:
        return ''


def _norm_ds(ds):
    """Normalize a stride spec → scalar int (isotropic) or (ds_h, ds_w) int tuple (anisotropic)."""
    if isinstance(ds, (tuple, list)):
        return (int(ds[0]), int(ds[1]))
    return int(ds)


def _ds_tag(ds):
    """Block-name-safe stride tag: 8 → '8', (8, 16) → '8x16', 'nyquist'/callable → 'nyq'."""
    if callable(ds):
        return 'nyq'
    if isinstance(ds, str):
        return 'nyq' if ds == 'nyquist' else ds
    ds = _norm_ds(ds)
    return f'{ds[0]}x{ds[1]}' if isinstance(ds, tuple) else str(ds)


def _dhw(ds):
    """Stride spec → (ds_h, ds_w) int tuple (scalar → both axes)."""
    nd = _norm_ds(ds)
    return nd if isinstance(nd, tuple) else (nd, nd)


def f_ds_from_wl(wl_px, factor=0.5, ds_min=1, ds_max=None):
    """Nyquist-matched point-sample stride for a filter of pixel wavelength `wl_px`. The quadrature
    MAGNITUDE map is phase-invariant → smooth on the envelope scale σ=sigma_scale·λ (default 0.5λ),
    so ds≈`factor`·λ samples it without aliasing (factor≈sigma_scale). Returns
    max(ds_min, round(factor·wl_px)) clamped to ds_max — high-SF (small λ) → fine ds, low-SF → coarse."""
    ds = max(int(ds_min), int(round(float(factor) * float(wl_px))))
    return min(ds, int(ds_max)) if ds_max else ds


def f_sample_conv_to_block(repo_dir, ds, select=None, side_tags=None, verbose=True):
    """Point-sample the repo's dense maps at stride `ds` → a raw feature block (X, names) ready for
    f_apply_block_pca / f_score_representations / viz. `ds` = a scalar stride (both axes), a
    (ds_h, ds_w) tuple (ANISOTROPIC), OR a CALLABLE `ds(entry) → scalar/tuple` for PER-FILTER strides
    (e.g. Nyquist-matched to each filter's wavelength via f_ds_from_wl — high-SF sampled finer). With
    a callable, different filters get different r{iy}c{ix} grids (variable columns per filter — fine).
    `select` = optional predicate on a filter entry. Memory-safe: mmap each uint16 map, subsample
    BEFORE dequant. Columns are side-block-contiguous + named with the channel prefix and r{iy}c{ix}
    lattice. `verbose` prints per filter the file + running column count + process RAM (psutil)."""
    repo_dir = Path(repo_dir)
    entries = f_list_conv_filters(repo_dir)
    if select is not None:
        entries = [e for e in entries if select(e)]
    if side_tags is None:
        side_tags = json.loads((repo_dir / 'manifest.json').read_text())['side_tags']
    _ds_fn = ds if callable(ds) else None
    ds_h, ds_w = _dhw(ds) if _ds_fn is None else (None, None)
    total = len(entries)
    k = 0
    per_side = []
    for side in side_tags:
        cols, names = [], []
        for e in [e for e in entries if e['side'] == side]:
            k += 1
            dh, dw = _dhw(_ds_fn(e)) if _ds_fn is not None else (ds_h, ds_w)   # per-filter stride if callable
            q = np.load(repo_dir / e['file'], mmap_mode='r')          # uint16 (T,H,W) on disk
            oh = int(_sample_idx(q.shape[1], dh)[0])                  # CENTERED sample offset (matches build)
            ow = int(_sample_idx(q.shape[2], dw)[0])
            sub = np.array(q[:, oh::dh, ow::dw])                      # materialize sampled uint16 (H,W strides)
            x = _dequant_u16(sub, e['vmin'], e['vmax'])               # → float32 (T,ny,nx)
            pfx = _chan_prefix(e)
            ny, nx = x.shape[1], x.shape[2]
            for iy in range(ny):
                for ix in range(nx):
                    cols.append(x[:, iy, ix])
                    names.append(f'{side}_{pfx}_r{iy}c{ix}')
            if verbose:
                print(f'  loading {k}/{total}  {e["file"]}  (+{ny*nx} cols, {len(cols)} so far)'
                      + _ram_str(), flush=True)
        if not cols:
            raise ValueError(f'no selected filters for side {side!r} in {repo_dir.name}')
        per_side.append((np.column_stack(cols).astype(np.float32), names))
    X = np.concatenate([p[0] for p in per_side], axis=1)
    names = [n for p in per_side for n in p[1]]
    if verbose:
        print(f'  sampled block: X {X.shape} ({X.nbytes / 1e9:.2f} GB)' + _ram_str(), flush=True)
    return X, names


def f_gridavg_conv_to_block(repo_dir, grid, select=None, side_tags=None, chunk_t=2000,
                            verbose=True):
    """Grid-AVERAGE the repo's dense maps into a (ny,nx) tiling → a raw feature block (X, names) —
    the mean-pool analogue of f_sample_conv_to_block (which point-samples). Cell tiling matches
    f_visual_features.f_grid_bounds + area-normalized reduceat, so values equal a fresh
    f_gabor_features(pool='grid', grid=…) / f_motion_energy_features extraction within the uint16
    quant tolerance. `grid`=(ny,nx). `select` = optional predicate on a filter entry. Memory-safe:
    mmap each uint16 map, dequant + pool per time-chunk (never materializes a full dense map).
    Columns are side-block-contiguous + named with the standard channel prefix and r{iy}c{ix}
    lattice, so downstream treats it like any sampled/grid visual block. Returns (X, names)."""
    from f_visual_features import f_grid_bounds
    repo_dir = Path(repo_dir)
    entries = f_list_conv_filters(repo_dir)
    if select is not None:
        entries = [e for e in entries if select(e)]
    if side_tags is None:
        side_tags = json.loads((repo_dir / 'manifest.json').read_text())['side_tags']
    ny, nx = grid
    total = len(entries)
    k = 0
    per_side = []
    for side in side_tags:
        cols, names = [], []
        for e in [e for e in entries if e['side'] == side]:
            k += 1
            if verbose:
                print(f'  grid-avg {k}/{total}  {e["file"]}', flush=True)
            q = np.load(repo_dir / e['file'], mmap_mode='r')          # uint16 (T,H,W) on disk
            T, H, W = q.shape
            ys, xs = f_grid_bounds(H, W, grid)
            area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)
            pooled = np.empty((T, ny, nx), np.float32)
            for s in range(0, T, chunk_t):                           # dequant + pool per time-chunk
                ee = min(s + chunk_t, T)
                x = _dequant_u16(np.asarray(q[s:ee]), e['vmin'], e['vmax'])   # (chunk,H,W) float32
                pooled[s:ee] = (np.add.reduceat(np.add.reduceat(x, ys[:-1], axis=1),
                                                xs[:-1], axis=2) / area[None])
            pfx = _chan_prefix(e)
            for iy in range(ny):
                for ix in range(nx):
                    cols.append(pooled[:, iy, ix])
                    names.append(f'{side}_{pfx}_r{iy}c{ix}')
        if not cols:
            raise ValueError(f'no selected filters for side {side!r} in {repo_dir.name}')
        per_side.append((np.column_stack(cols).astype(np.float32), names))
    X = np.concatenate([p[0] for p in per_side], axis=1)
    names = [n for p in per_side for n in p[1]]
    return X, names


def f_conv_repo_prune(repo_dir, delete_orphans=False, verbose=True):
    """Reconcile a repo's manifest.json with the .npy files actually on disk. DROPS manifest entries
    whose .npy is missing (so after you delete stale files by hand, loading won't try to read them),
    and if `delete_orphans` also removes .npy files NOT listed in the manifest. Use it after deleting
    old-named files (e.g. the old tf-INDEX meng maps) so a rebuild adds only the new ones cleanly.
    Returns (n_dropped_entries, n_orphans_removed)."""
    repo_dir = Path(repo_dir)
    man_path = repo_dir / 'manifest.json'
    manifest = json.loads(man_path.read_text())
    kept = [e for e in manifest['filters'] if (repo_dir / e['file']).exists()]
    dropped = [e for e in manifest['filters'] if not (repo_dir / e['file']).exists()]
    manifest['filters'] = kept
    man_path.write_text(json.dumps(manifest, indent=1))
    n_orphans = 0
    if delete_orphans:
        listed = {e['file'] for e in kept}
        for p in repo_dir.glob('*.npy'):
            if p.name != 'frame_t.npy' and p.name not in listed:
                p.unlink()
                n_orphans += 1
    if verbose:
        print(f'  pruned manifest: dropped {len(dropped)} missing-file entries, '
              f'removed {n_orphans} orphan .npy')
    return len(dropped), n_orphans


def f_conv_repo_retag(repo_dir, dry_run=False, verbose=True):
    """Migrate a repo's on-disk filenames to the CURRENT _filter_fname scheme. Two cases it fixes:
    (1) PHYSICAL repos whose maps were named with the raw pixel wavelength (long float, e.g.
    'wl11.732669724420347') → the clean physical 'sf{cyc/deg}[_tf{Hz}]' names (uses the manifest's
    sf_cyc_deg/tf_hz); (2) the OLD lossy 2-decimal tf tag (0.025 → 'tf003') → the lossless _tf_tag.
    Renames each .npy whose filename no longer matches _filter_fname(...), rewrites the manifest 'file'
    fields, and re-canonicalizes isf/itf
    (_reindex_conv_filters — fixes the incremental-build index collision too). The manifest's tfreq
    VALUES are already exact (only the filename tag was lossy), so this just realigns the storage keys
    on disk. Idempotent: files already at the new tag are left alone. `dry_run` prints the planned
    renames without touching disk. Run it ONCE per existing repo before any further incremental build
    (else the new tag would reconvolve a duplicate map). Returns (n_renamed, n_unchanged)."""
    repo_dir = Path(repo_dir)
    man_path = repo_dir / 'manifest.json'
    manifest = json.loads(man_path.read_text())
    n_ren = n_same = 0
    for e in manifest['filters']:
        new_fn = _filter_fname(e['stat'], e['side'], e['odeg'], e['wavelength'], e.get('tfreq'),
                               sf_cyc_deg=e.get('sf_cyc_deg'), tf_hz=e.get('tf_hz'))
        if new_fn == e['file']:
            n_same += 1
            continue
        src, dst = repo_dir / e['file'], repo_dir / new_fn
        if verbose:
            print(f"  {'(dry) ' if dry_run else ''}rename {e['file']} → {new_fn}")
        if not dry_run:
            if dst.exists() and dst != src:
                raise FileExistsError(f'retag target already exists: {new_fn} (from {e["file"]})')
            if src.exists():
                src.rename(dst)
            e['file'] = new_fn
        n_ren += 1
    if not dry_run:
        manifest['filters'] = _reindex_conv_filters(manifest['filters'])
        man_path.write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f'  retag {repo_dir.name}: {n_ren} renamed, {n_same} already current'
              + (' (dry run)' if dry_run else ''))
    return n_ren, n_same


def f_conv_repo_meta(repo_dir):
    """Top-level repo meta from manifest.json (mouse / dset_name / gen_tag / num_samp / mon_hw /
    side_tags). `mon_hw` = actual rendered (H, W) — may be non-square (anisotropic render); falls
    back to (num_samp, num_samp) for legacy repos built before mon_hw was recorded."""
    man = json.loads((Path(repo_dir) / 'manifest.json').read_text())
    out = {k: man.get(k) for k in ('mouse', 'dset_name', 'gen_tag', 'num_samp', 'mon_hw', 'side_tags')}
    if out.get('mon_hw') is None and out.get('num_samp') is not None:
        out['mon_hw'] = [int(out['num_samp']), int(out['num_samp'])]
    return out


def f_conv_repo_to_h5(repo_dir, out_path=None, dtype='float32', compression=None,
                      chunk_t=2000, verbose=True):
    """Convert a conv_repo (per-filter .npy + manifest.json) → ONE browsable HDF5 (open in
    HDFView / ImageJ / h5py): a dataset per filter (named like the .npy stem) with attrs
    (kind/stat/side/odeg/theta/isf/wavelength/itf/tfreq/vmin/vmax) + a `frame_t` dataset + file
    meta. `dtype`: 'float32' (DEQUANTIZED real magnitudes — directly viewable; default) |
    'uint16' (raw quantized, half size; interpret via the vmin/vmax attrs). `compression`
    None|'gzip'|'lzf'. Streamed in time chunks (RAM-safe). Returns the written .h5 Path."""
    import h5py
    repo_dir = Path(repo_dir)
    out_path = Path(out_path) if out_path else repo_dir / 'conv_maps.h5'
    entries = f_list_conv_filters(repo_dir)
    with h5py.File(out_path, 'w') as hf:
        for kk, vv in f_conv_repo_meta(repo_dir).items():
            if vv is not None:
                hf.attrs[kk] = vv
        ftp = repo_dir / 'frame_t.npy'
        if ftp.exists():
            hf.create_dataset('frame_t', data=np.load(ftp))
        for i, e in enumerate(entries):
            q = np.load(repo_dir / e['file'], mmap_mode='r')          # uint16 (T,H,W)
            name = e['file'][:-4] if e['file'].endswith('.npy') else e['file']
            d = hf.create_dataset(name, shape=q.shape,
                                  dtype=('uint16' if dtype == 'uint16' else 'float32'),
                                  compression=compression)
            for s in range(0, q.shape[0], chunk_t):
                blk = np.asarray(q[s:s + chunk_t])
                d[s:s + chunk_t] = (blk if dtype == 'uint16'
                                    else _dequant_u16(blk, e['vmin'], e['vmax']))
            for kk in ('kind', 'stat', 'side', 'odeg', 'theta', 'isf', 'wavelength',
                       'itf', 'tfreq', 'vmin', 'vmax'):
                if e.get(kk) is not None:
                    d.attrs[kk] = e[kk]
            if verbose:
                print(f'  [{i + 1}/{len(entries)}] {name}', flush=True)
    if verbose:
        print(f'wrote {out_path}')
    return out_path


def f_conv_activity(repo_dir, select=None, reduce='absmean', chunk_t=2000, frame_ds=1):
    """Per-filter RESPONSE MAGNITUDE over time (frame-mean over monitor space) + a single
    time+space mean per filter — a diagnostic for CHOOSING orientations / wavelengths / temporal
    freqs (which filters the movie actually drives; especially useful for meng temporal freqs).
    `reduce`: 'absmean' (mean |value| per frame — right for signed meng opponent), 'mean' (signed
    frame-mean), 'rms'. `select` = optional predicate on a filter entry. `frame_ds` > 1 reads only
    every frame_ds-th FRAME (whole frames skipped → I/O cut ~frame_ds×; the trace is coarser in
    time but the which-filters-are-active picture is unchanged) — the main speed knob, since the
    diagnostic otherwise reads every dense map in full. Memory-safe (chunked). Returns (df, ts): df
    one row per filter (side/kind/stat/odeg/wavelength/tfreq/scalar), ts = (n_filter, T') traces."""
    import pandas as pd
    repo_dir = Path(repo_dir)
    entries = f_list_conv_filters(repo_dir)
    if select is not None:
        entries = [e for e in entries if select(e)]
    frame_ds = max(1, int(frame_ds))
    rows, traces = [], []
    _n = len(entries)
    for _i, e in enumerate(entries):
        print(f'  activity {_i + 1}/{_n}  {e["file"]}', flush=True)
        q = np.load(repo_dir / e['file'], mmap_mode='r')          # uint16 (T,H,W)
        idx = np.arange(0, q.shape[0], frame_ds)                  # frames to read
        ts = np.empty(idx.size, np.float32)
        for s in range(0, idx.size, chunk_t):
            sel = idx[s:s + chunk_t]
            x = _dequant_u16(np.asarray(q[sel]), e['vmin'], e['vmax'])   # reads only these frames
            if reduce == 'mean':
                ts[s:s + sel.size] = x.mean(axis=(1, 2))
            elif reduce == 'rms':
                ts[s:s + sel.size] = np.sqrt((x * x).mean(axis=(1, 2)))
            else:                                                 # absmean (magnitude)
                ts[s:s + sel.size] = np.abs(x).mean(axis=(1, 2))
        rows.append({k: e[k] for k in ('side', 'kind', 'stat', 'odeg', 'wavelength', 'tfreq')}
                    | {'scalar': float(ts.mean())})
        traces.append(ts)
    return pd.DataFrame(rows), (np.vstack(traces) if traces else np.zeros((0, 0), np.float32))


def f_conv_repo_scan(repo_dir, select=None, frame_ds=8, max_vals=60000, verbose=True):
    """Scan conv_repo .npy maps → per-filter VALUE stats + a value sample for histograms (diagnostic
    for dynamic-range / outlier problems, e.g. un-normalized motion energy). Reads every `frame_ds`-th
    frame (mmap) and dequantizes. `select` = optional predicate on a filter entry (e.g. _conv_filter
    (kinds=('meng',))). Returns (df, samples): df one row per filter (kind/side/odeg/wavelength/tfreq,
    vmin/vmax, mean/std/absmax, p50_abs/p99_abs/p999_abs, tail = absmax/p50_abs — high = heavy outlier
    tail); samples[name] = 1-D dequantized value sample (≤ max_vals) for the histogram."""
    import pandas as pd
    repo_dir = Path(repo_dir)
    entries = f_list_conv_filters(repo_dir)
    if select is not None:
        entries = [e for e in entries if select(e)]
    frame_ds = max(1, int(frame_ds))
    rows, samples = [], {}
    for _i, e in enumerate(entries):
        q = np.load(repo_dir / e['file'], mmap_mode='r')             # uint16 (T,H,W)
        idx = np.arange(0, q.shape[0], frame_ds)
        x = _dequant_u16(np.asarray(q[idx]), e['vmin'], e['vmax']).ravel()
        name = e['file'][:-4] if e['file'].endswith('.npy') else e['file']
        ax = np.abs(x)
        p50, p99, p999 = (float(v) for v in np.percentile(ax, [50, 99, 99.9]))
        rows.append({'name': name, 'kind': e['kind'], 'side': e['side'], 'odeg': e['odeg'],
                     'wavelength': e['wavelength'], 'tfreq': e.get('tfreq'),
                     'vmin': e['vmin'], 'vmax': e['vmax'], 'mean': float(x.mean()),
                     'std': float(x.std()), 'absmax': float(ax.max()),
                     'p50_abs': p50, 'p99_abs': p99, 'p999_abs': p999,
                     'tail': float(ax.max() / (p50 + 1e-30))})
        samples[name] = (x[np.linspace(0, x.size - 1, max_vals).astype(int)]
                         if x.size > max_vals else x)
        if verbose:
            print(f"  {name:34s} range[{e['vmin']:.3g},{e['vmax']:.3g}] std={rows[-1]['std']:.3g} "
                  f"p50|abs|={p50:.3g} p99.9|abs|={p999:.3g} tail={rows[-1]['tail']:.1f}", flush=True)
    return pd.DataFrame(rows), samples


def _conv_filter(kinds=None, wavelengths=None, odegs=None, tfreqs=None, select=None,
                 sf_cyc_deg=None, tf_hz=None):
    """Build a filter predicate on a conv_repo entry from convenient field limits (all ANDed;
    None = don't filter that field). `kinds`/`wavelengths`/`tfreqs`/`odegs` are value lists (px /
    cyc-frame / int-deg); `sf_cyc_deg`/`tf_hz` are the PHYSICAL selectors (matched to the entry's
    stored physical fields, rounded to 4/3 dp). `select` = an extra predicate ANDed on top. Shared by
    f_load_conv_blocks / f_conv_activity so limiting is uniform across the scripts."""
    # tolerate a scalar / bare string for any field (wrap → 1-elem list)
    kset = ({kinds} if isinstance(kinds, str) else set(kinds)) if kinds is not None else None
    wset = {float(w) for w in np.atleast_1d(wavelengths)} if wavelengths is not None else None
    tset = {float(t) for t in np.atleast_1d(tfreqs)} if tfreqs is not None else None
    oset = {int(o) for o in np.atleast_1d(odegs)} if odegs is not None else None
    sfset = {round(float(s), 4) for s in np.atleast_1d(sf_cyc_deg)} if sf_cyc_deg is not None else None
    hzset = {round(float(h), 3) for h in np.atleast_1d(tf_hz)} if tf_hz is not None else None

    def _match(e):
        if kset is not None and e['kind'] not in kset:
            return False
        if wset is not None and float(e['wavelength']) not in wset:
            return False
        if oset is not None and int(e['odeg']) not in oset:
            return False
        # tfreq limit applies ONLY to entries that HAVE a temporal freq (meng); gabor (tfreq=None)
        # is never excluded by it — so `tfreqs` can be set alongside gabor without dropping it.
        if tset is not None and e.get('tfreq') is not None and float(e['tfreq']) not in tset:
            return False
        if sfset is not None and e.get('sf_cyc_deg') is not None and round(float(e['sf_cyc_deg']), 4) not in sfset:
            return False
        if hzset is not None and e.get('tf_hz') is not None and round(float(e['tf_hz']), 3) not in hzset:
            return False
        return select is None or select(e)
    return _match


def f_load_conv_blocks(repo_dir, ds_list, n_pca=50, kinds=('gabor', 'meng'),
                       wavelengths=None, odegs=None, tfreqs=None,
                       sf_cyc_deg=None, tf_hz=None,
                       select=None, side_tags=None, labels=None, verbose=True):
    """SHARED conv_repo → built_blocks loader (used by cebra / feature-viz / feature-recon).
    For each kind × ds in `kinds`×`ds_list`: point-sample the repo maps (f_sample_conv_to_block),
    PCA (f_apply_block_pca), and assemble a built_blocks-compatible entry named '<label>_ds<ds>'
    (label from `labels`, default gabor→'gaborS', meng→'mengS'). Each entry of `ds_list` is a
    scalar stride (isotropic) OR a (ds_h, ds_w) tuple (ANISOTROPIC h/w strides → name '<label>_ds8x16');
    a single anisotropic stride must be WRAPPED, ds_list=[(8, 16)] (a bare (8, 16) reads as two
    isotropic strides 8 and 16). LIMIT what's loaded with `kinds`
    (e.g. ('gabor',) for gabor-only), and by PHYSICAL units `sf_cyc_deg` (cyc/deg, e.g. (0.1,)) /
    `tf_hz` (meng Hz, e.g. (1.5,)) — or the legacy pixel selectors `wavelengths` (px) / `tfreqs`
    (cyc/frame) — plus `odegs` (orientation degrees). Value lists, None = no limit; `select` = an
    extra predicate. `n_pca` int/dict/None (None=raw). `verbose` → per-filter load + RAM + PCA-start
    prints. Returns (blocks dict, side_tags). Pair with f_conv_repo_meta for num_samp/mon_hw +
    f_conv_frame_t for the clock."""
    from f_visual_features import f_apply_block_pca
    if isinstance(kinds, str):                    # ('gabor') is a STRING, not a tuple — wrap it
        kinds = (kinds,)
    if side_tags is None:
        side_tags = list(f_conv_repo_meta(repo_dir)['side_tags'])
    labels = labels or {'gabor': 'gaborS', 'meng': 'mengS'}
    _lim = _conv_filter(wavelengths=wavelengths, odegs=odegs, tfreqs=tfreqs,
                        sf_cyc_deg=sf_cyc_deg, tf_hz=tf_hz, select=select)
    _all = f_list_conv_filters(repo_dir)
    blocks = {}
    for kind in kinds:
        if not any(e['kind'] == kind and _lim(e) for e in _all):   # repo has none → skip, don't crash
            print(f'  (no {kind} filters match the load filters — skipped)', flush=True)
            continue
        for ds in (ds_list if isinstance(ds_list, (list, tuple)) else np.atleast_1d(ds_list)):
            ds = _norm_ds(ds)
            bn = f'{labels.get(kind, kind)}_ds{_ds_tag(ds)}'
            if verbose:
                print(f'loading {bn} ...', flush=True)
            sel = (lambda e, k=kind: e['kind'] == k and _lim(e))
            X, names = f_sample_conv_to_block(repo_dir, ds, select=sel, side_tags=side_tags, verbose=verbose)
            if verbose:
                print(f'  {bn}: PCA (n_pca={n_pca}) on {X.shape} ...' + _ram_str(), flush=True)
            Xp, pns, pca, pg = f_apply_block_pca(X, names, bn, n_pca)
            blocks[bn] = {'X': Xp, 'names': pns, 'raw_names': names, 'raw_dim': X.shape[1],
                          'pca': pca, 'pca_groups': pg, 'kind': kind,
                          'params': {'pool': 'sample', 'sample_ds': ds}}   # ds scalar or (h,w) tuple
    return blocks, list(side_tags)


def f_load_conv_combined(repo_dir, ds, n_pca=200, kinds=('gabor', 'meng'), wavelengths=None,
                         odegs=None, tfreqs=None, label='convS', side_tags=None, verbose=True):
    """Sample MULTIPLE conv kinds (default gabor + meng) into ONE raw matrix and fit a SINGLE joint
    PCA → one COMBINED block `{label}`. Gabor magnitude ≈ motion-energy at temporal-freq ≈ 0, so the
    two share spatial structure; combining them BEFORE the PCA (vs the default of PCA-ing each kind
    separately to `n_pca` then merging) folds the two truncation steps into one and can't discard a
    shared low-variance signal that only one kind alone would drop. Same block shape as an
    f_load_conv_blocks entry (X/names/pca/raw_names). Point-sampled at stride `ds`, sides concatenated
    like the per-kind loader. Returns ({label: block}, side_tags) — set encoding feature_block=label,
    n_feature_pcs=None (this IS the single feature PCA)."""
    from f_visual_features import f_apply_block_pca
    if side_tags is None:
        side_tags = list(f_conv_repo_meta(repo_dir)['side_tags'])
    sel = _conv_filter(kinds=kinds, wavelengths=wavelengths, odegs=odegs, tfreqs=tfreqs)
    if not any(sel(e) for e in f_list_conv_filters(repo_dir)):
        raise ValueError(f'f_load_conv_combined: no filters match kinds={kinds} in {repo_dir}')
    ds = _norm_ds(ds)
    if verbose:
        print(f'{label}: sampling {"+".join(kinds)} raw (ds={ds}) ...', flush=True)
    X, names = f_sample_conv_to_block(repo_dir, ds, select=sel, side_tags=side_tags, verbose=verbose)
    if verbose:
        print(f'{label}: fitting joint PCA (n_pca={n_pca}) on {X.shape} ...' + _ram_str(), flush=True)
    Xp, pns, pca, pg = f_apply_block_pca(np.asarray(X, float), names, label, n_pca)
    if verbose:
        print(f'{label}: PCA done -> {np.asarray(Xp).shape}' + _ram_str(), flush=True)
    block = {'X': np.asarray(Xp, np.float32), 'names': pns, 'raw_names': names, 'raw_dim': X.shape[1],
             'pca': pca, 'pca_groups': pg, 'kind': 'combined',
             'params': {'pool': 'sample', 'sample_ds': ds, 'kinds': tuple(kinds)}}
    return {label: block}, list(side_tags)
