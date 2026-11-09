# -*- coding: utf-8 -*-
"""
f_feature_search.py — feature-PARAMETER search on ONE dataset, scored by ENCODE + DECODE.

The neural-scored sibling of f_feature_optim.py (which scores feature params by MOVIE
reconstruction, ignoring neurons). Here each candidate feature config is resolved on ONE
loaded session through the plan grammar (f_default_feature_plan / f_resolve_features — so
separate/combined/raw PCA, sampling ds, filter subset, and feature COMBINATIONS are all
expressible). By DEFAULT (force_build=True) every feature is REGENERATED fresh on the GPU
— the cache is bypassed so the search spans the full parameter space, not the few cached
settings. The monitor render runs ONCE (shared across configs) and each unique RAW extraction
is CONVOLVED ONCE and memoized, so comb/sep/raw (and prep variants) of one extraction reuse a
single conv — only the PCA + prep re-run. (Derived combos / cache loads use the full resolve
and rebuild their deps.) Each config is then scored by f_dataset_scan.f_score_representations:
    decode_r2      neural PCs → feature block (leak-free within-fold PCA, variance-weighted)
    encode_r_med   feature block → neural (per-neuron blocked-CV, f_encode_per_neuron)
    encode_frac_sig  fraction of cells with encode r > r_sig
Returns a tidy ranked table (config × block × decode/encode/n_dim). Compute-only; the driver
+ the inline candidate-config list live in VR_feature_search.py.

Feature preprocessing (normalize / whiten) is the `feat_prep` axis (per-config or global; expand with
f_prep_configs): None | 'zscore' (unit-variance columns) | 'whiten' (PCs ÷ √eigenvalue). The search
scores with proc_before_cv=True — no within-fold standardization of the predictor — so feat_prep moves
ENCODE (ridge penalty depends on column scale) as well as DECODE (features = target, variance-weighted
R²: raw/combined weights the top PCs, whiten/zscore equal-weights all PCs).

A "config" is a dict:
    {'name':   str,                    # label for this candidate (shown in the table/plot)
     'blocks': [block names to SCORE], # subset of the plan (deps auto-pulled; e.g. 'gaborS+mengS')
     'plan':   {block: override} | None,   # f_default_feature_plan overrides (origin/inputs/n_pca/…)
     'source': {cache_tag: ...} | None}    # per-config f_resolve_features source overrides
The plan's per-block `n_pca` IS the separate/combined/raw switch on a FRESH build (dict = per-filter
separate, int = whole-block combined, None = raw); for CACHE loads the granularity is baked at
gen time and picked by `source['cache_tag']` (…_pcasep / …_pcacomb / …_pcaraw) — see f_variant_configs.

FEATURE-GRID instructions (2026-07-20, ONE param builder — replaced the 3 two-layer builders): a config
with a `'grid'` key uses `f_build_grid`, a memory-safe streaming builder. The bank = kind's orientation ×
wavelength(SF) × temporal-freq grid. `f_feature_grid(kind, marginalize=(...), ...)` picks which axes to
MARGINALIZE (pool); the REMAINING axes form a param matrix — one scored CELL per remaining-axis-value
combo, each pooling the marginalized axes → a matrix of decode/encode scores. marginalize=() → per filter,
{'ori'} → per (sf,tf), {'ori','wl'} → per tf, {'ori','wl','tf'} → whole bank. `layer1_proc` (per individual
filter) and `layer2_proc` (per cell) are UNIFIED f_proc step-lists (f_feature_helpers.f_proc: ordered
[('norm',…),('pca',n),('whiten',True)], each optional). f_build_grid streams: conv ONE (sf,tf) at a time
(orientation always batched), apply layer1_proc per filter, FREE the raw, route filters to cells by the
non-marginalized axes, apply layer2_proc per cell. Peak RAM ≈ one conv's raw + Σ reduced.
`f_score_grid(res,metric)` pivots the cell scores into the ori×sf×tf matrix.

UNIFIED PROCESSING (2026-07-20): the same `f_proc` step-list is used everywhere — grid layer1/layer2
(global, build-time), AND within-fold on the decode/encode predictor (leak-free) via `neural_proc`/
`feature_proc` passed to f_score_representations → f_blocked_cv_r2's optional `pred_proc` (default None =
byte-identical to the old standardize+PCA, so cebra/encoding/decoding are untouched). Render-once /
memoization / scoring unchanged.

Created 2026-07-18.
"""

import hashlib
import json
import time
import numpy as np
import pandas as pd

from f_utils import f_gauss_smooth
from f_feature_helpers import f_zscore, f_proc, f_num_tag
from f_feature_source import (f_default_feature_plan, f_resolve_features, _whiten_block, _layer1_names,
                              _normalize_spec, f_pyramid_specs, NO_TF_KINDS, f_marg_axes,
                              DEFAULT_MARGINALIZE,
                              f_marg_spec, f_resolve_feat_sets)
from f_visual_features import f_apply_block_pca, f_combine_blocks_pca, f_pca_group_cols
from f_dataset_scan import f_score_representations, f_nan_metrics
from f_decoding import f_imaging_fs

_VAR_SUFFIX = {'combined': '_pcacomb', 'separate': '_pcasep', 'raw': '_pcaraw'}
_FEAT_PREPS = (None, 'zscore', 'whiten')   # feature preprocessing before scoring (see f_prep_configs)


def _apply_feat_prep(X, prep):
    """Feature preprocessing applied to a block's X before scoring. 'zscore' = unit-variance columns
    (f_zscore); 'whiten' is done at resolve time via _whiten_block (PCs ÷ √eigenvalue) so here it's a
    no-op; None = as-is. Affects mainly DECODE (feature = target, variance-weighted R²); the encode
    scorer standardizes features internally, so prep is ~a no-op there."""
    if prep == 'zscore':
        return f_zscore(np.asarray(X, float))
    return X   # None / 'whiten' (whiten applied to PCs by _reduce_raw / f_resolve_features)


def _extraction_sig(name, inputs):
    """Hashable signature of a RAW extraction (the render→conv→sample recipe, PCA/prep-independent).
    Configs that share a signature convolve ONCE — comb/sep/raw of one extraction, or a duplicated
    single-block config across preps."""
    return (name, json.dumps(inputs, sort_keys=True, default=str))


def _raw_block(session, build_cfg, name, inputs, prerendered, cache, verbose, timing=False):
    """Build (or reuse from `cache`) the RAW (n_pca=None) block for a single-input extraction, memoized
    by signature so the GPU conv runs once per unique extraction. Returns (raw_block, frame_t)."""
    sig = _extraction_sig(name, inputs)
    if sig in cache:
        if verbose:
            print(f'    reuse raw {name!r} (already built this run)')
        return cache[sig]
    # SSD autosave lives DOWNSTREAM now (f_visual_features.build_visual_blocks, keyed on the resolved
    # extractor params) so the search and VR_ca_cebra share one store — nothing to do here beyond the
    # in-run memo above. Opt in with build_cfg['conv_autosave']=True.
    plan = f_default_feature_plan(**{name: {'origin': 'build', 'n_pca': None, 'inputs': inputs}})
    built, frame_t, _st, _mv = f_resolve_features(
        session, plan, build_cfg, only=[name], want_movie=False, verbose=verbose, prerendered=prerendered)
    cache[sig] = (built[name], np.asarray(frame_t).ravel())
    return cache[sig]


def _reduce_raw(raw, name, n_pca, prep, pre_pca_zscore=False):
    """Apply the feature-normalization chain to a memoized RAW block → a scorable block:
    [pre_pca_zscore: z-score raw cols] → [PCA: n_pca granularity] → [prep: post-PCA zscore/whiten].
    Does NOT mutate the cached raw (a fresh dict/array is returned)."""
    rn = raw.get('raw_names') or raw['names']
    X = raw['X']
    if pre_pca_zscore:
        X = f_zscore(np.asarray(X, float))            # z-score raw feature columns BEFORE PCA
    Xp, names, pca, pg = f_apply_block_pca(np.asarray(X), rn, name, n_pca)
    out = {**raw, 'X': Xp, 'names': names, 'pca': pca, 'pca_groups': pg, 'raw_names': rn,
           'raw_dim': int(np.asarray(raw['X']).shape[1])}
    out.pop('whitened', None)
    if prep == 'whiten':
        _whiten_block(out)          # PCs ÷ √eigenvalue (no-op when pca is None, i.e. raw granularity)
    else:
        out['X'] = _apply_feat_prep(out['X'], prep)
    return out


class _NestedFit:
    """Prefix of an exact eigen-decomposition — the fields consumers read from a sklearn PCA/TruncatedSVD."""
    def __init__(self, V, ratio, var, mean):
        self.components_ = V.T                            # (k, d)
        self.explained_variance_ratio_ = ratio
        self.explained_variance_ = var
        self.mean_ = mean                                 # None for svd (uncentered)
        self.n_components = V.shape[1]

    def transform(self, X):
        X = np.asarray(X, np.float32)
        if self.mean_ is not None:
            X = X - self.mean_.astype(np.float32)
        return X @ self.components_.T.astype(np.float32)


def _l1_decomp(X, op, chunk=20000):
    """Exact full decomposition of one filter's columns, fit ONCE and sliced per sweep value.
    pca: eigh of the covariance. svd: eigh of XᵀX (uncentered), ratios = projected variance ÷ total
    variance (sklearn TruncatedSVD's definition). Returns (V desc-sorted (d,d), ratio, var, mean)."""
    T, d = X.shape
    G = np.zeros((d, d)); mu = np.zeros(d)
    for i in range(0, T, chunk):                          # float64 accumulation in row chunks
        Xc = np.asarray(X[i:i + chunk], np.float64)
        G += Xc.T @ Xc; mu += Xc.sum(0)
    mu /= T
    C = G / T - np.outer(mu, mu)                          # covariance (ddof 0)
    tot = float(np.trace(C)) or 1.0
    w, V = np.linalg.eigh(C if op == 'pca' else G)
    V = V[:, ::-1]
    var = np.einsum('ij,jk,ki->i', V.T, C, V)             # variance of each projection
    return V, var / tot, var, (mu if op == 'pca' else None)


def _l1_nested(X, op, n, dec):
    """Apply one ('pca'|'svd', n) step from a cached decomposition with the SAME k / identity rules as
    _pca_fit / _svd_fit. Returns (scores, fit_or_None)."""
    V, ratio, var, mu = dec
    d = X.shape[1]
    cum = np.cumsum(ratio)
    if isinstance(n, (tuple, list)) and len(n) == 2:     # (variance-fraction, cap)
        cap = min(int(n[1]), d if op == 'pca' else d - 1)
        k = min(max(int(np.searchsorted(cum[:cap], float(n[0])) + 1), 1), cap)
        if op == 'pca' and k >= d:
            return X, None
    elif isinstance(n, float) and 0 < n < 1:              # variance fraction
        if op == 'pca':
            k = min(int(np.searchsorted(cum, n, side='right') + 1), d)
        else:
            cap = d - 1
            if cum[cap - 1] < n:                          # d-1 comps miss the fraction → keep all d
                return X, None
            k = min(max(int(np.searchsorted(cum[:cap], n) + 1), 1), cap)
    else:
        k = int(n)
        if k >= d:
            return X, None
    fit = _NestedFit(V[:, :k], ratio[:k], var[:k], mu)
    return fit.transform(X), fit


# ── Feature-grid builder (ONE param builder; memory-safe streaming: conv (sf,tf) → PCA → free raw) ──

def f_feature_grid(kind, marginalize=(), *, sf_cyc_deg=(0.13,), tf_hz=(1.0, 2.0, 3.0),
                   n_orient=4, odegs=None, ds=4, ds_factor=None, layer1_proc=None, layer2_proc=None,
                   tag=None, transform='signed_sqrt', output='opponent', stats=('magnitude',),
                   reduce='sample', n_pca_layer1=None, n_pca_layer2=None, input_norm=None, prep=None):
    """THE feature-grid builder. The bank = kind's orientation × wavelength(SF) × temporal-freq grid
    (gabor: ori × sf, no tf). `marginalize` ⊆ {'ori','wl'/'sf','tf'} = axes to POOL/collapse; the
    REMAINING axes form a PARAM MATRIX — one scored CELL per remaining-axis-value combo, each pooling the
    marginalized axes. marginalize=() → per filter; {'ori'} → per (sf,tf); {'ori','wl'} → per tf;
    {'ori','wl','tf'} → whole bank.
    Orientation: `odegs` = explicit angles in DEGREES (e.g. (0, 90)) → those exact orientations; when
    None, `n_orient` evenly-spaced angles in [0, 180). Cell names carry the angle either way.
    `layer1_proc` (applied per INDIVIDUAL filter) and `layer2_proc` (applied per CELL) are f_proc
    step-lists — ordered [(op,arg),…] of ('norm',…)/('pca',n)/('whiten',True); None/[] = identity. LEGACY
    kwargs n_pca_layer1 / n_pca_layer2 / input_norm / prep are still accepted and folded into the procs.
    `tag` suffixes the config name (':<tag>') so a PROC sweep — same kind+marginalize, different
    layer1/layer2 procs — gets distinct config names (else they collide in the scorecard / f_score_grid).
    Returns [ONE config]; a marginalize sweep = sum([f_feature_grid(kind, marginalize=m, ...) for m], []);
    a proc sweep = sum([f_feature_grid(..., layer2_proc=p, tag=nm) for nm, p in proc_sets], [])."""
    kname = kind if isinstance(kind, str) else '+'.join(kind)
    marg = f_marg_spec(marginalize)                      # bare 'all' is ONE axis, not ('a','l','l')
    odegs = tuple(float(a) for a in np.atleast_1d(odegs)) if odegs is not None else None
    if odegs is not None:                                # explicit angles → n_orient follows
        n_orient = len(odegs)
    if layer1_proc is None:                              # fold legacy kwargs → step-list
        layer1_proc = [('pca', n_pca_layer1)] if n_pca_layer1 else []
    if layer2_proc is None:
        layer2_proc = (([('norm', input_norm)] if input_norm else [])
                       + ([('pca', n_pca_layer2)] if n_pca_layer2 else [])
                       + ([('whiten', True)] if prep in ('whiten', 'zscore') else []))
    sf_cyc_deg = tuple(float(s) for s in np.atleast_1d(sf_cyc_deg))    # tolerate a scalar (e.g. 0.13)
    tf_hz = tuple(float(t) for t in np.atleast_1d(tf_hz))              # tolerate a scalar (e.g. 1.0)
    return [{'name': f'{kname}:marg={"+".join(marg) or "none"}' + (f':{tag}' if tag else ''),
             'grid': {'kind': kind, 'marginalize': marg, 'sf_cyc_deg': sf_cyc_deg,
                      'tf_hz': tf_hz, 'n_orient': n_orient, 'odegs': odegs, 'ds': ds,
                      'ds_factor': ds_factor,
                      'reduce': reduce, 'transform': transform, 'output': output, 'stats': tuple(stats),
                      'layer1_proc': list(layer1_proc), 'layer2_proc': list(layer2_proc)},
             'feat_prep': None}]


def f_feature_specs(specs, *, marginalize=(), layer1_proc=None, layer2_proc=None, tag=None):
    """FEATURE-CREATOR: a LIST of level-1 feature spec dicts (+ optional LEVEL-2) → [ONE grid config] for
    f_feature_search. Each spec = one feature with its own sf/tf/ds/ori (see f_feature_source._normalize_
    spec; omitted keys → defaults; aliases sf/tf/samp/ori). `layer1_proc` = per-filter LEVEL-1 proc;
    `marginalize` (⊆ {'ori','sf','tf'}) + `layer2_proc` = the optional LEVEL-2 (pool feature subsets).
    Generators (f_pyramid_specs) PRODUCE `specs`; hand-written lists (even mixed kinds) work too. Streams
    via f_iter_grid_cells over the explicit inputs."""
    inputs = [_normalize_spec(s) for s in specs]
    kinds = {i['source'] for i in inputs}
    kname = next(iter(kinds)) if len(kinds) == 1 else 'mixed'
    marg = f_marg_spec(marginalize)                      # bare 'all' is ONE axis, not ('a','l','l')
    return [{'name': f'{kname}:specs' + (f':{tag}' if tag else ''),
             'grid': {'kind': kname, 'inputs': inputs, 'marginalize': marg,
                      'layer1_proc': list(layer1_proc or []), 'layer2_proc': list(layer2_proc or [])},
             'feat_prep': None}]


def f_pyramid_grid(kind='gabor', *, sf_cyc_deg=None, sf_range=None, n_sf=5, cyc_fov=None,
                   hFOV_deg=98.4, n_orient=8, odegs=None, ds='nyquist', samp_factor=1.0,
                   marginalize=DEFAULT_MARGINALIZE, tag=None, transform=None, output='opponent',
                   stats=('magnitude',), layer1_proc=None, layer2_proc=None, tf_hz=(1.0,),
                   n_sf_sub=None, n_ori_sub=None, patch_snap=None, **kind_kw):
    """Kay/Gallant WAVELET-PYRAMID search config (compare a gabor pyramid vs meng). Builds the SF
    param-space from a RANGE (f_pyramid_specs) and Nyquist-samples per SF, recombined via layer1/layer2
    (f_feature_specs). SF list: sf_cyc_deg | cyc_fov/hFOV_deg | geomspace(sf_range, n_sf).
    marginalize=('ori',) → one cell per SF (a pyramid level); () = per filter. Add 'luminance' to `stats`
    for the DC wavelet. `**kind_kw` forwards kind-specific knobs (wavelet family/decimate/levels, …);
    f_pyramid_specs validates them against that kind's KIND_KNOBS."""
    specs = f_pyramid_specs(kind, sf_cyc_deg=sf_cyc_deg, sf_range=sf_range, n_sf=n_sf, cyc_fov=cyc_fov,
                            hFOV_deg=hFOV_deg, n_orient=n_orient, odegs=odegs, ds=ds, samp_factor=samp_factor,
                            transform=transform, output=output, stats=stats, tf_hz=tf_hz,
                            n_sf_sub=n_sf_sub, n_ori_sub=n_ori_sub, patch_snap=patch_snap,
                            **kind_kw)
    return f_feature_specs(specs, marginalize=marginalize, layer1_proc=layer1_proc,
                           layer2_proc=layer2_proc, tag=tag)


def f_pca_tag(n, op='pca'):
    """Short DISTINCT tag for one ('pca', n) / ('svd', n) spec — for naming a component-sweep pass.
    int → 'pca50' (component count) | float → 'pcav0.95' (variance fraction) |
    (frac, cap) → 'pcav0.99c200' | None/0 → 'pcaraw'. The 'v' matters: without it a count of 50 and a
    fraction of 0.5 would collide in the scorecard. `op='svd'` swaps the stem ('svd50', 'svdv0.95', …)
    so a centered and an uncentered pass of the same size stay distinct."""
    if not n:
        return f'{op}raw'
    if isinstance(n, (tuple, list)) and len(n) == 2:
        return f'{op}v{f_num_tag(n[0])}c{int(n[1])}'
    return f'{op}{int(n)}' if isinstance(n, (int, np.integer)) else f'{op}v{f_num_tag(n)}'


def _pix_step_tag(op, arg):
    """Compact name tag for one proc step in a pix config name."""
    if op in ('pca', 'svd'):
        return f_pca_tag(arg, op)
    if op == 'norm':
        return str(arg)
    if op == 'whiten':
        return 'whiten'
    return f'{op}{arg}'


def f_pix_config(proc=(('pca', 300),), name=None):
    """RAW-movie pixel baseline config (the real visual input) for the search — the reference to compare
    the gabor/fft/meng grids against. The pix block is delivered RAW (every pixel, n_pix_pca=None) and
    the ENTIRE ordered `proc` step-list is then applied by the ordinary block-proc path, so pix goes
    through the SAME f_proc stages as a bank's layer1/layer2 and has NO pix-specific reduction of its
    own. `proc=[]` → raw pixels, untouched. Name derived from the ordered steps, so order is visible
    ('pix:zscore_pca300' = correlation-PCA vs 'pix:pca300_zscore' = z-scored PCs).

    NB the build flattens the movie to (T, H·W·n_side) float32 either way, so moving the PCA out of the
    build costs no extra peak there; it does mean f_resample_to_imaging interpolates all ~30k raw columns
    to the imaging clock instead of a handful of PCs (it column-chunks in float32, so this is bounded,
    just slower)."""
    steps = list(proc or [])
    inp = {'source': 'pix', 'reduce': 'raw', 'n_pix_pca': None}      # raw pixels; all reduction in `proc`
    nm = name or ('pix:' + ('_'.join(_pix_step_tag(op, a) for op, a in steps) or 'raw'))
    return {'name': nm, 'blocks': ['pix'],
            'plan': {'pix': {'origin': 'build', 'n_pca': None, 'inputs': [inp]}},
            'proc': steps or None}


def f_feature_configs(feat_sets, strict=True, tag_suffix=None, **common):
    """ONE declarative dict holding EVERY parameter → a flat config list for f_feature_search.

    `feat_sets` is self-contained: a reserved `'_common'` entry carries the DEFAULTS shared by all sets
    (cyc_fov / odegs / marginalize / ds / layer1_proc / layer2_proc / …) and every other key is one
    feature set, `{'kind': 'gabor'|'fft'|'meng'|'pix', **knobs}`, stating only what makes it different.
    A knob repeated in an entry OVERRIDES the default, which is all a variant needs — no second pass,
    no selector list. Nothing has to be supplied later:

        feat_sets = {
            '_common':    dict(cyc_fov=(1, 2, 4, 8, 16), odegs=(0, 45, 90), marginalize='all',
                               layer2_proc=[('pca', 100)]),
            'gabmag':     dict(kind='gabor', stats=('magnitude',)),
            'gabmag_raw': dict(kind='gabor', stats=('magnitude',), layer2_proc=[]),   # overrides
            'mengopp':    dict(kind='meng', tf_hz=(0.25, 0.5, 1.0, 2.0), output='opponent'),
            'pix':        dict(kind='pix'),
        }
        configs = f_feature_configs(feat_sets)

    The dict KEY becomes the config `tag` → scorecard label '<kind>:specs:<name>' (pix → 'pix:<name>').
    Dispatch: 'pix' → f_pix_config; every other kind → f_pyramid_grid(kind, tag=name, **knobs), a plain
    pass-through, so anything f_pyramid_grid accepts still works and hand-building a config list stays
    available. pix takes the SAME layer1_proc / layer2_proc as every other kind — it is one block with
    no filter axis, so the two simply run back-to-back on it and there is no pix-specific reduction
    stage. Only the AXIS knobs (cyc_fov / odegs / marginalize / stats / …) are withheld from pix, since
    the raw movie has no ori/sf/tf axes to place or pool. So one '_common' entry drives every kind.

    PRECEDENCE: an entry's own key > a `common` KWARG > `'_common'`. The kwarg layer exists for a
    PROGRAMMATIC sweep — many values across every set at once, without writing each variant out. Give
    each pass a `tag_suffix` so names don't collide in the scorecard; a set that PINS the swept knob
    keeps its own value, since the entry still wins:

        configs = sum([f_feature_configs(feat_sets, layer2_proc=[('pca', n)], tag_suffix=f'pca{n}')
                       for n in (50, 100, 200, 400)], [])

    `strict` (default) raises on an unknown `kind`, and on a per-entry knob the chosen kind IGNORES
    (tf_hz/output on gabor or fft, stats/n_sf_sub on meng). Knobs inherited from `'_common'` or a kwarg
    are exempt — a shared default need not apply to every kind. This is the guard for the silent failure
    where a set of 'fft'/'meng' entries was built through f_pyramid_grid('gabor', …): the kind never
    reached the builder, so gabor banks were scored under fft/meng names with no error anywhere.

    The dict is READ by f_feature_source.f_resolve_feat_sets — the same canonical reader the plan side
    (f_vis_sets_to_plan → VR_ca_cebra) uses, so one dict cannot mean two different things."""
    out = []
    for name, tag, kind, knobs in f_resolve_feat_sets(feat_sets, strict=strict,
                                                      tag_suffix=tag_suffix, **common):
        if kind == 'pix':
            # pix is ONE block, so there is no per-filter/per-cell distinction: layer1 then layer2 run
            # back-to-back on it, in list order, via the ordinary block-proc path. No pix-specific stage.
            out += [f_pix_config(proc=list(knobs.get('layer1_proc') or [])
                                 + list(knobs.get('layer2_proc') or []), name=f'pix:{tag}')]
        else:
            out += f_pyramid_grid(kind, tag=tag, **knobs)
    return out


def f_feature_sizes(res, sort_by='decode_peak_gb', ram_gb=None, verbose=True):
    """SIZE table for every feature block that was scored — the dataset-size monitor.

    Because the grid build STREAMS (each cell is scored then freed, so all blocks never coexist), this
    reads the shapes back off the search result `res` rather than a live registry: one row per scored
    block with `n_frames` × `n_dim` (the width actually decoded), `raw_dim` (width BEFORE the layer/PCA
    proc — so you can see what the reduction bought), `block_mb`, and `decode_peak_gb` = the projected
    peak of the decode step (f_dataset_scan.DECODE_PEAK_FACTOR · T · n_dim · 8 bytes, measured factor).
    Also what the kept components COST: `l1_var%` (per-filter layer1 PCAs, width-weighted) / `var%`
    (the block/layer2 PCA) / `varTot%` (their product = fraction of the RAW variance still present).
    A block that came out the SAME WIDTH it went in dropped nothing and reads 100.0, so the column
    compares straight down against the reduced rows; '-' means the width DID change but no PCA object
    was attached, i.e. unknown rather than lossless. `reduction` still tells you whether a reduction
    ran at all. `ram_gb` adds a `ram_fit` headroom VERDICT per row — ok /
    tight (>50%) / OVER (>90%) — not a count.
    `sort_by`: any column (descending) | None / 'none' = keep the INPUT order, i.e. the order the
    configs were listed in (`res` comes back in build order). Returns the DataFrame."""
    import pandas as pd
    from f_dataset_scan import DECODE_PEAK_FACTOR
    cols = ['config', 'block', 'n_frames', 'raw_dim', 'n_dim', 'l1_var_kept', 'var_kept',
            'var_kept_total', 'block_mb', 'decode_peak_gb']
    have = [c for c in cols if c in res.columns]
    out = res.loc[res.get('status', 'ok').eq('ok') if 'status' in res else slice(None), have].copy()
    if 'raw_dim' in out and 'n_dim' in out:                  # how much the proc reduced this block
        out['reduction'] = [('-' if r == d else f'{r / max(d, 1):.0f}x')
                            for r, d in zip(out['raw_dim'], out['n_dim'])]
    # Variance the kept components explain, formatted so the column reads straight down. NaN means no
    # PCA object. At LAYER 1 that always means the stage ran no PCA (f_iter_grid_cells sets 1.0 when it
    # ran one without reducing, and a non-grid block has no layer-1 stage at all) → nothing dropped →
    # 100.0. At the BLOCK level NaN is only lossless if the width is unchanged; if it narrowed by some
    # other means we cannot claim a number → '-'. `reduction` still shows whether a reduction ran.
    _nored = ([r == d for r, d in zip(out['raw_dim'], out['n_dim'])]
              if ('raw_dim' in out and 'n_dim' in out) else [False] * len(out))
    for src, dst, always in (('l1_var_kept', 'l1_var%', True), ('var_kept', 'var%', False),
                             ('var_kept_total', 'varTot%', False)):
        if src in out:
            out[dst] = [(f'{100 * v:.1f}' if v == v else ('100.0' if (always or nr) else '-'))
                        for v, nr in zip(out.pop(src), _nored)]
    if ram_gb and 'decode_peak_gb' in out:               # RAM verdict, NOT a count
        out['ram_fit'] = ['OVER' if g > 0.9 * ram_gb else ('tight' if g > 0.5 * ram_gb else 'ok')
                          for g in out['decode_peak_gb']]
    if sort_by not in (None, 'none'):                    # None/'none' = keep the input (build) order
        if sort_by not in out.columns:
            raise ValueError(f"f_feature_sizes: sort_by {sort_by!r} is not a column "
                             f"(have {list(out.columns)}); use None/'none' to keep the input order")
        out = out.sort_values(sort_by, ascending=False)
    if verbose:
        print(f'\n=== feature block sizes (decode peak ≈ {DECODE_PEAK_FACTOR}·T·n_dim·8B'
              + (f'; RAM {ram_gb:g} GB)' if ram_gb else ')')
              + (f'  [{_order_label(sort_by)}]'))
        print(out.to_string(index=False))
        if 'block_mb' in out:
            print(f'total resident if all blocks were held at once: '
                  f'{out["block_mb"].sum() / 1e3:.1f} GB (the stream holds ONE at a time)')
    return out


def f_render_clear():
    """Drop the cached monitor movie (frees the render's RAM; the next f_render_once re-renders)."""
    getattr(f_render_once, '_cache', {}).clear()


def f_render_once(session, build_cfg, cache=None, verbose=True, lazy=True, shape_dir=None):
    """Render the monitor movie ONCE and REUSE it across f_feature_search calls.

    f_feature_search renders once per CALL and shares it across that call's configs, but a script with
    several sweep cells (scorecard / layer2 sweep / layer1 sweep) calls it several times and so renders
    the same movie again each time. Pass the result as `prerendered=` to every call:

        res = fsearch.f_feature_search(..., prerendered=fsearch.f_render_once(session, build_cfg))

    The cache is held ON THIS FUNCTION (`f_render_once._cache`), created by the first call. Pass your
    own `cache` dict for an isolated one; `f_render_clear()` frees the default.

    SELF-CONTAINED ON PURPOSE — it references no module-level name that did not already exist. Under
    IPython/Spyder autoreload an existing function's BODY is updated in place, but a NEWLY added
    module-level name (a constant, a helper) is not always added to the live module, so a body that
    calls one dies with `NameError: name '…' is not defined` on a warm kernel. Keeping the cache and
    the skip-list inside the body means editing this function can never reintroduce that.

    KEYED on the render-relevant build_cfg plus the dataset, so editing a render knob and re-running
    re-renders instead of silently reusing the old movie; editing a conv/proc knob reuses it. Returns
    the (left, right, two) movie tuple.

    LAZY (`lazy=True` + `shape_dir`): a conv_autosave HIT only needs the movie's SHAPE (it is part of the
    cache key), not its pixels. When this dataset × render config was rendered before, its shapes are in
    <shape_dir>/render_shapes.json and placeholders are returned that know shape/dtype and render the real
    movie the FIRST time anything needs the pixels (a convolution not in the cache, or a pix block). Unknown
    shapes → render now (as before) and record them. Features are identical either way."""
    if cache is None:                       # default cache lives on the function; created on first call
        cache = getattr(f_render_once, '_cache', None)
        if cache is None:
            cache = f_render_once._cache = {}
    # build_cfg keys that do NOT change the RENDERED movie (conv/proc/feature-side knobs). Everything
    # else — side, obj_size, lat/vert_samp, max_frames, cam_params, the terrain block — DOES, so a
    # change there must force a re-render. A SKIP-list, not an allow-list, on purpose: a new render
    # knob then invalidates the cache by default instead of silently reusing a stale movie.
    skip = ('conv_backend', 'conv_device', 'conv_autosave', 'conv_autosave_tag',
            'conv_autosave_root', 'edge_min_frac', 'n_pix_pca', 'chunk_t', 'verbose',
            'agg_mode', 'obj_collapse', 'nan_absent', 'self_motion_decomp',
            'motion_smooth_sigma', 'self_motion_smooth_sigma',
            'pix_motion_smooth_sigma', 'lick_reward_smooth_sigma')
    key = (session.get('dset_name'),
           json.dumps({k: v for k, v in (build_cfg or {}).items() if k not in skip},
                      sort_keys=True, default=str))
    if cache.get('key') == key and cache.get('movie') is not None:
        if verbose:
            print('[render] reusing the cached movie (render knobs unchanged)', flush=True)
        return cache['movie']
    if lazy and cache.get('key') == key and cache.get('lazy') is not None:
        return cache['lazy']
    import hashlib
    import os
    from f_pipeline import f_render_session
    _kh = hashlib.md5(json.dumps(key, default=str).encode()).hexdigest()[:12]
    _shp = os.path.join(shape_dir, 'render_shapes.json') if shape_dir else None

    def _render():                                   # the real render, once per key
        if cache.get('key') == key and cache.get('movie') is not None:
            return cache['movie']
        if verbose and lazy:
            print('[render] pixels needed (convolution not cached, or pix) — rendering now', flush=True)
        _t = time.perf_counter()
        mv = f_render_session(session, build_cfg or {}, render_verbose=False, verbose=False)
        cache['movie'], cache['key'] = mv, key
        if _shp:                                     # record shapes so later sessions can defer
            try:
                rec = json.load(open(_shp)) if os.path.exists(_shp) else {}
            except Exception:
                rec = {}
            rec[_kh] = {'dset': str(key[0]), 'shapes': [list(map(int, m.shape)) for m in mv],
                        'dtypes': [str(m.dtype) for m in mv]}
            os.makedirs(shape_dir, exist_ok=True)
            with open(_shp + '.tmp', 'w') as f:
                json.dump(rec, f, indent=1)
            os.replace(_shp + '.tmp', _shp)
        if verbose:
            print(f'[render] monitor movie once … {time.perf_counter() - _t:.1f}s', flush=True)
        return mv

    _rec = None
    if lazy and _shp and os.path.exists(_shp):
        try:
            _rec = json.load(open(_shp)).get(_kh)
        except Exception:
            _rec = None
    if _rec is None:
        return _render()

    class _LazyMovie:                                # shape/dtype now, pixels on first use
        def __init__(self, i, shape, dtype):
            self._i, self.shape, self.dtype = i, tuple(int(x) for x in shape), np.dtype(dtype)
            self.ndim = len(self.shape)

        def _arr(self):
            return _render()[self._i]

        def __array__(self, dtype=None, copy=None):
            a = self._arr()
            return a if dtype is None else a.astype(dtype, copy=False)

        def __getitem__(self, k):
            return self._arr()[k]

        def __len__(self):
            return self.shape[0]

        def __getattr__(self, name):
            return getattr(self._arr(), name)

    ph = tuple(_LazyMovie(i, sh, dt) for i, (sh, dt) in enumerate(zip(_rec['shapes'], _rec['dtypes'])))
    cache['key'], cache['movie'], cache['lazy'] = key, None, ph
    if verbose:
        print('[render] deferred — movie shape known; renders only if a convolution is not cached '
              '(or pix)', flush=True)
    return ph


def f_proc_sweep(feat_sets, values, layer='layer2', names=None, verbose=True, op='pca', post=([],),
                 pre=([],)):
    """COMPONENT SWEEP: vary the per-cell (layer2) PCA over `values`, for the chosen feature sets.

    `op`: 'pca' (centered, default) | 'svd' (UNCENTERED truncated SVD — the mean is kept, so the first
    component points along the mean/energy direction). Run the same `values` under both to test whether
    centering costs anything on a non-negative block; the tags ('pca50' vs 'svd50') keep them distinct
    in one scorecard. NB `var_kept` is defined differently for the two — compare within an op.

    `layer` defaults to 'layer2' and should stay there — it is not a meaningful experimental choice.
    MEASURED: for a set with ONE filter, layer1 and layer2 act on the SAME matrix back-to-back, so the
    two sweeps return bit-identical blocks; for a COMBINATION, layer1 is applied per filter and yields
    k×n_filters dims, which is not a component sweep of the block at all. A per-filter reduction before
    combining is a property of the FEATURE — pin `layer1_proc` in the feat_sets entry — not of the
    sweep. 'layer1' is kept only so the machinery can express it.

    THE TRAP this closes: written by hand as `f_feature_configs(sets, layer2_proc=…, tag_suffix=…)`,
    precedence is entry > kwarg, so any set that PINS the swept key keeps its own value while the NAME
    still changes — N identically-built configs under N distinct names, reading as "components do
    nothing". A sweep means to vary that knob, so the pinned key is STRIPPED from the selected entries
    (each one reported) and the kwarg then applies to all of them.

    `values`: the ('pca', n) argument — int comps | float var-frac | (frac,cap) | None/0 = unreduced.
    `post`: list of step-lists appended AFTER the reduction, one sweep pass (= one line) each, e.g.
    [[], [('norm','zscore')], [('whiten', True)]]. Tagged into the name before the size tag
    ('…_whiten_l1svd16'). 'whiten' is dropped for an unreduced value (nothing to whiten).
    `pre`: list of step-lists inserted BEFORE the reduction, crossed with `post` (one line per pair),
    e.g. [[], [('norm','zscore')]] — zscore before = correlation-matrix PCA (every sample point equal
    weight). Tagged 'pre<steps>' ('…_prezscore_l1svd16'). 'whiten' is not allowed here (it needs
    components).
    `names`: which sets to sweep (None or 'all' = all). Returns the flat config list for f_feature_search."""
    if layer not in ('layer1', 'layer2'):
        raise ValueError(f"layer must be 'layer1' or 'layer2', got {layer!r}")
    key = f'{layer}_proc'
    if isinstance(names, str):
        names = [names]
    if names is not None and 'all' in names:              # 'all' = every set
        names = None
    if names is not None:
        # Resolve each name, tolerating the `_l1<op><n>` suffix a layer-1 override appends to the key,
        # so the same sweep_names work with the override on or off.
        _res, _unknown = [], []
        for n in names:
            if n in feat_sets:
                _res.append(n)
                continue
            cand = [k for k in feat_sets if k.startswith(n + '_l1')]
            if len(cand) == 1:
                _res.append(cand[0])
            elif len(cand) > 1:
                raise ValueError(f'f_proc_sweep: {n!r} matches several sets {cand} — name one exactly')
            else:
                _unknown.append(n)
        if _unknown:
            raise ValueError(f'f_proc_sweep: unknown set name(s) {_unknown}; have '
                             f'{[k for k in feat_sets if k != "_common"]}')
        names = _res
    sel = {k: dict(v) for k, v in feat_sets.items()
           if k == '_common' or names is None or k in names}
    pinned = [k for k, v in sel.items() if k != '_common' and key in v]
    for k in pinned:                                     # entry beats kwarg → the pin would win
        sel[k].pop(key)
    if verbose and pinned:
        print(f'f_proc_sweep: {key} was pinned in {pinned} — dropped so the sweep actually applies')
    if op not in ('pca', 'svd'):
        raise ValueError(f"f_proc_sweep: op must be 'pca' or 'svd', got {op!r}")
    tagp = 'l1' if layer == 'layer1' else 'l2'
    out = []
    for prs in pre:
        prs = [tuple(st) for st in (prs or [])]
        if any(st[0] in ('whiten', 'pca', 'svd') for st in prs):
            raise ValueError(f"f_proc_sweep: pre steps must be ('norm', …) only, got {prs}")
        rtag = f'pre{_post_tag(prs)}' if prs else ''
        for pst in post:
            pst = [tuple(st) for st in (pst or [])]
            ptag = '_'.join(t for t in (rtag, _post_tag(pst)) if t)
            for n in values:
                steps = prs + ([(op, n)] if n else []) + [st for st in pst if n or st[0] != 'whiten']
                out += f_feature_configs(sel, tag_suffix=f'{ptag + "_" if ptag else ""}{tagp}{f_pca_tag(n, op)}',
                                         **{key: steps})
    return out


def _post_tag(steps):
    """Short name for a post-reduction step-list: [('norm','zscore')] → 'zscore', [('whiten',True)] →
    'whiten', [('norm','total_var'),('whiten',True)] → 'total_var-whiten'; [] → ''."""
    out = []
    for op, arg in steps:
        out.append('whiten' if op == 'whiten' else str(arg) if op == 'norm' else f'{op}{arg}')
    return '-'.join(out)


def _marg_axes(marginalize):
    """Marginalize spec → concrete axis SET. Delegates to f_feature_source.f_marg_axes (the canonical
    one, shared with the plan-side _build_grid_block so the two can't drift)."""
    return f_marg_axes(marginalize)


def f_grid_shape(g):
    """(#cells, #conv-units) for a grid spec `g`. cells = # distinct NON-marginalized axis-value combos;
    conv-units = the extraction count. Handles an explicit `inputs` spec list (heterogeneous / pyramid) OR
    a homogeneous sf×tf bank (orientation batched in the conv, never a conv-unit axis)."""
    marg = _marg_axes(g['marginalize'])
    if g.get('inputs') is not None:                          # explicit level-1 spec list
        keys = set()
        for inp in g['inputs']:
            isg = inp.get('source') == 'gabor'
            sfv = float(np.atleast_1d(inp['sf_cyc_deg'])[0]) if inp.get('sf_cyc_deg') is not None else None
            tfv = float(np.atleast_1d(inp['tf_hz'])[0]) if inp.get('tf_hz') is not None else None
            no = len(inp['odegs']) if inp.get('odegs') else (inp.get('n_orient') or 1)
            for o in ([None] if 'ori' in marg else range(no)):
                k = []
                if 'ori' not in marg:
                    k.append(('o', o))
                if 'sf' not in marg:
                    k.append(('sf', sfv))
                if not isg and 'tf' not in marg:
                    k.append(('tf', tfv))
                keys.add(tuple(k))
        return len(keys), len(g['inputs'])
    is_gabor = (g['kind'] == 'gabor')
    dims = {'o': g['n_orient'], 'sf': len(g['sf_cyc_deg'])}
    if not is_gabor:
        dims['tf'] = len(g['tf_hz'])
    n_cells = 1
    for ax, n in dims.items():
        if ax not in marg:
            n_cells *= n
    return n_cells, len(g['sf_cyc_deg']) * (1 if is_gabor else len(g['tf_hz']))


def f_iter_grid_cells(session, build_cfg, cfg, prerendered, raw_cache, verbose=True, timing=False,
                      keep_raw=False):
    """STREAMING feature-grid build as a GENERATOR — yields (cell_name, block) the instant each cell
    FINALIZES (its LAST contributing conv-unit is done), freeing the conv raw per unit and dropping each
    finalized cell from the accumulator. Convolve ONE (sf,tf) at a time (orientation batched), apply
    `layer1_proc` per filter, route to cells by the NON-marginalized axes, apply `layer2_proc` per cell.
    Peak RAM ≈ one conv's raw + the currently-OPEN cells (conv-local marginalize → one conv-unit's cells;
    pooled sf/tf → until the last contributor). Lets the caller score+free each cell as it comes.

    `keep_raw=True` KEEPS each pooled raw in `raw_cache` instead of freeing it after its conv-unit. For
    a PROC SWEEP (many configs over the SAME bank, differing only in layer1/layer2) that is the whole
    cost: the default frees the raw, so every later config re-acquires it — a conv_autosave disk read +
    uint16 dequant per conv-unit, per config. Trades RAM for that: the pooled raw is n_col·T·4 B, e.g.
    the cyc_fov=(1,2,4,8,16) × 3-orientation magnitude pyramid is 5178 cols ≈ 2.0 GB at T=96k. Leave it
    off when configs span DIFFERENT banks (nothing to reuse, RAM grows with every one)."""
    import re
    from collections import OrderedDict
    g = cfg['grid']
    marg = _marg_axes(g['marginalize'])
    l1, l2 = g.get('layer1_proc') or [], g.get('layer2_proc') or []

    # UNITS = extraction inputs. Explicit level-1 spec list (heterogeneous / pyramid) OR the sf×tf product
    # of a homogeneous bank (f_feature_grid). Each unit = (label, extraction spec, sfv, tfv, is_gabor).
    units = []
    if g.get('inputs') is not None:
        for s in g['inputs']:
            s = dict(s)
            isg = (s.get('source') in NO_TF_KINDS)          # spatial-only kind → no tf in the label/key
            sfv = float(np.atleast_1d(s['sf_cyc_deg'])[0]) if s.get('sf_cyc_deg') is not None else None
            tfv = float(np.atleast_1d(s['tf_hz'])[0]) if s.get('tf_hz') is not None else None
            units.append((f"{s.get('source')}_sf{f_num_tag(sfv)}"
                          + ('' if isg else f'tf{f_num_tag(tfv)}'), s, sfv, tfv, isg))
    else:
        kind = g['kind']; is_gabor = (kind in NO_TF_KINDS)   # 'gabor'/'fft' → spatial only, no tf axis
        for sfv in g['sf_cyc_deg']:
            for tfv in ([None] if is_gabor else list(g['tf_hz'])):
                s = {'source': kind, 'reduce': g['reduce'], 'ds': g['ds'], 'n_orient': g['n_orient'],
                     'odegs': g.get('odegs'), 'sf_cyc_deg': (sfv,)}
                if g.get('ds_factor') is not None:           # per-SF Nyquist stride (ds='nyquist')
                    s['ds_factor'] = g['ds_factor']
                if is_gabor:
                    s['stats'] = tuple(g['stats'])
                    for _k in ('n_sf_sub', 'n_ori_sub', 'patch_snap'):   # fft-only extras
                        if g.get(_k) is not None:
                            s[_k] = g[_k]
                else:
                    s['tf_hz'], s['output'] = (tfv,), g['output']
                    if g.get('transform') is not None:
                        s['transform'] = g['transform']
                units.append((f'{kind}_sf{f_num_tag(sfv)}'
                              + ('' if is_gabor else f'tf{f_num_tag(tfv)}'), s, sfv, tfv, is_gabor))

    def _has_later_contributor(key, u):                    # any unit AFTER u feeding this cell key?
        kd = dict(key)
        fsf, ftf = kd.get('sf'), kd.get('tf')
        for (_l, _s, sfv2, tfv2, _g) in units[u + 1:]:
            if (fsf is None or sfv2 == fsf) and (ftf is None or tfv2 == ftf):
                return True
        return False

    def _finalize(key, members):                           # concat cell members → layer2 → standard block
        cell_name = '_'.join(                              # format numeric sf/tf; 'o' is already a string
            f'{ax}{f_num_tag(val) if isinstance(val, (int, float)) else val}' for ax, val in key) or 'all'
        Xcat = np.nan_to_num(np.concatenate([X for _, X, _, _, _ in members], axis=1))
        raw_names = [f'{lbl}:{nm}' for lbl, _, nms, _, _ in members for nm in nms]
        Xp, fit = f_proc(Xcat, l2)
        _pca = next((p['pca'] for op, p in fit if op in ('pca', 'svd') and p.get('pca') is not None), None)
        names = ([f'{cell_name}_PC{i + 1}' for i in range(Xp.shape[1])] if _pca is not None
                 else list(raw_names))
        # LAYER-1 variance kept, pooled over this cell's filters: variance-WEIGHTED by each filter's
        # input width, so it reads as "fraction of the cell's raw variance surviving layer1".
        l1v = [(w_in, w_in * v) for _, _, _, w_in, v in members if v == v]      # skip NaN (no l1 pca)
        return cell_name, {'X': np.asarray(Xp, np.float32), 'names': names, 'pca': _pca,
                           'pca_groups': None, 'raw_names': raw_names, 'raw_dim': Xcat.shape[1],
                           'source_blocks': [lbl for lbl, _, _, _, _ in members], 'proc_fit': fit,
                           'l1_var_kept': (sum(v for _, v in l1v) / sum(w for w, _ in l1v)
                                           if l1v else np.nan),
                           'l1_dim_in': sum(w_in for _, _, _, w_in, _ in members),
                           'kind': cell_name}

    cells = OrderedDict()
    for u, (label, spec, sfv, tfv, is_gabor) in enumerate(units):
        if timing:
            print(f'      [{time.strftime("%H:%M:%S")}] → conv {label} [{u + 1}/{len(units)}]', flush=True)
        _tconv = time.perf_counter()
        raw, _ft = _raw_block(session, build_cfg, label, [spec], prerendered, raw_cache, False,
                              timing=timing)
        if timing:
            print(f'      [{time.strftime("%H:%M:%S")}] ✓ conv {label} '
                  f'{time.perf_counter() - _tconv:.1f}s (raw {np.asarray(raw["X"]).shape[1]} col)',
                  flush=True)
        rn = raw.get('raw_names') or raw['names']
        Xraw = np.asarray(raw['X'])
        _od, _no = spec.get('odegs'), (spec.get('n_orient') or 1)
        for glabel, info in f_pca_group_cols(rn, 'none', None).items():   # one group per filter (o,sf,tf)
            cols = info['cols']
            # nested path: norm-only pre steps, then a pca/svd fit ONCE per (filter, pre) and sliced
            _ir = next((i for i, st in enumerate(l1) if st[0] in ('pca', 'svd')), None)
            _lead = l1[_ir] if _ir is not None else None
            if (_lead and _lead[1] and len(cols) >= 2
                    and all(st[0] == 'norm' for st in l1[:_ir])):
                _pre = l1[:_ir]
                _Xin = f_proc(Xraw[:, cols], _pre)[0] if _pre else Xraw[:, cols]
                _dk = ('l1fit', _extraction_sig(label, [spec]), glabel, _lead[0], repr(_pre))
                if _dk not in raw_cache:                   # ONE decomposition per filter per call
                    raw_cache[_dk] = _l1_decomp(_Xin, _lead[0])
                Xp, _p = _l1_nested(_Xin, _lead[0], _lead[1], raw_cache[_dk])
                _f1 = [(_lead[0], {'pca': _p, 'eig': None if _p is None else _p.explained_variance_})]
                for _st in l1[_ir + 1:]:                   # trailing steps
                    if _st[0] == 'whiten':                 # ÷ √variance of each kept component
                        if _p is not None:
                            _ev = _p.explained_variance_
                            Xp = Xp / np.sqrt(np.where(_ev > 0, _ev, 1.0)).astype(np.float32)
                        _f1.append(('whiten', {'eig': None if _p is None else _p.explained_variance_}))
                    else:
                        Xp, _rest = f_proc(Xp, [_st])
                        _f1 += _rest
            else:
                Xp, _f1 = f_proc(Xraw[:, cols], l1)        # LAYER-1 proc per individual filter
            _p1 = next((p['pca'] for op, p in _f1 if op in ('pca', 'svd') and p.get('pca') is not None), None)
            _v1 = (float(np.sum(_p1.explained_variance_ratio_)) if _p1 is not None
                   else (1.0 if l1 else np.nan))           # no pca fired → nothing dropped
            mo = re.search(r'_o(\d+)', glabel)               # orientation INDEX in the raw name (build path)
            key = []
            if 'ori' not in marg:                            # name the cell by DEGREES
                io = int(mo.group(1)) if mo else 0
                odeg = _od[io] if _od else io * 180.0 / _no
                key.append(('o', f'{odeg:g}'))
            if 'sf' not in marg:                             # RAW float in the key (contributor match);
                key.append(('sf', sfv))                      # formatted at the _finalize name join
            if not is_gabor and 'tf' not in marg:
                key.append(('tf', tfv))
            cells.setdefault(tuple(key), []).append(
                (f'{label}:{glabel}', Xp, _layer1_names(Xp, [rn[c] for c in cols], glabel), len(cols), _v1))
        if not keep_raw:                                       # FREE the conv raw NOW (default)
            raw_cache.pop(_extraction_sig(label, [spec]), None)
        del raw, Xraw
        for key in [k for k in cells if not _has_later_contributor(k, u)]:   # finalize + yield complete cells
            if timing:
                _kn = '_'.join(f'{a}{f_num_tag(v) if isinstance(v, (int, float)) else v}'
                               for a, v in key) or 'all'
                print(f'      [{time.strftime("%H:%M:%S")}] → finalize/PCA cell {_kn}', flush=True)
            _tfin = time.perf_counter()
            out = _finalize(key, cells.pop(key))
            if timing:
                print(f'      [{time.strftime("%H:%M:%S")}] ✓ finalize/PCA {out[0]} '
                      f'{time.perf_counter() - _tfin:.1f}s ({np.asarray(out[1]["X"]).shape[1]} dim)',
                      flush=True)
            yield out


def f_build_grid(session, build_cfg, cfg, prerendered, raw_cache, verbose=True, keep_raw=False):
    """Materialize f_iter_grid_cells into a {cell_name: block} dict (non-interleaved use / tests)."""
    return dict(f_iter_grid_cells(session, build_cfg, cfg, prerendered, raw_cache, verbose,
                                  keep_raw=keep_raw))


def f_score_grid(res, metric='decode_r2'):
    """Parse a feature-grid result (f_feature_grid) into a tidy long DataFrame with `ori` (DEGREES),
    `sf`, `tf` columns + the chosen `metric` — ready to pivot into the grid, e.g.
    `f_score_grid(res).pivot_table(metric, 'ori', ['sf','tf'])`. Reads the o/sf/tf VALUES straight off
    each cell's block name (e.g. 'o45_sf0.13_tf2.0'); an axis absent from the name (marginalized) → NaN."""
    import re
    import pandas as pd
    rows = []
    for _, r in res[res['status'] == 'ok'].iterrows():
        nm = str(r['block'])
        mo = re.search(r'o([\d.]+)', nm)                      # orientation in DEGREES
        msf, mtf = re.search(r'sf([\d.]+)', nm), re.search(r'tf([\d.]+)', nm)
        rows.append({'ori': float(mo.group(1)) if mo else np.nan,
                     'sf': float(msf.group(1)) if msf else np.nan,
                     'tf': float(mtf.group(1)) if mtf else np.nan,
                     metric: r.get(metric, np.nan), 'block': nm})
    return pd.DataFrame(rows)


def f_prep_configs(base_configs, preps=(None, 'whiten')):
    """Expand each base config across feature-PREPROCESSING modes → test normalize/whiten before
    scoring. `preps` ⊆ {None (as-is), 'zscore' (unit-variance columns), 'whiten' (PCs ÷ √eigenvalue via
    the block's fitted PCA)}. Each expanded config sets `feat_prep` + a '|<prep>' name suffix. NOTE: the
    search scores with proc_before_cv=True (no within-fold standardization), so feat_prep moves ENCODE as
    well as DECODE — raw/combined weights top PCs, whiten/zscore equal-weights all PCs."""
    out = []
    for c in base_configs:
        for p in preps:
            if p not in _FEAT_PREPS:
                raise ValueError(f"prep {p!r} not in {_FEAT_PREPS}")
            out.append({**c, 'feat_prep': p, 'name': f"{c.get('name', 'cfg')}|{p or 'raw'}"})
    return out


def f_make_neural_target(est1, sigma_frames=6, peak_norm=True):
    """Neural rate proxy (T, n_active) matching VR_ca_cebra's Y_neu: drop silent cells, Gaussian-
    smooth, optional per-cell peak-normalize. Cell ORDER is irrelevant for scoring (cells are treated
    symmetrically), so the hclust sort is skipped. Dead/constant cells are dropped again in the scorer.
    Returns (Y_neu (T, n_active) float32, active_cells bool over est1['S'] rows)."""
    from f_utils import f_neural_rate
    R, active = f_neural_rate(est1['S'], sigma_frames=sigma_frames, peak_norm=peak_norm)
    return R.T.astype(np.float32), active


_GRAN_NPCA = {'combined': (lambda k: k), 'separate': (lambda k: ('filter', k)),
              'raw': (lambda k: None)}   # granularity → block n_pca spec (see f_apply_block_pca)


def f_granularity_configs(block, inputs, granularities=('combined', 'separate', 'raw'), n_pca=300):
    """FRESH-BUILD configs comparing PCA granularity for ONE block + extraction (the separate / combined
    / raw question, generated on GPU — NO cache). Each config forces origin='build' with `inputs` (the
    extraction spec list, e.g. [{'source':'gabor','reduce':'sample','ds':4,'sf_cyc_deg':(0.13,),
    'odegs':None}]); the render happens once in f_feature_search and only the conv re-runs. The block's
    n_pca per granularity: 'combined' → n_pca (whole-block PCA); 'separate' → ('filter', n_pca)
    (per-filter grouped PCA); 'raw' → None (no PCA). One config per granularity, named '<block>:<gran>'."""
    out = []
    for g in granularities:
        if g not in _GRAN_NPCA:
            raise ValueError(f"granularity {g!r} not in {list(_GRAN_NPCA)}")
        out.append({'name': f'{block}:{g}', 'blocks': [block],
                    'plan': {block: {'origin': 'build', 'n_pca': _GRAN_NPCA[g](n_pca),
                                     'inputs': list(inputs)}}})
    return out


def f_variant_configs(blocks, variants=('combined', 'separate'), base_tag='gen_7_16_26'):
    """CACHE-load variant of f_granularity_configs (use only with f_feature_search(force_build=False)):
    score each block loaded from each pca_feat cache variant with NO reconvolution. `variants` ⊆
    {'combined','separate','raw'} → cache_tag base_tag+{_pcacomb,_pcasep,_pcaraw}. 'raw' needs a
    _pcaraw cache (VR_feature_gen SAVE_RAW=True). Prefer f_granularity_configs for a real search."""
    out = []
    for b in blocks:
        for v in variants:
            if v not in _VAR_SUFFIX:
                raise ValueError(f"variant {v!r} not in {list(_VAR_SUFFIX)}")
            out.append({'name': f'{b}:{v}', 'blocks': [b],
                        'source': {'cache_tag': base_tag + _VAR_SUFFIX[v]}})
    return out


def _encode_cfg_cols(built, name, Y, frame_t, enc_cfg):
    """Encode block `name` with the SHARED encoder config (f_encoding.f_run_encoder) → the encode_*
    columns of a scorecard row (same names as f_score_representations, so plots/tables are unchanged)
    + population scores + enc_tag. Double-reduction rule: feature_block=[name], n_feature_pcs=None —
    the feature set's own layer1/layer2 processing alone sets the dims."""
    from f_encoding import f_run_encoder, f_enc_config_tag
    from f_dataset_scan import _dist_stats
    if enc_cfg.get('covariate_blocks'):
        raise ValueError('enc_cfg covariate_blocks are not supported in the feature search yet')
    cfg = {**enc_cfg, 'feature_block': [name], 'n_feature_pcs': None}
    X = np.asarray(built[name]['X'])
    T = min(X.shape[0], np.asarray(Y).shape[0], np.asarray(frame_t).size)
    out = f_run_encoder(cfg, {name: {**built[name], 'X': X[:T]}}, np.asarray(Y)[:T],
                        np.asarray(frame_t).ravel()[:T])
    r = np.asarray(out['res']['r_per_cell'], float)
    r2 = np.asarray(out['res']['r2_per_cell'], float)
    cols = {'encode_r_med': float(np.nanmedian(r)), 'encode_r_mean': float(np.nanmean(r)),
            'encode_r2_med': float(np.nanmedian(r2)), 'encode_r2_mean': float(np.nanmean(r2)),
            'encode_frac_sig': float(np.nanmean(r > cfg['r_sig'])),
            **{f'encode_r_{k}': v for k, v in _dist_stats(r).items()},
            **{f'encode_r2_{k}': v for k, v in _dist_stats(r2).items()},
            **{f'encode_{k}': out['scores'][k] for k in ('r2', 'r2_vw', 'rv', 'procrustes', 'rsa')
               if k in out['scores']},
            'enc_tag': f_enc_config_tag(cfg)}
    return cols


def f_feature_search(session, configs, Y_neu, base_source, build_cfg=None, base_plan=None, *,
                     n_pcs=30, lags_ms=(0, 100, 200), embargo_sec=0.5, n_folds=5, r_sig=0.1,
                     force_build=True, prerendered=None, feat_prep=None,
                     feat_pre_pca_zscore=False, scorer_standardize=True, neural_zscore=True,
                     feature_proc=None, neural_proc=None, encode_neural_proc=None, verbose=True,
                     timing=False, keep_raw=False, scores=('decode', 'encode'), targets=None,
                     enc_cfg=None):
    """Score a list of candidate feature `configs` on ONE dataset by encode + decode → tidy table.

    session      : f_load_session output (ONE dataset; provides est1/bh/frame_t/cam_params).
    configs      : list of config dicts (see module docstring).
    Y_neu        : (T, n_active) neural rate proxy (f_make_neural_target). Dead cells dropped in scorer.
    base_source  : f_default_source(...) — features_dir / mouse_id / cache_tag / conv_tag; a config's
                   'source' overrides these (only used when force_build=False).
    build_cfg    : f_default_build_cfg(...) shared render/side config (conv_backend='gpu' recommended).
    base_plan    : dict of plan overrides applied to EVERY config before its own 'plan' (default {}).
    force_build  : True (default) → REGENERATE every feature fresh on GPU (each block forced to
                   origin='build'), never touching the cache — the render runs ONCE (shared across
                   configs), only the conv re-runs per config. False → honor each config's origin
                   ('auto'/'cache') and load from the caches (the f_variant_configs path).
    prerendered  : (left, right, two) movie to reuse; None + force_build → rendered once here. Pass
                   `f_render_once(session, build_cfg, cache)` to share ONE render across SEVERAL calls
                   (the scorecard cell + each sweep cell), else each call re-renders the same movie.
    keep_raw     : keep each pooled conv raw in the in-run cache instead of freeing it per conv-unit.
                   For a PROC SWEEP over ONE bank this is the difference between convolving/reading it
                   once and re-reading it from the conv_autosave store for every config; costs
                   n_col·T·4 B of RAM (see f_iter_grid_cells). Leave off across different banks.
    Feature-normalization chain (each stage an explicit knob; applied per config in order):
      feat_pre_pca_zscore : z-score RAW feature columns BEFORE the (combined/separate) PCA. Memoized
                            (single-block) path only; derived combos ignore it (set zscore_joint in the
                            config plan instead).
      feat_prep           : AFTER PCA — None (as-is) | 'zscore' (unit-var cols) | 'whiten' (PCs ÷
                            √eigenvalue). A config's own 'feat_prep' overrides this global default.
      scorer_standardize  : the scorer's within-fold z-score of the PREDICTOR (encode: feature cols;
                            decode: neural PCs). Set False to test feat_prep WITHOUT the scorer
                            re-normalizing (else encode features are effectively whitened regardless).
      neural_zscore       : per-cell z-score the neural → sets which directions the decode neural-PCA
                            picks (amplitude-dominated off / cell-balanced on). Decode-only (encode
                            uses the non-z-scored per-cell target; its r/R² are scale-invariant).
    (Peak-normalize the neural target is a knob on f_make_neural_target — `peak_norm` — in the script.)
    n_pcs/lags_ms/embargo_sec/n_folds/r_sig : passed to f_score_representations.
    scores  : which fits to run, ('decode', 'encode'); a skipped one leaves its columns NaN.
    enc_cfg : encoder config (f_encoding.f_default_encoder_params) → ENCODE with f_run_encoder under
              that config (the model shared with VR_ca_cebra; feature_block = the scored block,
              n_feature_pcs=None). Fills the same encode_* columns + encode_r2_vw/rv/procrustes/rsa +
              enc_tag. None → the older f_score_representations encode (lags_ms/r_sig/… above).
    targets : {name: (T, n) array} neural targets, each block scored against every one (built once);
              rows carry a `target` column. None → {'cells': Y_neu}. E.g. cells + ensemble traces
              from f_ensemble_pipeline.f_ens_targets.
    A grid cell whose block is byte-identical to one already scored in this call (e.g. a sweep value
    past a set's full width) reuses that score instead of refitting.
    (Encode CPU is bounded by the OMP/MKL BLAS thread env vars atop the script — f_encode_per_neuron
     is serial.)
    verbose : the standard per-config / per-cell summary lines.
    timing  : EXTENDED log (additive) — flushed timestamped START + duration markers for each major
              step (render, per conv-unit, decode, encode) so progress is readable live while
              monitoring CPU. Default False = compact (verbose-only) output.

    Returns a DataFrame: config / block / decode_r2 / decode_validfrac / encode_r_med /
    encode_frac_sig / n_dim / status (ok | FAIL:<Err>), sorted most-decodable first.
    """
    fs = f_imaging_fs(session['frame_t'])
    embargo = int(round(embargo_sec * fs))
    base_plan = base_plan or {}
    if build_cfg and build_cfg.get('conv_autosave'):     # thread the store root (from base_source) into
        build_cfg = {**build_cfg,                        # build_cfg so _raw_block can locate the store
                     'conv_autosave_root': base_source.get('features_dir')}
    _needs_render = force_build or any(c.get('grid') for c in configs)
    if _needs_render and prerendered is None:            # render the movie ONCE — identical across configs
        from f_pipeline import f_render_session
        if timing:
            print(f'[{time.strftime("%H:%M:%S")}] → render monitor movie once', flush=True)
        _tr = time.perf_counter()
        prerendered = f_render_session(session, build_cfg or {}, render_verbose=False, verbose=False)
        if verbose:
            print(f'[render] monitor movie once … {time.perf_counter() - _tr:.1f}s', flush=True)
    frames = []
    raw_cache = {}                                       # memoize RAW extractions across configs (conv once)
    score_memo = {}                                      # (block hash, target) → scored row
    if targets is None:
        targets = {'cells': Y_neu}
    targets = {k: np.asarray(v[0] if isinstance(v, tuple) else v) for k, v in targets.items()}
    _multi = len(targets) > 1
    _enc_new = enc_cfg is not None and 'encode' in scores        # encode via the shared config
    _sc_scores = tuple(x for x in scores if x != 'encode') if _enc_new else scores
    _ft_all = np.asarray(session['frame_t']).ravel()
    for i, cfg in enumerate(configs):
        _tc = time.perf_counter()
        name = cfg.get('name', f'cfg{i}')
        is_grid = bool(cfg.get('grid'))                  # feature-grid instruction (f_feature_grid)
        blocks = list(cfg.get('blocks') or [name])
        prep = cfg.get('feat_prep', feat_prep)
        if is_grid:
            plan, src, memoizable = None, {}, False
        else:
            plan = f_default_feature_plan(**{**base_plan, **(cfg.get('plan') or {})})
            if force_build:                              # force fresh GPU build (no cache) for every block
                for n in plan:
                    if plan[n].get('origin') != 'derived':   # derived reuses already-built blocks
                        plan[n] = {**plan[n], 'origin': 'build'}
            src = {**base_source, **(cfg.get('source') or {})}
            # Memoized path: every scored block is a single-input FRESH build → build each raw extraction
            # ONCE (cached across configs), then apply this config's PCA granularity + prep. Derived
            # combos / cache loads use the full resolve.
            memoizable = force_build and all(
                plan[b].get('origin') != 'derived' and len(plan[b].get('inputs') or []) == 1
                and not plan[b].get('zscore_joint') for b in blocks)
        if verbose:                                      # clean per-config header + bank/blocks descriptor
            if is_grid:
                g = cfg['grid']
                if g.get('inputs') is not None:          # explicit level-1 spec list
                    desc = f"{g['kind']} · {len(g['inputs'])} specs · marg={'+'.join(g['marginalize']) or 'none'}"
                else:
                    _tf = '' if g['kind'] == 'gabor' else f" × {len(g['tf_hz'])} tf"
                    desc = (f"{g['kind']} · {g['n_orient']} ori × {len(g['sf_cyc_deg'])} sf{_tf} · "
                            f"marg={'+'.join(g['marginalize']) or 'none'}")
            else:
                desc = f'blocks={blocks}'
            print(f'[{i + 1}/{len(configs)}] {name}   {desc}')
        try:
            if is_grid:                                  # INTERLEAVE: build cell → score → FREE (bounded RAM)
                M, U = f_grid_shape(cfg['grid'])
                if verbose:
                    print(f'      {M} cells from {U} conv-units (interleaved build->score->free)')
                cell_rows, j = [], 0
                for cell_name, block in f_iter_grid_cells(session, build_cfg, cfg, prerendered,
                                                          raw_cache, verbose, timing=timing,
                                                          keep_raw=keep_raw):
                    j += 1
                    _tcell = time.perf_counter()
                    if prep:
                        block = {**block, 'X': _apply_feat_prep(block['X'], prep)}
                    _Xb = np.ascontiguousarray(block['X'])
                    _hb = (_Xb.shape, str(_Xb.dtype), hashlib.blake2b(_Xb, digest_size=16).hexdigest())
                    for tname, Yt in targets.items():    # one build, scored per target
                        _tt = time.perf_counter()
                        _tl = f' [{tname}]' if _multi else ''
                        _hk = (_hb, tname)
                        if _hk in score_memo:            # identical block already scored → reuse
                            sc1 = score_memo[_hk].copy()
                            sc1['method'] = cell_name
                            cell_rows.append(sc1)
                            if verbose:
                                print(f'      [{j:>2}/{M}] {cell_name:<26}{_tl} identical block already '
                                      f'scored → reused')
                            continue
                        T = min(int(_Xb.shape[0]), Yt.shape[0])
                        sc1 = f_score_representations({cell_name: block}, Yt[:T], [cell_name],
                                                      n_pcs=n_pcs, lags_ms=lags_ms, embargo=embargo,
                                                      n_folds=n_folds, r_sig=r_sig, fs=fs,
                                                      standardize=scorer_standardize,
                                                      neural_zscore=neural_zscore,
                                                      feature_proc=feature_proc, neural_proc=neural_proc,
                                                      encode_neural_proc=encode_neural_proc,
                                                      proc_before_cv=True, scores=_sc_scores, verbose=False,
                                                      timing=timing)
                        if _enc_new:
                            for _k, _v in _encode_cfg_cols({cell_name: block}, cell_name, Yt[:T],
                                                           _ft_all, enc_cfg).items():
                                sc1[_k] = _v
                        sc1['target'] = tname
                        score_memo[_hk] = sc1
                        cell_rows.append(sc1)
                        if verbose:
                            _r = sc1.iloc[0]
                            print(f'      [{j:>2}/{M}] {cell_name:<26}{_tl} dec={_r["decode_r2"]:+.3f}  '
                                  f'enc={_r["encode_r_med"]:+.3f}   {time.perf_counter() - _tt:.1f}s')
                    del block, _Xb                       # release the scored cell immediately
                sc = pd.concat(cell_rows, ignore_index=True)
            else:                                        # non-grid: build all blocks then score at once
                if memoizable:
                    built, frame_t = {}, None
                    for b in blocks:
                        # 'build' = extract raw (gabor/meng CONVOLVE here; pix just selects pixels), then PCA
                        if timing:
                            print(f'      [{time.strftime("%H:%M:%S")}] → build+pca {b}', flush=True)
                        _tb = time.perf_counter()
                        raw, frame_t = _raw_block(session, build_cfg, b, plan[b]['inputs'],
                                                  prerendered, raw_cache, False, timing=timing)
                        if timing:
                            print(f'      [{time.strftime("%H:%M:%S")}]   raw {b} '
                                  f'{time.perf_counter() - _tb:.1f}s ({np.asarray(raw["X"]).shape[1]} col)'
                                  f' → pca', flush=True)
                        _tp = time.perf_counter()
                        built[b] = _reduce_raw(raw, b, plan[b].get('n_pca'), prep,
                                               pre_pca_zscore=feat_pre_pca_zscore)
                        if timing:
                            print(f'      [{time.strftime("%H:%M:%S")}] ✓ build+pca {b} '
                                  f'(pca {time.perf_counter() - _tp:.1f}s, total '
                                  f'{time.perf_counter() - _tb:.1f}s)', flush=True)
                else:
                    built, frame_t, _st, _mv = f_resolve_features(
                        session, plan, build_cfg, only=blocks, want_movie=False, verbose=False,
                        whiten=(prep == 'whiten'), prerendered=(prerendered if force_build else None), **src)
                    for b in blocks:                                  # 'zscore' post-process (whiten done above)
                        built[b] = {**built[b], 'X': _apply_feat_prep(built[b]['X'], prep)}
                _proc = cfg.get('proc')                               # f_proc step-list (norm/pca/whiten) on the block
                if _proc:
                    for b in blocks:
                        _raw_d = int(np.asarray(built[b]['X']).shape[1])   # width BEFORE the block proc
                        Xp, _fit = f_proc(np.asarray(built[b]['X']), _proc)
                        _pca = next((pp['pca'] for op, pp in _fit if op in ('pca', 'svd') and pp.get('pca') is not None), None)
                        nm = ([f'{b}_PC{i + 1}' for i in range(Xp.shape[1])] if _pca is not None
                              else built[b]['names'])
                        # keep the fitted pca + pre-proc width so the size/variance report covers pix too
                        built[b] = {**built[b], 'X': np.asarray(Xp, np.float32), 'names': nm,
                                    'pca': _pca, 'proc_fit': _fit,
                                    'raw_dim': built[b].get('raw_dim') or _raw_d}
                _scs = []
                for tname, Yt in targets.items():
                    T = min(int(np.asarray(frame_t).size), Yt.shape[0])   # align neural ↔ block clock
                    _s1 = f_score_representations(built, Yt[:T], blocks, n_pcs=n_pcs, lags_ms=lags_ms,
                                                  embargo=embargo, n_folds=n_folds, r_sig=r_sig, fs=fs,
                                                  standardize=scorer_standardize,
                                                  neural_zscore=neural_zscore,
                                                  feature_proc=feature_proc, neural_proc=neural_proc,
                                                  encode_neural_proc=encode_neural_proc,
                                                  proc_before_cv=True, scores=_sc_scores, verbose=verbose,
                                                  timing=timing)
                    if _enc_new:
                        for _b in blocks:
                            for _k, _v in _encode_cfg_cols(built, _b, Yt[:T], _ft_all, enc_cfg).items():
                                _s1.loc[_s1['method'] == _b, _k] = _v
                    _s1['target'] = tname
                    _scs.append(_s1)
                sc = pd.concat(_scs, ignore_index=True)
            sc.insert(0, 'config', name)
            sc.insert(1, 'feat_prep', str(prep))
            sc = sc.rename(columns={'method': 'block'})
            sc['status'] = 'ok'
            if verbose:
                print(f'      done in {time.perf_counter() - _tc:.1f}s')
        except Exception as e:                        # one bad config never kills the sweep
            if verbose:
                import traceback
                traceback.print_exc()                 # full traceback → the real throw site
                print(f'    FAIL {name}: {type(e).__name__}: {e}')
            _err = f'{type(e).__name__}: {e}'
            sc = pd.DataFrame([{'config': name, 'feat_prep': str(prep), 'block': b, 'target': t,
                                **f_nan_metrics(), 'err': _err, 'status': f'FAIL:{type(e).__name__}'}
                               for b in blocks for t in targets])
        frames.append(sc)
    df = pd.concat(frames, ignore_index=True)
    _nb = df.groupby(['config', 'target'])['block'].transform('size')
    df['label'] = np.where(_nb > 1, df['config'] + ':' + df['block'], df['config'])
    if _multi:                                           # several targets → name the target too
        df['label'] = df['label'] + ' [' + df['target'].astype(str) + ']'
    return df.reset_index(drop=True)                     # build order; sort at plot time


# Selectable scorecard metrics → (column, panel title, dist-prefix). DECODE = neural→feature;
# ENCODE = feature→neural. dist-prefix = the column stem whose {_min,_p5,_p25,_p75,_p95,_max} give the
# per-unit distribution (feature cols for decode, cells for encode) → percentile whiskers; None = scalar.
DECODE_METRICS = {
    'r2':         ('decode_r2',         'decode R² (var-weighted)', 'decode_r2'),  # big-var cols dominate
    'r2_uniform': ('decode_r2_uniform', 'decode R² (uniform-avg)',  'decode_r2'),  # cols count equally
    'r2_med':     ('decode_r2_med',     'decode R² (median col)',   'decode_r2'),  # robust to bad cols
    'validfrac':  ('decode_validfrac',  'decode valid-frac',        None),         # frac usable time rows
}
ENCODE_METRICS = {
    'r_med':    ('encode_r_med',    'encode r (median cell)',    'encode_r'),   # median per-neuron OOF r
    'r_mean':   ('encode_r_mean',   'encode r (mean cell)',      'encode_r'),   # mean per-neuron OOF r
    'r2_med':   ('encode_r2_med',   'encode R² (median cell)',   'encode_r2'),  # median per-neuron OOF R²
    'r2_mean':  ('encode_r2_mean',  'encode R² (mean cell)',     'encode_r2'),  # mean per-neuron OOF R²
    'frac_sig': ('encode_frac_sig', 'encode frac cells r>r_sig', None),         # fraction of encoded cells
    # population (neural-PCA) scores — need encode_neural_proc set in f_feature_search / f_score_representations
    'pop_r2':   ('encode_pop_r2',         'encode R² (pop, var-wt PCs)', 'encode_pop'),  # var-wt over PCs
    'pop_r2u':  ('encode_pop_r2_uniform', 'encode R² (pop, mean PCs)',   'encode_pop'),  # mean over PCs
    'pop_r':    ('encode_pop_r',          'encode r (pop, var-wt PCs)',  None),          # var-wt corr / PC
    # population scores from the shared encoder (enc_cfg → f_run_encoder)
    'r2_vw':      ('encode_r2_vw',      'encode R² (pop, var-wt)',  None),
    'rv':         ('encode_rv',         'encode RV coefficient',    None),
    'procrustes': ('encode_procrustes', 'encode procrustes',        None),
    'rsa':        ('encode_rsa',        'encode RSA',               None),
}


def _resolve_metrics(decode_metric, encode_metric):
    if decode_metric not in DECODE_METRICS:
        raise ValueError(f"decode_metric {decode_metric!r} not in {list(DECODE_METRICS)}")
    if encode_metric not in ENCODE_METRICS:
        raise ValueError(f"encode_metric {encode_metric!r} not in {list(ENCODE_METRICS)}")
    return DECODE_METRICS[decode_metric], ENCODE_METRICS[encode_metric]


def _scorecard_rows(df, sort_by, dcol, ecol):
    """Rows to plot. sort_by: None/'none' = INPUT order (the order the configs were listed in — `res`
    comes back in build order) | 'decode'/'encode' = that panel's metric (descending) | any column
    name. First row ends up at TOP (axes inverted by the caller)."""
    d = df[df['status'] == 'ok']
    if sort_by not in (None, 'none'):
        col = {'decode': dcol, 'encode': ecol}.get(sort_by, sort_by)
        d = d.sort_values(col, ascending=False, na_position='last')
    d = d.reset_index(drop=True)
    return d, np.arange(len(d))


def _order_label(sort_by):
    """Row-order caption. None/'none' = the order the configs were listed in (res is in build order)."""
    return 'input order' if sort_by in (None, 'none') else f'sorted by {sort_by}'


def _scorecard_ylabels(d, show_n_dim=True):
    """Y-tick labels for the scorecard panels. `show_n_dim` appends the width actually scored, so a
    bar can be read against what it cost (a pca-50 cell winning over a 3780-col raw one is the point
    of the sweep). Falls back to the bare label when n_dim is absent or NaN."""
    if not show_n_dim or 'n_dim' not in d:
        return list(d['label']), 'config'
    return ([f'{lb}  [{int(n)}d]' if n == n else str(lb)
             for lb, n in zip(d['label'], d['n_dim'])], 'config  [n_dim]')


def f_plot_feature_search(df, sort_by='none', title='', decode_metric='r2', encode_metric='r_med',
                          show_n_dim=True):
    """Two horizontal-bar panels (clean bars, no annotations) — one bar per config. `sort_by`: 'none'
    (as-built, default) | 'decode' | 'encode' | any df column. `decode_metric` ∈ DECODE_METRICS (r2 |
    r2_uniform | r2_med | validfrac); `encode_metric` ∈ ENCODE_METRICS (r_med | r_mean | r2_med | r2_mean |
    frac_sig). `show_n_dim` appends the scored width to each y label as `[Nd]`. For the per-unit
    distribution use f_plot_feature_dist. `df` from f_feature_search."""
    import matplotlib.pyplot as plt
    (dcol, dttl, _), (ecol, ettl, _) = _resolve_metrics(decode_metric, encode_metric)
    d, y = _scorecard_rows(df, sort_by, dcol, ecol)
    if d.empty:
        print('no successful configs to plot'); return None
    fig, ax = plt.subplots(1, 2, figsize=(12, max(3, 0.42 * len(d) + 1)), sharey=True)
    for a, col, ttl, arrow, color in ((ax[0], dcol, dttl, 'neural → feature', 'steelblue'),
                                      (ax[1], ecol, ettl, 'feature → neural', 'indianred')):
        a.barh(y, d[col].to_numpy(dtype=float), color=color)
        a.set_title(f'{ttl}\n({arrow})')
        a.set_xlabel(f'{ttl}   [{col}]')                    # spell out the exact metric on the axis
        a.axvline(0, color='gray', lw=0.6)
        a.grid(axis='x', alpha=0.3)
    _ylab, _yttl = _scorecard_ylabels(d, show_n_dim)
    ax[0].set_yticks(y); ax[0].set_yticklabels(_ylab)
    ax[0].set_ylabel(_yttl); ax[0].invert_yaxis()           # first row at top
    fig.suptitle(f"{title or 'feature-param search — encode + decode scorecard'}"
                 f"   ({_order_label(sort_by)}, n={len(d)})")
    fig.tight_layout()
    return None


def _sweep_series(df):
    """Split a component-sweep result into one SERIES per feature set: strip the trailing `_pca…` tag
    added by the sweep pass, group by what remains → {series_label: sub-df sorted by n_dim}."""
    import re
    import pandas as pd
    d = df[df['status'] == 'ok'].copy() if 'status' in df else df.copy()
    if d.empty:
        return {}
    # strip the sweep tag: bare `_pca…`/`_svd…` or the layer-prefixed `_l1pca…` / `_l2svd…`. The op
    # stem is KEPT out of the series key deliberately — a pca and an svd pass of the same set become
    # ONE series otherwise, and the whole point is to see the two curves side by side.
    base = d['config'].astype(str).str.replace(
        r'_(?:l[12])?(pca|svd)(?:v[\d.]+(?:c\d+)?|\d+|raw)$', r'_\1', regex=True)
    d['series'] = base + ':' + d['block'].astype(str)
    if 'target' in d and d['target'].nunique() > 1:      # several targets → one line each
        d['series'] = d['series'] + ' [' + d['target'].astype(str) + ']'
    return {s: g.sort_values('n_dim') for s, g in d.groupby('series')}


def _clip_saturated(g, layer='layer1'):
    """Drop sweep points past the first one that keeps ALL components (the rest repeat the same block).
    Full width = `l1_dim_in` for a layer1 sweep, `raw_dim` for layer2. Returns (kept, n_dropped)."""
    full_col = 'l1_dim_in' if layer == 'layer1' else 'raw_dim'
    if full_col not in g or not np.isfinite(g[full_col]).any():
        return g, 0
    full = float(np.nanmax(g[full_col]))
    g = g.sort_values('n_dim')
    sat = g['n_dim'].to_numpy(float) >= full
    keep = ~sat
    if sat.any():
        keep[np.flatnonzero(sat)[0]] = True               # keep one point at full width
    return g[keep], int((~keep).sum())


def _point_label(config, n_dim, var):
    """Point label = requested value + the other currency in parentheses: a fraction request shows the
    components it gave ('0.9(34)'), a count shows the variance kept ('16(0.97)'), raw shows its width."""
    tag = _sweep_tag(config)
    if tag == 'raw' or 'c' in tag or (tag and '.' in tag):    # fraction / (frac,cap) / raw → components
        return f'{tag}({int(n_dim)})'
    return f'{tag}({var:.2f})' if var == var else tag             # count → variance kept


def _sweep_tag(config):
    """Requested sweep value from a config name: '..._l1svd16' → '16', '..._l1svdv0.95' → '0.95'."""
    import re
    m = re.search(r'_(?:l[12])?(?:pca|svd)(v?)([\d.]+(?:c\d+)?|raw)$', str(config))
    return (m.group(2) if m else '')


def f_plot_comp_sweep(df, score='encode', metric=None, layer='layer1', clip_saturated=True, logx=False,
                      annotate=True, title=''):
    """Line plot: ENCODE or DECODE performance vs components kept (n_dim), one line per feature set.
    `score`: 'encode' | 'decode'. `metric`: key of ENCODE_METRICS / DECODE_METRICS (None → 'r_med'
    for encode, 'r2' for decode).
    `layer`: which layer was swept ('layer1' | 'layer2') — sets the full width used for clipping.
    `clip_saturated`: drop points past the first that keeps every component (a set with fewer
    components than requested gives the same block again). `annotate` labels each point with the
    requested value and the other currency in parentheses: '0.9(34)' = fraction → components,
    '16(0.97)' = count → variance kept. `df` from f_feature_search. Returns None."""
    import re
    import matplotlib.pyplot as plt
    if score == 'encode':
        _, (ecol, ettl, _) = _resolve_metrics('r2', metric or 'r_med')
    elif score == 'decode':
        (ecol, ettl, _), _ = _resolve_metrics(metric or 'r2', 'r_med')
    else:
        raise ValueError(f"score must be 'encode' or 'decode', got {score!r}")
    series = _sweep_series(df)
    if not series:
        print('no successful configs to plot'); return None
    fig, a = plt.subplots(figsize=(8, 5))
    # colour = feature set, line style = target (cells solid, ensembles dashed/dotted)
    _tg = list(dict.fromkeys(df['target'])) if 'target' in df else [None]
    _styles = ['-', '--', ':', '-.']
    _bases = list(dict.fromkeys(re.sub(r' \[[^\]]*\]$', '', k) for k in series))
    _cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for lbl, g in series.items():
        _ci = _bases.index(re.sub(r' \[[^\]]*\]$', '', lbl)) % len(_cols)
        _ti = _tg.index(g['target'].iloc[0]) if 'target' in g else 0
        g = g.dropna(subset=['n_dim', ecol])
        n_drop = 0
        if clip_saturated:
            g, n_drop = _clip_saturated(g, layer)
        g = g.sort_values('n_dim')
        if g.empty:
            continue
        name = re.sub(r'^\w+:specs:', '', lbl)
        if n_drop:
            print(f'{name}: clipped {n_drop} point(s) past full width ({int(g["n_dim"].max())} dims)')
        x, y = g['n_dim'].to_numpy(float), g[ecol].to_numpy(float)
        a.plot(x, y, marker='o', ms=4, ls=_styles[_ti % len(_styles)], color=_cols[_ci], label=name)
        if annotate:
            vcol = 'l1_var_kept' if layer == 'layer1' else 'var_kept'
            vk = g[vcol].to_numpy(float) if vcol in g else np.full(len(g), np.nan)
            for xi, yi, c, v in zip(x, y, g['config'], vk):
                a.annotate(_point_label(c, xi, v), (xi, yi), textcoords='offset points', xytext=(3, 3),
                           fontsize=7)
    if logx:
        a.set_xscale('log')
    a.set_xlabel('components kept  [n_dim]')
    a.set_ylabel(f'{ettl}   [{ecol}]')
    a.axhline(0, color='gray', lw=0.6)
    a.grid(alpha=0.3)
    a.legend(fontsize=7, loc='best')
    a.set_title(title or f'{score} vs components kept ({layer} sweep)')
    fig.tight_layout()
    return None


def f_plot_pca_sweep(df, decode_metric='r2', encode_metric='r_med', title='', logx=True,
                     var_col='var_kept'):
    """PERFORMANCE vs COMPONENTS KEPT — the sweep companion to the scorecard. Two rows:
      row 1  metric vs `n_dim`     (how many dims survived)
      row 2  metric vs `var_col`   (fraction of block variance those dims explain)
    and two columns (decode / encode), so a curve saturating in dims but not in variance — or the
    reverse — is visible at a glance. One LINE PER FEATURE SET (the `_pca…` / `_l1pca…` sweep tag is
    stripped), so gabor/fft/meng/wavelet overlay directly. Points are annotated with the sweep tag.

    `var_col`: which variance column row 2 plots — 'var_kept' (the layer2/block PCA) for a LAYER-2
    sweep, or 'l1_var_kept' (the per-filter layer1 PCAs, width-weighted) for a LAYER-1 sweep. Pointing
    it at the layer you did NOT sweep gives a flat row 2, which is the tell you picked the wrong one.

    Reads the columns off the score rows (f_dataset_scan._block_size), so it works for a sweep over
    component COUNTS, variance FRACTIONS, or a mix. `df` from f_feature_search. Returns None."""
    import matplotlib.pyplot as plt
    (dcol, dttl, _), (ecol, ettl, _) = _resolve_metrics(decode_metric, encode_metric)
    series = _sweep_series(df)
    if not series:
        print('no successful configs to plot'); return None
    if var_col not in df.columns:
        raise ValueError(f"f_plot_pca_sweep: var_col {var_col!r} is not a column — use 'var_kept' "
                         f"(layer2 sweep) or 'l1_var_kept' (layer1 sweep)")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for row, (xcol, xlab) in enumerate((('n_dim', 'dims kept  [n_dim]'),
                                        (var_col, f'variance kept  [{var_col}]'))):
        for col, (mcol, mttl, arrow) in enumerate(((dcol, dttl, 'neural → feature'),
                                                   (ecol, ettl, 'feature → neural'))):
            a = ax[row, col]
            for lbl, g in series.items():
                gx = g.dropna(subset=[xcol, mcol]).sort_values(xcol)
                if gx.empty:
                    continue
                a.plot(gx[xcol].to_numpy(float), gx[mcol].to_numpy(float), 'o-', ms=4, label=lbl)
            if row == 0 and logx:
                a.set_xscale('log')
            a.set_xlabel(xlab)
            a.set_ylabel(f'{mttl}   [{mcol}]')
            a.set_title(f'{mttl}  ({arrow})' if row == 0 else '')
            a.axhline(0, color='gray', lw=0.6)
            a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7, loc='best')
    # name the feature set(s) in the title — a sweep is usually ONE feature, and then the legend says
    # nothing the title doesn't. Strip the '<kind>:specs:' prefix; list a few, else just count them.
    import re
    _names = [re.sub(r'^\w+:specs:', '', s) for s in series]
    _sub = ', '.join(_names) if len(_names) <= 3 else f'{len(_names)} feature sets'
    fig.suptitle(f"{title or 'performance vs components kept'}   [{_sub}]")
    fig.tight_layout()
    return None


def f_plot_feature_dist(df, sort_by='none', title='', decode_metric='r2_med', encode_metric='r_med',
                        show_iqr=True, show_pct=True, show_minmax=True, show_n_dim=True):
    """SECOND scorecard view — bars + percentile ERROR BARS showing the per-unit distribution behind each
    summary. Same two panels / metric selectors as f_plot_feature_search. `sort_by`: 'none' (as-built,
    default) | 'decode' | 'encode' | any df column. Each bar carries the per-unit distribution over
    feature cols (decode) or cells (encode). Toggle each overlay:
      show_iqr    25–75 box (thick line).
      show_pct    5–95 whiskers with caps (thin line).
      show_minmax min/max diamonds.
      show_n_dim  append the scored width to each y label as `[Nd]`.
    For median/mean metrics only; scalar metrics (validfrac/frac_sig) draw bars only. `df` from
    f_feature_search. Returns None."""
    import matplotlib.pyplot as plt
    (dcol, dttl, dpre), (ecol, ettl, epre) = _resolve_metrics(decode_metric, encode_metric)
    d, y = _scorecard_rows(df, sort_by, dcol, ecol)
    if d.empty:
        print('no successful configs to plot'); return None
    fig, ax = plt.subplots(1, 2, figsize=(12, max(3, 0.42 * len(d) + 1)), sharey=True)
    for a, col, ttl, pre, arrow, color, unit in (
            (ax[0], dcol, dttl, dpre, 'neural → feature', 'steelblue', 'feature cols'),
            (ax[1], ecol, ettl, epre, 'feature → neural', 'indianred', 'cells')):
        a.barh(y, d[col].to_numpy(dtype=float), color=color, alpha=0.35, zorder=1)
        need = [f'{pre}_{k}' for k in ('min', 'p5', 'p25', 'p75', 'p95', 'max')] if pre else []
        if pre and all(c in d.columns for c in need):
            mn, p5, p25, p75, p95, mx = (d[c].to_numpy(dtype=float) for c in need)
            for yi, a5, a25, a75, a95, amn, amx in zip(y, p5, p25, p75, p95, mn, mx):
                if not np.isfinite(a5):
                    continue
                if show_pct:
                    a.plot([a5, a95], [yi, yi], color='0.15', lw=1.0, zorder=3,        # 5–95 whisker
                           marker='|', ms=7, mew=1.2)
                if show_iqr:
                    a.plot([a25, a75], [yi, yi], color='0.15', lw=3.0, zorder=4)       # 25–75 box
                if show_minmax:
                    a.plot([amn, amx], [yi, yi], ls='none', marker='d', ms=4,          # min / max
                           color='firebrick', zorder=5)
            leg = ', '.join([s for s, on in (('◆=min/max', show_minmax), ('|—|=5–95', show_pct),
                                             ('▬=25–75', show_iqr)) if on]) or 'none'
            sub = f'   (bar={ttl.split("(")[-1].rstrip(")")}; {leg} over {unit})'
        else:
            sub = '   (no per-unit distribution — scalar metric)'
        a.set_title(f'{ttl}\n({arrow})')
        a.set_xlabel(f'{ttl}   [{col}]{sub}')
        a.axvline(0, color='gray', lw=0.6)
        a.grid(axis='x', alpha=0.3)
    _ylab, _yttl = _scorecard_ylabels(d, show_n_dim)
    ax[0].set_yticks(y); ax[0].set_yticklabels(_ylab)
    ax[0].set_ylabel(_yttl); ax[0].invert_yaxis()           # first row at top
    fig.suptitle(f"{title or 'feature-param search — distribution scorecard'}"
                 f"   ({_order_label(sort_by)}, n={len(d)})")
    fig.tight_layout()
    return None
