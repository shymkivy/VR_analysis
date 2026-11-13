# -*- coding: utf-8 -*-
"""
f_feature_search.py — feature-PARAMETER search on ONE dataset, scored by ENCODE + DECODE.

A `feat_sets` dict (the same format VR_ca_cebra / VR_ca_encoding use) → f_feature_configs → one config per
set → f_feature_search builds each set's feature cells and scores them:
    decode   neural → feature (f_dataset_scan.f_score_representations, procs applied before CV)
    encode   feature → neural, with the SHARED encoder config (enc_cfg → f_encoding.f_run_encoder)
             [or the older f_score_representations encode when enc_cfg is None]
Returns a tidy table (config × cell × target → scores + sizes). Driver: VR_feature_search.py.

Build: specs come from f_feature_source.f_feature_plan (ONE spec builder) and cells from
f_feature_source.f_iter_feature_cells (ONE streaming grid engine, exact cached layer-1), both shared with
cebra / encoding. Raw per conv-unit = _raw_block (fresh build + conv_autosave; memoized in the call). The
movie is LAZY (f_render_once): it renders only when a convolution is not cached or a pix block needs
pixels. Two config kinds: GRID (every bank kind) and a single raw BLOCK (pix: raw pixels → its `proc`).
A grid cell byte-identical to one already scored in the call reuses that score.

Sweeps: f_proc_sweep (layer1 / layer2 sizes, pre/post steps, pca / svd). Plots: f_plot_feature_search
(scorecard), f_plot_comp_sweep (score vs components), f_plot_pca_sweep, f_plot_feature_dist. Sizes / RAM:
f_feature_sizes. Removed 2026-11-12 (last in commit 3e5b405): f_feature_grid, f_feature_specs,
f_pyramid_grid, f_granularity_configs, f_variant_configs, f_prep_configs, f_build_grid and the
cache-load / derived-combo / feat_prep paths.

Created 2026-07-18.
"""

import hashlib
import json
import time
import numpy as np
import pandas as pd

from f_feature_helpers import f_proc, f_num_tag
from f_feature_source import (f_default_feature_plan, f_resolve_features, _layer1_names, f_feature_plan,
                              f_iter_feature_cells, _NestedFit, _l1_decomp, _l1_nested, NO_TF_KINDS,
                              f_marg_axes, f_resolve_feat_sets)
from f_dataset_scan import f_score_representations, f_nan_metrics
from f_decoding import f_imaging_fs

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


# ── Feature-grid builder (ONE param builder; memory-safe streaming: conv (sf,tf) → PCA → free raw) ──

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
    Built by f_feature_source.f_feature_plan (the ONE spec builder, shared with cebra / encoding); each
    grid block becomes a config here, pix → f_pix_config. pix takes the SAME layer1_proc / layer2_proc as every other kind — it is one block with
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
    where a set of 'fft'/'meng' entries was built as gabor (old builder chain): the kind never
    reached the builder, so gabor banks were scored under fft/meng names with no error anywhere.

    The dict is READ by f_feature_source.f_resolve_feat_sets — the same canonical reader the plan side
    (f_vis_sets_to_plan → VR_ca_cebra) uses, so one dict cannot mean two different things."""
    # ONE spec builder (f_feature_source.f_feature_plan, also used by cebra / encoding); here each plan
    # block is wrapped as a search config. pix: layer1 then layer2 back-to-back on the one block.
    out = []
    for tag, blk in f_feature_plan(feat_sets, strict=strict, tag_suffix=tag_suffix, **common).items():
        if 'grid' not in blk:                                   # pix
            out.append(f_pix_config(proc=list(blk.get('proc') or []), name=f'pix:{tag}'))
            continue
        g = blk['grid']
        out.append({'name': f"{g['kind']}:specs:{tag}",
                    'grid': {'kind': g['kind'], 'inputs': blk['inputs'], 'marginalize': g['marginalize'],
                             'layer1_proc': list(g['layer1_proc']), 'layer2_proc': list(g['layer2_proc'])},
                    'feat_prep': None})
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
    """(#cells, #conv-units) for a grid config `g` (explicit `inputs`): cells = distinct combos of the
    NON-marginalized axes; conv-units = one per input (orientation batched inside a unit)."""
    marg = _marg_axes(g['marginalize'])
    keys = set()
    for inp in g['inputs']:
        sfv = float(np.atleast_1d(inp['sf_cyc_deg'])[0]) if inp.get('sf_cyc_deg') is not None else None
        tfv = float(np.atleast_1d(inp['tf_hz'])[0]) if inp.get('tf_hz') is not None else None
        no = len(inp['odegs']) if inp.get('odegs') else (inp.get('n_orient') or 1)
        for o in ([None] if 'ori' in marg else range(no)):
            k = []
            if 'ori' not in marg:
                k.append(('o', o))
            if 'sf' not in marg and sfv is not None:
                k.append(('sf', sfv))
            if 'tf' not in marg and tfv is not None:
                k.append(('tf', tfv))
            keys.add(tuple(k))
    return len(keys), len(g['inputs'])


def f_iter_grid_cells(session, build_cfg, cfg, prerendered, raw_cache, verbose=True, timing=False,
                      keep_raw=False):
    """STREAMING feature-grid build for the search → yields (cell_name, block). Thin wrapper over the shared
    engine f_feature_source.f_iter_feature_cells (also behind cebra / encoding): raw per conv-unit from
    _raw_block (fresh build + conv_autosave), freed after its unit unless `keep_raw` (then a PROC SWEEP over
    one bank reuses it; costs n_col·T·4 B RAM per unit). Layer-1 decompositions are cached in `raw_cache`."""
    g = cfg['grid']
    inputs = g['inputs']

    def get_raw(i, inp, label):
        raw, _ft = _raw_block(session, build_cfg, label, [inp], prerendered, raw_cache, False, timing=timing)
        return raw['X'], (raw.get('raw_names') or raw['names'])

    def release(i, inp, label):
        if not keep_raw:
            raw_cache.pop(_extraction_sig(label, [inp]), None)

    yield from f_iter_feature_cells(inputs, g, get_raw, origin='build', cache=raw_cache, release=release,
                                    timing=timing)


def f_score_grid(res, metric='decode_r2'):
    """Parse a feature-grid result (f_feature_search on grid configs) into a tidy long DataFrame with `ori` (DEGREES),
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


def f_make_neural_target(est1, sigma_frames=6, peak_norm=True):
    """Neural rate proxy (T, n_active) matching VR_ca_cebra's Y_neu: drop silent cells, Gaussian-
    smooth, optional per-cell peak-normalize. Cell ORDER is irrelevant for scoring (cells are treated
    symmetrically), so the hclust sort is skipped. Dead/constant cells are dropped again in the scorer.
    Returns (Y_neu (T, n_active) float32, active_cells bool over est1['S'] rows)."""
    from f_utils import f_neural_rate
    R, active = f_neural_rate(est1['S'], sigma_frames=sigma_frames, peak_norm=peak_norm)
    return R.T.astype(np.float32), active


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
    """Score a list of candidate feature `configs` (f_feature_configs / f_proc_sweep) on ONE dataset.

    session      : f_load_session output (ONE dataset).
    configs      : GRID configs ({'name', 'grid': {kind, inputs, marginalize, layer1_proc, layer2_proc}}) or a
                   single raw BLOCK config (pix: {'name', 'blocks', 'plan', 'proc'}).
    Y_neu        : (T, n_active) neural target (f_encoding.f_enc_target); also the decode predictor.
    base_source  : f_default_source(...) — features_dir (conv_autosave root + render-shape file).
    build_cfg    : f_default_build_cfg(...) render / conv config.
    prerendered  : movie to reuse; None → f_render_once (LAZY: renders only if pixels are needed).
    keep_raw     : keep each conv-unit's raw in the call cache (a sweep over ONE bank reuses it).
    scores       : ('decode', 'encode'); a skipped one leaves its columns NaN.
    enc_cfg      : encoder config → encode with f_run_encoder (shared with VR_ca_cebra); None → the older
                   f_score_representations encode (lags_ms / r_sig / feature_proc / encode_neural_proc).
    targets      : {name: (T, n)} → every cell scored against each (built once); None → {'cells': Y_neu}.
    n_pcs / embargo_sec / n_folds / scorer_standardize / neural_zscore / neural_proc : decode scorer.
    verbose / timing : per-config summary / timestamped per-step log.
    Removed 2026-11-12 (raise if used): base_plan, force_build=False, feat_prep, feat_pre_pca_zscore.
    Returns a DataFrame: config / block / target / decode_* / encode_* / sizes / status (ok | FAIL:<Err>)."""
    if base_plan or not force_build or feat_prep is not None or feat_pre_pca_zscore:
        raise ValueError('base_plan / force_build=False / feat_prep / feat_pre_pca_zscore were removed '
                         '2026-11-12 (last in commit 3e5b405); use feat_sets layer procs / sweep_pre / '
                         'sweep_post instead')
    fs = f_imaging_fs(session['frame_t'])
    embargo = int(round(embargo_sec * fs))
    if build_cfg and build_cfg.get('conv_autosave'):     # thread the store root (from base_source) into
        build_cfg = {**build_cfg,                        # build_cfg so _raw_block can locate the store
                     'conv_autosave_root': base_source.get('features_dir')}
    if prerendered is None:                              # LAZY: renders only if a conv-unit / pix needs pixels
        prerendered = f_render_once(session, build_cfg or {}, verbose=verbose,
                                    shape_dir=base_source.get('features_dir'))
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
        is_grid = bool(cfg.get('grid'))                  # GRID config (every bank kind) | raw BLOCK (pix)
        blocks = list(cfg.get('blocks') or [name])
        prep = None                                      # feat_prep removed; column kept for old tables
        if verbose:                                      # per-config header + descriptor
            if is_grid:
                g = cfg['grid']
                desc = f"{g['kind']} · {len(g['inputs'])} specs · marg={'+'.join(g['marginalize']) or 'none'}"
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
            else:                                        # single raw BLOCK (pix): raw (memoized) → its proc
                built, frame_t = {}, None
                for b in blocks:
                    pb = (cfg.get('plan') or {}).get(b) or {}
                    if len(pb.get('inputs') or []) != 1 or pb.get('n_pca') is not None:
                        raise ValueError(f'block {b!r}: only single-input raw blocks (pix) are supported; '
                                         f'multi-input / derived / n_pca blocks were removed 2026-11-12')
                    _tb = time.perf_counter()
                    raw, frame_t = _raw_block(session, build_cfg, b, pb['inputs'], prerendered, raw_cache,
                                              False, timing=timing)
                    _X = np.asarray(raw['X'])
                    _rn = list(raw.get('raw_names') or raw['names'])
                    built[b] = {**raw, 'X': _X, 'names': _rn, 'raw_names': _rn, 'raw_dim': _X.shape[1],
                                'pca': None, 'pca_groups': None}
                    if timing:
                        print(f'      [{time.strftime("%H:%M:%S")}] ✓ raw {b} '
                              f'{time.perf_counter() - _tb:.1f}s ({_X.shape[1]} col)', flush=True)
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
