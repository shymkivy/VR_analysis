# -*- coding: utf-8 -*-
"""
f_feature_source.py — centralized, per-block feature acquisition shared across all scripts.

ONE per-block plan + ONE resolver (f_resolve_features) that every analysis script calls, so the
grid-pooled ("old") and conv_repo-sampled gabor/meng become interchangeable under the SAME block
name, and features load from whatever source is available.

Each block is defined by a LIST of `inputs` (a combiner): every input names a `source` (the
extractor identity — agg/pix/motion/self_mot/pix_mot/beh/per_obj/grid/flow/gabor/meng) plus its
params. Each input is extracted to RAW imaging-clock columns; the input columns are CONCATENATED
and then ONE block-level PCA (`n_pca`) is applied AFTER the concatenation (optional `zscore_joint` first).
A plain single-extractor block is just a 1-input block. A block declares:

    origin : 'build' (render + extract)         | 'cache' (joblib whole-registry bundle)
             'repo'  (conv_repo dense maps)      | 'auto'  (cache → repo → build by availability)
             'derived' (auto when any input has `from_block`) — assembled from OTHER already-built
             blocks, no render/cache/repo — BLOCK-LEVEL; all inputs of a block share it.
    inputs : [ {source, reduce, ds|grid, n_orient, odegs, sf_cyc_deg, tf_hz, stats,
                output, n_pix_pca (pix), transform, label}, … ]
             OR an input may be {from_block: NAME, transform, label} — reuse a PREVIOUS block's
             columns instead of extracting (the block must appear earlier in the plan / be requested;
             it is pulled in automatically). Lets a composite (e.g. gaborS+mengS) reuse the loaded
             per-input PCAs (gaborS, mengS): concat + input_norm + one joint PCA, no reconvolution.
             sf_cyc_deg : gabor/meng spatial freq in CYCLES/DEGREE (physical; repo select + build via
                          deg/px). tf_hz : meng temporal freq in HZ (physical). Legacy pixel/cyc-frame
                          selectors `wavelengths`/`temporal_freqs`/`tfreqs` still accepted.
             reduce    : 'sample' (point-sample stride `ds`) | 'grid' (grid-average) | 'raw' (pix)
             transform : per-input value map before concat — 'signed_sqrt' (energy→amplitude, e.g.
                         meng) | 'sqrt' | 'signed_log1p' | 'log1p' | None.
    n_pca  : block-level PCA after concatenation (int | float(0,1) | None | {prefix:ncomp} dict
             for single-input grouped PCA like gabor's {'gabM','gabP'}).
    input_norm : per-input normalization before concat — 'total_var' | 'first_sv' | 'standardize'
             (see f_default_feature_plan) | None.
    zscore_joint : per-column z-score of the CONCATENATED columns before the block PCA (default
             False). Distinct from input_norm='standardize' (per-input, single scalar, before concat).
    label  : per-input channel-name prefix to disambiguate repeated sources (e.g. two meng at
             different ds); auto-derived from source+params when omitted.

The plan is a plain dict passed INTO f_resolve_features (no hidden globals), so an optimizer can
mutate it to sweep feature params. Render/neural-side params that are NOT per-block (num_samp,
obj_size, side, agg_mode, terrain, smooth sigmas, n_pix_pca) live in a shared `build_cfg` dict —
reused as the render config AND the gen_config provenance.

Origins share the expensive op ONCE: all 'build' blocks → one f_build_features render; all 'cache'
blocks → one bundle load; all 'repo' blocks → one repo open. Returns (built_blocks, frame_t,
side_tags, movie) with the standard block-dict schema for every origin, so downstream is unchanged.
The block KEY stays the canonical name ('gabor', not 'gaborS_ds8'), so target lists like
sweep_target_blocks=['gabor','meng'] decode the same whether gabor came from build+grid or
repo+sample — that's the interchange.

Created 2026-07-13 (feature-source centralization; VR_ca_cebra.py is the first consumer).
"""

import copy
import json

import numpy as np

from f_pipeline import f_build_features, f_render_session
from f_feature_cache import (f_features_path, f_save_features, f_load_features,
                             f_resolve_feature_file, f_list_feature_files, f_pick_frames)
from f_conv_repo import (f_conv_repo_dir, f_conv_frame_t, f_conv_repo_meta, f_list_conv_filters,
                         f_sample_conv_to_block, f_gridavg_conv_to_block, _conv_filter,
                         _norm_ds, _ds_tag, f_deg_per_px, f_ds_from_wl, f_auto_support)
from f_visual_features import f_apply_block_pca


# Blocks build_feature_blocks ALWAYS returns (no build flag needed).
_ALWAYS_BUILT = ('agg', 'pix')
# Neural-side optional blocks → the cfg flag each needs in f_build_features.
_BUILD_FLAG = {'motion': 'build_motion', 'self_mot': 'build_self_motion',
               'pix_mot': 'build_pix_motion', 'beh': 'build_lick_reward',
               'per_obj': 'build_per_obj', 'obj_vis': 'build_obj_vis'}


# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults (single source of truth for loading params)
# ─────────────────────────────────────────────────────────────────────────────

def f_default_build_cfg(**overrides):
    """Shared render/neural-side build config (the non-per-block params). Reused as the render
    config for origin='build'/want_movie AND as gen_config provenance. `cam_params` comes from the
    session at build time, so it is NOT stored here. Override any key via kwargs."""
    cfg = {
        # which monitor(s) to build: 'L' | 'R' | 'both'
        'side':                     'both',
        # agg distance encoding: 'raw' | 'angular' | 'salience' | 'angular_salience'
        'agg_mode':                 'angular',
        # multi-object collapse: 'nearest' | 'salience_mean'
        'obj_collapse':             'salience_mean',
        # NaN the agg channels when no object present (vs collapsing to 0)
        'nan_absent':               True,
        # pixel-movie PCA dimensionality (pix block)
        'n_pix_pca':                50,
        # Gaussian σ (behavior samples) before differentiating each velocity block
        'motion_smooth_sigma':      2.0,    # object screen velocity (motion)
        'self_motion_smooth_sigma': 2.0,    # mouse egomotion (self_mot)
        'pix_motion_smooth_sigma':  2.0,    # d/dt of pix-PCs (pix_mot)
        'lick_reward_smooth_sigma': 0.0,    # 0 = raw events; >0 → smoothed rate (beh)
        # object geometry (world units) for the render + angular sizes
        'obj_size':                 {'x': 5, 'y': 20, 'z': 5, 'height': 2},
        # monitor render resolution. lat_samp (width) / vert_samp (height); None falls back to
        # num_samp (square). lat:vert = hFOV:vFOV ≈ 1.49 → angularly-square pixels (~0.66 deg/px both
        # axes at 150×101; the angle→pixel map is linear, NOT the physical 16:9).
        'num_samp':                 101,
        'lat_samp':                 150,
        'vert_samp':                101,
        # cap render length for debugging; None = full session
        'max_frames':               None,
        # add heightmap ground plane behind objects (Z-composited)
        'render_terrain':           False,
        'terrain_clip_len':         None,   # None = cam_params['clip_len']; else terrain draw distance
        'terrain_stride':           2,      # int ≥1 keeps every Nth cell; float <1 oversamples
        'terrain_point_size':       1,      # half-size of pixel patch painted per terrain sample
        # terrain surface + occlusion (when render_terrain). 'points' = scatter (gappy, objects leak);
        # 'surface' = watertight column fill so objects behind terrain are occluded. terrain_shade:
        # 'none' = invisible occluder (occlude only, no terrain content in features) | 'gray' = solid
        # gray_level | 'depth' = distance-shaded. terrain_close_iter = horizontal-close passes (smooth
        # jagged horizon + 1px slivers).
        'terrain_mode':             'surface',
        'terrain_shade':            'none',
        'terrain_gray_level':       128,
        'terrain_close_iter':       2,
        # per-object terrain-visible fraction block 'obj_vis' (needs render_terrain). Auto-enabled by
        # f_resolve_features when 'obj_vis' is requested; set True to force it on a direct build.
        'build_obj_vis':            False,
        # gabor/meng convolution backend for the BUILD path: 'gpu' (torch/f_conv_gpu, streamed — much
        # faster for long sessions) | 'cpu' (scipy fftconvolve). conv_device: torch device ('gpu' only;
        # None → auto = CUDA if available else CPU, so 'gpu' degrades gracefully when no GPU is present).
        'conv_backend':             'gpu',
        'conv_device':              None,
    }
    cfg.update(overrides)
    return cfg


def f_default_feature_plan(**overrides):
    """Shared default per-block plan (all feature loading params). Every block is a list of `inputs`
    (a combiner): each input has a `source` + params; inputs are extracted RAW, concatenated, then a
    single block-level PCA (`n_pca`) is applied. A single-extractor block is a 1-input block. Scripts
    pick a subset via f_resolve_features(only=…). Override a block with name=<dict> (merged into the
    block dict — so name={'origin':'repo'} keeps its inputs) or name=None (drop).

    origin : block-level 'auto' = first available cache → repo → build | 'cache' | 'repo' | 'build'
    inputs : [{source, reduce ('grid'|'sample'|'raw'), ds|grid, n_orient, odegs, sf_cyc_deg (cyc/deg),
               tf_hz (Hz), output, stats, n_pix_pca (pix), transform, label}, …]  (legacy px selectors
               wavelengths/temporal_freqs/tfreqs still accepted)
             transform : per-input value map — 'signed_sqrt' (energy→amplitude, e.g. meng ±8e9 →
                         ~±9e4) | 'sqrt' | 'signed_log1p' | 'log1p' | None.
    n_pca  : block PCA after concatenation (int | float(0,1) | None | {prefix:ncomp} single-input)
    input_norm : per-input normalization before concat (multi-input): 'total_var' (÷ sqrt Σvar →
                 equal total variance) | 'first_sv' (MFA; ÷ top singular value) | 'standardize'
                 ((X−mean)/std, full/global per input) | 'pca' (PCA each input separately, whitened,
                 keep `n_init` comps) | None. Scalar variants preserve within-block structure.
    n_init : ('pca' only) components kept per input's separate PCA (None = all). pca_whiten (default
             True) unit-scales them.
    zscore_joint : per-COLUMN z-score of the concatenated columns before the block PCA (default False)"""
    plan = {
        # object-position scalars (pres/lat/vert/dist per side)
        'agg':      {'origin': 'build', 'inputs': [{'source': 'agg'}]},
        # pixel movie PCs (pix PCA is intrinsic via n_pix_pca; block n_pca=None → no 2nd PCA).
        # origin='auto' so pix LOADS from the pca_feat cache when saved there (SAVE_PIX) — no render;
        # falls back to build (render + movie PCA) when absent.
        'pix':      {'origin': 'auto', 'n_pca': None,
                     'inputs': [{'source': 'pix', 'reduce': 'raw', 'n_pix_pca': 300}]},
        # object screen velocity
        'motion':   {'origin': 'build', 'inputs': [{'source': 'motion'}]},
        # mouse egomotion (speed + yaw)
        'self_mot': {'origin': 'build', 'inputs': [{'source': 'self_mot'}]},
        # d/dt of pix-PCs (needs pix → renders when built)
        'pix_mot':  {'origin': 'build', 'inputs': [{'source': 'pix_mot'}]},
        # lick + reward events
        'beh':      {'origin': 'build', 'inputs': [{'source': 'beh'}]},
        # per-object terrain-visible fraction (needs render_terrain=True — the occlusion test)
        'obj_vis':  {'origin': 'build', 'inputs': [{'source': 'obj_vis'}]},
        # retinotopic grid pooling (occupancy/mean/edge per cell)
        'grid':     {'origin': 'auto', 'n_pca': 200,
                     'inputs': [{'source': 'grid', 'reduce': 'grid', 'grid': (12, 12),
                                 'stats': ('occupancy', 'mean', 'edge')}]},
        # optical flow on the grid (speed/dir/divergence)
        'flow':     {'origin': 'auto', 'n_pca': 200,
                     'inputs': [{'source': 'flow', 'reduce': 'grid', 'grid': (12, 12),
                                 'method': 'lucas_kanade', 'stats': ('speed', 'dir', 'div')}]},
        # ---- gabor / meng blocks: PHYSICAL units (resolution/framerate-independent) ----
        # sf_cyc_deg = spatial freq in cycles/degree; tf_hz = meng temporal freq in Hz.
        # Build path: sf→px via render deg/px, tf→cyc/frame via fs (in _resolve_build).
        # Repo path (gaborS/mengS): sf_cyc_deg/tf_hz select filters from the conv_repo manifest.
        # At 150×101 (deg/px≈0.656): sf 0.085/0.10/0.13 ≈ 18/15/12 px; tf_hz 2 ≈ 0.13 cyc/frame at fs 15.
        # Legacy pixel selectors (wavelengths/temporal_freqs/tfreqs) still accepted for old px repos.
        # Gabor edge energy (magnitude + phase), grid-averaged.
        'gabor':    {'origin': 'auto', 'n_pca': {'gabM': 50, 'gabP': 50},
                     'inputs': [{'source': 'gabor', 'reduce': 'grid', 'grid': (12, 12),
                                 'n_orient': 4, 'sf_cyc_deg': (0.13, 0.26),
                                 'stats': ('magnitude', 'phase')}]},
        # motion energy (dir+speed), grid-averaged.
        'meng':     {'origin': 'auto', 'n_pca': 100,
                     'inputs': [{'source': 'meng', 'reduce': 'grid', 'grid': (12, 12),
                                 'n_orient': 4, 'sf_cyc_deg': (0.13,), 'tf_hz': (2.0,),
                                 'output': 'opponent'}]},
        # Gabor, point-sampled (A/B vs grid gabor). sf_cyc_deg/odegs limit the conv_repo load
        # (None = load ALL). repo (physical) has: sf_cyc_deg 0.085/0.10/0.13, odegs 0/45/90/135.
        # `ds` (point-sample stride on the phase-invariant magnitude map; Nyquist FLOOR ≲ σ=0.5·λ_px,
        # e.g. σ≈5.9px for the finest SF 0.13): int/(h,w) fixed | 'nyquist' → per-filter ds≈0.5·λ
        # (+optional 'ds_factor'). ds=2 OVERSAMPLES (~3 samples/σ, margin beyond Nyquist) — heavier raw
        # block (∝1/ds², PCA compresses the redundancy). Raise ds (4-6) if RAM-bound / many SF.
        'gaborS':   {'origin': 'auto', 'n_pca': 300,
                     'inputs': [{'source': 'gabor', 'reduce': 'sample', 'ds': 2,
                                 'sf_cyc_deg': (0.13,), 'odegs': None}]},
        # motion energy, point-sampled (A/B vs grid meng). +tf_hz limit (meng only). repo (physical)
        # has: sf_cyc_deg 0.085/0.10/0.13, odegs 0/45/90/135, tf_hz 1/2/3.
        'mengS':    {'origin': 'auto', 'n_pca': 300,
                     'inputs': [{'source': 'meng', 'reduce': 'sample', 'ds': 2, 'transform': 'signed_sqrt',
                                 'sf_cyc_deg': (0.13,), 'odegs': None, 'tf_hz': (1.0,)}]},
        'mengS2':    {'origin': 'auto', 'n_pca': 300,
                     'inputs': [{'source': 'meng', 'reduce': 'sample', 'ds': 2, 'transform': 'signed_sqrt',
                                 'sf_cyc_deg': (0.13,), 'odegs': None, 'tf_hz': (2.0,)}]},
        'mengS3':    {'origin': 'auto', 'n_pca': 300,
                     'inputs': [{'source': 'meng', 'reduce': 'sample', 'ds': 2, 'transform': 'signed_sqrt',
                                 'sf_cyc_deg': (0.13,), 'odegs': None, 'tf_hz': (3.0,)}]},
        
        # COMBINE gaborS + mengS → one joint-PCA block. Preprocessing knobs (see the diagnostic print):
        #   per-input 'transform' : 'signed_sqrt' (meng ENERGY ±8e9 → amplitude, matches gabor) |
        #                           'sqrt' | 'signed_log1p' | 'log1p' | None.
        #   block   'input_norm'  : how each input is put on equal footing before concat —
        #       'total_var'   ÷ sqrt Σvar (equal total variance)
        #       'first_sv'    ÷ top singular value (MFA; no leading PC dominates)
        #       'standardize' (X−mean)/std, full/global per input
        #       'pca'         per-input PCA → keep 'n_init' comps; 'pca_whiten' True = unit-var
        #                     (÷√eigenvalue) | False = keep eigenvalue scaling
        #       None          raw (huge-variance block dominates)
        #   block   'n_pca'/'zscore_joint' : joint PCA size after concat / z-score concat cols first.
        # DERIVED block: inputs reference OTHER blocks (from_block) instead of a raw source, so the
        # per-input PCAs (gaborS/mengS — loaded from cache or built once) are REUSED, concatenated,
        # balanced, and jointly PCA'd. No reconvolution. input_norm balances the two loaded inputs
        # (total_var = equal total variance; standardize/first_sv also work); n_pca = joint-PCA size.
        'gaborS+mengS':    {'origin': 'derived', 'n_pca': 500,
                            'input_norm': 'first_sv', 'zscore_joint': False,
                     'inputs': [{'from_block': 'gaborS'},
                                {'from_block': 'mengS'}]},
        # DERIVED: reuse the per-tf mengS blocks (mengS=1Hz, mengS2=2Hz, mengS3=3Hz) — pick which tf
        # each composite includes via from_block. No reconvolution.
        'gaborS+mengS2':    {'origin': 'derived', 'n_pca': 500,
                            'input_norm': 'first_sv', 'zscore_joint': False,
                     'inputs': [{'from_block': 'gaborS'},
                                {'from_block': 'mengS3'},          # 3 Hz
                                {'from_block': 'mengS'}]},         # 1 Hz
        'gaborS+mengS3':    {'origin': 'derived', 'n_pca': 500,
                            'input_norm': 'first_sv', 'zscore_joint': False,
                     'inputs': [{'from_block': 'gaborS'},
                                {'from_block': 'mengS'},
                                {'from_block': 'mengS2'},
                                {'from_block': 'mengS3'}]},        # 3 Hz only
        
        
        
        
        # EXAMPLE combine block — concat gaborS(ds8) + mengS(ds8) + mengS(ds4) RAW, then ONE PCA.
        # Opt-in by adding 'convS' to feature_list (kept out of the default set).
        # 'convS':  {'origin': 'repo', 'n_pca': 200, 'zscore_joint': False, 'inputs': [
        #                {'source': 'gabor', 'reduce': 'sample', 'ds': 8, 'label': 'gab8'},
        #                {'source': 'meng',  'reduce': 'sample', 'ds': 8, 'label': 'meng8'},
        #                {'source': 'meng',  'reduce': 'sample', 'ds': 4, 'label': 'meng4'}]},
    }
    for k, v in overrides.items():
        if v is None:
            plan.pop(k, None)
        elif k in plan and isinstance(v, dict):
            plan[k] = {**plan[k], **v}
        else:
            plan[k] = v
    return plan


PYRAMID_KINDS = ('gabor', 'fft', 'meng', 'fftglobal', 'wavelet')   # built from a filter bank
                                             # ('pix' = the raw movie, no SF axis)
# Knobs each kind actually READS (f_pyramid_specs). A knob outside its kind's list is silently dropped
# there, so f_resolve_feat_sets flags it. `transform` / ds / samp_factor / the procs apply to all.
KIND_KNOBS = {'gabor':     ('stats',),
              'fft':       ('stats', 'n_sf_sub', 'n_ori_sub', 'patch_snap'),
              'fftglobal': ('stats', 'n_sf_sub', 'n_ori_sub'),
              'wavelet':   ('family', 'decimate', 'n_levels', 'levels'),
              'meng':      ('tf_hz', 'output')}
_PIX_KNOBS = ('layer1_proc', 'layer2_proc')


def f_resolve_feat_sets(feat_sets, strict=True, tag_suffix=None, **common):
    """CANONICAL reader for the declarative feature-set dict → an ordered list of
    `(name, tag, kind, knobs)`, with `kind` and the per-entry knobs VALIDATED.

    Shared by BOTH sides so the same dict cannot mean two different things: the search
    (f_feature_search.f_feature_configs → scored configs) and the plan (f_vis_sets_to_plan → cebra
    feature_plan blocks). Same lesson as f_marg_axes — one rule, one copy.

    A reserved `'_common'` entry holds the DEFAULTS every set inherits; each other key is one set
    `{'kind': 'gabor'|'fft'|'meng'|'pix', **knobs}` stating only what differs. PRECEDENCE:
    entry > `common` KWARG > `'_common'`. `tag_suffix` appends `_<suffix>` to each name (so a
    programmatic sweep's passes don't collide). `strict` raises on an unknown kind and on a per-entry
    knob the chosen kind IGNORES; knobs inherited from '_common'/kwargs are exempt, since a shared
    default need not apply to every kind. pix gets only the procs — the raw movie has no ori/sf/tf."""
    all_kind_knobs = {k for ks in KIND_KNOBS.values() for k in ks}
    shared = {**dict((feat_sets or {}).get('_common') or {}), **common}
    out = []
    for name, spec in (feat_sets or {}).items():
        if name == '_common':
            continue
        own = dict(spec or {})
        kind = own.pop('kind', 'gabor')
        tag = f'{name}_{tag_suffix}' if tag_suffix else name
        if strict and kind != 'pix' and kind not in PYRAMID_KINDS:
            raise ValueError(f"feature set {name!r}: unknown kind {kind!r} — expected one of "
                             f"{PYRAMID_KINDS + ('pix',)}")
        if kind == 'pix':
            extra = [k for k in own if k not in _PIX_KNOBS]
            if extra and strict:
                raise ValueError(f"feature set {name!r} (kind='pix'): {extra} do not apply to the raw "
                                 f"movie, which has no ori/sf/tf axes — pix takes only "
                                 f"{' / '.join(_PIX_KNOBS)} (the same step-lists as every other kind)")
            knobs = {k: own.get(k, shared.get(k)) for k in _PIX_KNOBS}
        else:
            foreign = sorted(k for k in own if k in all_kind_knobs - set(KIND_KNOBS.get(kind, ())))
            if strict and foreign:
                raise ValueError(f"feature set {name!r} (kind={kind!r}): {foreign} is ignored by "
                                 f"kind={kind!r} — it reads {KIND_KNOBS.get(kind, ())}. "
                                 f"Drop it or fix `kind`.")
            knobs = {**shared, **own}                # bank kinds read the procs too — pass all of shared
        out.append((name, tag, kind, knobs))
    return out


def f_feature_plan(feat_sets, hFOV_deg=98.4, strict=True, tag_suffix=None, **common):
    """THE spec builder: NAMED FEATURE SETS → plan blocks {tag: block}. Used directly by cebra / encoding
    (→ f_resolve_features) AND by the search (f_feature_search.f_feature_configs wraps each block as a
    config), so one dict means one thing everywhere. `tag_suffix` / `**common` as f_resolve_feat_sets
    (sweep passes). `f_vis_sets_to_plan` is the old name (alias). Lets
    the SAME dict be pasted between VR_feature_search.py (scored by encode+decode) and VR_ca_cebra.py
    (used for the encoding arm) — same reserved `'_common'` defaults, same per-entry overrides, same
    `kind` validation, identical keyword names, and cell names that match via f_num_tag.

    `vis_sets` = {'_common': {...}, name: {kind, …}}. Every bank entry — a pyramid (`cyc_fov` / `sf_range` /
    tuple `sf_cyc_deg`) or a single SF — goes through f_pyramid_specs + f_specs_block; the search's
    f_feature_configs wraps these same blocks, so one dict builds identical inputs in every script. Expands at
    build to `<name>:<cell>` blocks. `kind='pix'` maps to the plan's raw-movie pix block with the layer1+layer2
    procs run back-to-back on it (pix has no filter axis), matching f_pix_config on the search side.
    Returns {name: plan_block} to merge into a feature_plan; add the same names to feature_list."""
    out = {}
    for nm, tag, kind, s in f_resolve_feat_sets(feat_sets, strict=strict, tag_suffix=tag_suffix, **common):
        s = dict(s)
        marg = s.pop('marginalize', DEFAULT_MARGINALIZE)
        l1, l2 = s.pop('layer1_proc', ()) or (), s.pop('layer2_proc', ()) or ()
        origin = s.pop('origin', 'build')
        if kind == 'pix':                            # raw movie: no filter axis → l1 then l2 on one block
            out[tag] = {'origin': origin,
                       'inputs': [{'source': 'pix', 'reduce': 'raw', 'n_pix_pca': None}],
                       'proc': list(l1) + list(l2)}
            continue
        # every bank entry (single SF or pyramid) → f_pyramid_specs (the search uses these same blocks).
        # (Single-SF entries used to go through f_grid_block, whose own
        # defaults — 4 orientations, fixed ds=4 — differed from the search's 8 / Nyquist; fixed 2026-11-09.)
        specs = f_pyramid_specs(kind, hFOV_deg=hFOV_deg,
                                **{k: v for k, v in s.items() if k != 'ds_factor'})
        out[tag] = f_specs_block(specs, marginalize=marg, layer1_proc=l1, layer2_proc=l2, origin=origin)
    return out


f_vis_sets_to_plan = f_feature_plan                # old name, kept for existing scripts


# Fallback when a feature-set entry names no `marginalize`. CANONICAL — f_feature_plan (used by every
# script, search included) takes its default from here,
# so an omitted marginalize cannot mean 'per SF level' in one script and 'per filter' in the other.
DEFAULT_MARGINALIZE = ('ori',)


def f_marg_spec(marginalize):
    """Marginalize spec → a TUPLE, treating a BARE STRING as one axis name. `tuple('all')` splits it
    into ('a','l','l'), which f_marg_axes then reads as three unknown axes → NOTHING pooled, silently:
    the same failure mode as the 2026-07-27 drift bug. Every site that stores a marginalize goes
    through here; f_marg_axes resolves the stored value to the concrete axis set."""
    return (marginalize,) if isinstance(marginalize, str) else tuple(marginalize or ())


def f_marg_axes(marginalize):
    """Marginalize spec → concrete axis SET. Normalizes 'wl'→'sf' and expands the 'all' shortcut to
    every axis ('ori','sf','tf'); a bare tf in a gabor/fft cell is harmless (those kinds never add a
    tf key). CANONICAL — f_feature_search._marg_axes delegates here so the search and the plan can't
    drift (they did: 'all' pooled everything in the search and NOTHING in _build_grid_block)."""
    m = {('sf' if a == 'wl' else a) for a in ((marginalize,) if isinstance(marginalize, str)
                                              else (marginalize or ()))}
    return {'ori', 'sf', 'tf'} if 'all' in m else m


def f_expand_blocks(built_blocks, names, what='block'):
    """Resolve block NAMES against a built registry, expanding a grid BANK to its cells. A name present
    verbatim passes through unchanged (so legacy 'gaborS+mengS' / 'pix' / 'self_mot' are untouched);
    otherwise it expands to every `<name>:*` key in sorted order (a bank built by f_grid_block /
    f_specs_block emits one block per cell). Order is deterministic because downstream consumers
    CONCATENATE (f_encoding._joint_feature_block). Raises listing the available keys when neither
    matches. `names` may be a single string or a list; always returns a list."""
    if isinstance(names, str):
        names = [names]
    out, missing = [], []
    for n in (names or []):
        if n in built_blocks:
            out.append(n)
            continue
        cells = sorted(k for k in built_blocks if k.startswith(f'{n}:'))
        if cells:
            out.extend(cells)
        else:
            missing.append(n)
    if missing:
        raise ValueError(f'{what}(s) {missing} not in built_blocks and no "<name>:*" cells found. '
                         f'Available: {sorted(built_blocks)}')
    return out


# ── list-of-param-dicts feature interface (shared by the search + plan) ───────────────────────────
FEATURE_SPEC_DEFAULTS = {'kind': 'gabor', 'sf_cyc_deg': 0.13, 'tf_hz': 1.0, 'ds': 4, 'ds_factor': None,
                         'n_orient': 4, 'odegs': None, 'transform': None, 'output': 'opponent',
                         'stats': ('magnitude',), 'reduce': 'sample'}
_SPEC_ALIASES = {'sf': 'sf_cyc_deg', 'tf': 'tf_hz', 'samp': 'ds', 'ori': 'odegs'}
_NYQUIST_DS = 0.5   # sample spacing at TRUE NYQUIST = 0.5·λ (2 samples / period). samp_factor scales it:
                    # spacing = _NYQUIST_DS·λ / samp_factor → samp_factor 1=Nyquist, 2=2× oversample, <1=under


def _normalize_spec(d):
    """Normalize ONE level-1 feature spec dict → an extraction INPUT dict (defaults + aliases). A spec =
    one feature: kind / sf_cyc_deg / tf_hz / ds / samp_factor / odegs|n_orient / transform / output / stats /
    reduce; omitted keys fall to FEATURE_SPEC_DEFAULTS. Aliases: sf→sf_cyc_deg, tf→tf_hz, samp→ds,
    ori→odegs. sf/tf are single-valued per spec → returned as 1-tuples for the extractor. ds='nyquist' →
    per-SF Nyquist stride; `samp_factor` scales it (1=true Nyquist, 2=2× oversample, <1=undersample);
    raw `ds_factor` (spacing = ds_factor·λ) also accepted as a low-level override. Optional `proc` = a
    per-input LEVEL-1 f_proc step-list that OVERRIDES the block layer1_proc for THIS filter (omit to
    inherit layer1_proc; [] = force no L1 proc). It rides on the input dict but is NOT an extraction
    param — the grid strips it before the conv (so it can't split the shared raw extraction)."""
    d = {_SPEC_ALIASES.get(k, k): v for k, v in d.items()}
    kind = d.get('kind', FEATURE_SPEC_DEFAULTS['kind'])
    sf = d.get('sf_cyc_deg', FEATURE_SPEC_DEFAULTS['sf_cyc_deg'])
    inp = {'source': kind, 'reduce': d.get('reduce', FEATURE_SPEC_DEFAULTS['reduce']),
           'ds': d.get('ds', FEATURE_SPEC_DEFAULTS['ds']),
           'sf_cyc_deg': (float(np.atleast_1d(sf)[0]),),
           'n_orient': d.get('n_orient', FEATURE_SPEC_DEFAULTS['n_orient']),
           'odegs': (tuple(float(a) for a in np.atleast_1d(d['odegs']))
                     if d.get('odegs') is not None else None)}
    if inp['odegs'] is not None:
        inp['n_orient'] = len(inp['odegs'])
    if d.get('samp_factor') is not None:                    # Nyquist-relative: spacing = _NYQUIST_DS·λ/samp
        inp['ds_factor'] = _NYQUIST_DS / float(d['samp_factor'])
    elif d.get('ds_factor') is not None:                    # raw coefficient (spacing = ds_factor·λ)
        inp['ds_factor'] = float(d['ds_factor'])
    elif inp['ds'] == 'nyquist':
        inp['ds_factor'] = _NYQUIST_DS                      # default = true Nyquist (samp_factor 1)
    if kind in NO_TF_KINDS:                                 # spatial-only kinds carry `stats`, no tf
        inp['stats'] = tuple(d.get('stats', FEATURE_SPEC_DEFAULTS['stats']))
        for k in KIND_KNOBS.get(kind, ()):                 # this kind's extras, passed when given
            if k != 'stats' and d.get(k) is not None:
                inp[k] = d[k]
    else:
        tf = d.get('tf_hz', FEATURE_SPEC_DEFAULTS['tf_hz'])
        inp['tf_hz'] = (float(np.atleast_1d(tf)[0]),)
        inp['output'] = d.get('output', FEATURE_SPEC_DEFAULTS['output'])
    if d.get('transform') is not None:
        inp['transform'] = d['transform']
    if d.get('proc') is not None:                           # per-input LEVEL-1 proc override (else block layer1_proc)
        inp['proc'] = list(d['proc'])
    return inp


def f_specs_block(specs, *, marginalize=(), layer1_proc=(), layer2_proc=(), origin='build'):
    """Plan-side GRID block from a LIST of level-1 feature spec dicts (the shared list-of-dicts interface;
    see _normalize_spec). Each spec = one feature (own sf/tf/ds/ori). `layer1_proc` = per-filter LEVEL-1
    proc; `marginalize` + `layer2_proc` = the optional LEVEL-2 (pool feature subsets). Built by the
    existing _build_grid_block (which iterates the explicit inputs). Mixed kinds allowed."""
    inputs = [_normalize_spec(s) for s in specs]
    kinds = {i['source'] for i in inputs}
    return {'origin': origin, 'inputs': inputs,
            'grid': {'kind': (next(iter(kinds)) if len(kinds) == 1 else 'mixed'),
                     'marginalize': f_marg_spec(marginalize),
                     'layer1_proc': list(layer1_proc), 'layer2_proc': list(layer2_proc)}}


def f_pyramid_specs(kind='gabor', *, sf_cyc_deg=None, sf_range=None, n_sf=5, cyc_fov=None, hFOV_deg=98.4,
                    n_orient=8, odegs=None, ds='nyquist', samp_factor=1.0, transform=None,
                    output='opponent', stats=('magnitude',), tf_hz=(1.0,),
                    n_sf_sub=None, n_ori_sub=None, patch_snap=None, **kind_kw):
    """PARAM GENERATOR (no proc / no marginalize) → a LIST of level-1 feature spec dicts, one per SF
    (× tf for meng), for f_specs_block (via f_feature_plan). SF list priority: sf_cyc_deg |
    cyc_fov/hFOV_deg (paper cycles-per-FOV) | geomspace(sf_range, n_sf). Per-SF sampling: spacing =
    _NYQUIST_DS·λ/samp_factor → `samp_factor` 1 = true Nyquist (λ/2), 2 = 2× oversample, <1 = undersample
    (higher SF → finer either way). Pass layer1/layer2 procs + marginalize to the consumer, not here.

    `**kind_kw` carries any KIND-SPECIFIC knob onto the spec (wavelet family/decimate/levels, …) so a new
    kind needs no signature change here — but each key must be listed in that kind's KIND_KNOBS, so a
    typo raises instead of riding through as a silently-ignored extractor kwarg."""
    if kind_kw:
        _bad = sorted(set(kind_kw) - set(KIND_KNOBS.get(kind, ())))
        if _bad:
            raise ValueError(f'f_pyramid_specs: {_bad} not a knob of kind={kind!r} '
                             f'(its knobs: {KIND_KNOBS.get(kind, ())})')
    if sf_cyc_deg is not None:
        sfs = tuple(float(s) for s in np.atleast_1d(sf_cyc_deg))
    elif cyc_fov is not None:
        sfs = tuple(float(c) / float(hFOV_deg) for c in np.atleast_1d(cyc_fov))
    elif sf_range is not None:
        sfs = tuple(float(s) for s in np.geomspace(float(sf_range[0]), float(sf_range[1]), int(n_sf)))
    else:
        raise ValueError("f_pyramid_specs needs one of sf_cyc_deg / cyc_fov / sf_range")
    tfs = [None] if kind in NO_TF_KINDS else [float(t) for t in np.atleast_1d(tf_hz)]
    specs = []
    for s in sfs:
        for t in tfs:
            d = {'kind': kind, 'sf_cyc_deg': float(s), 'ds': ds, 'samp_factor': samp_factor,
                 'n_orient': n_orient, 'odegs': odegs}
            d.update({k: v for k, v in kind_kw.items() if v is not None})   # kind-specific knobs
            if kind in NO_TF_KINDS:
                d['stats'] = tuple(stats)
                for _k, _v in (('n_sf_sub', n_sf_sub), ('n_ori_sub', n_ori_sub),
                               ('patch_snap', patch_snap)):      # fft-only; ignored by gabor
                    if _v is not None:
                        d[_k] = _v
            else:
                d['tf_hz'] = t
                d['output'] = output
            if transform is not None:
                d['transform'] = transform
            specs.append(d)
    return specs


def f_proc_feature_plan(**overrides):
    """NEW feature-building design (2026-07-22): per-block processing is an ordered f_proc step-list
    (the SAME primitive as VR_feature_search's layer1/layer2 procs), replacing n_pca/input_norm/
    zscore_joint. Every block carries `proc` (block-level == layer2; applied to the concatenated block)
    and each input MAY carry `input_proc` (per-input == layer1; applied before concat). Both are
    ordered [('norm', zscore|standardize|total_var|first_sv), ('pca', n|frac|(frac,cap)),
    ('whiten', True)] run in list order; [] = no processing. All procs run GLOBALLY at build (before
    any CV). `transform` (signed_sqrt …) stays extraction-level. Parallel to f_default_feature_plan
    (same blocks/origins/extraction params, same override semantics: name=<dict> merges, name=None
    drops); a `proc` key routes the block to _build_block_proc. pix keeps intrinsic n_pix_pca (raw
    ~30k px reduced at build), proc = post steps. VISUAL banks (gaborG/mengG) are GRID blocks (a `grid`
    key, built by f_grid_block) — per-filter L1 + marginalization L2, each expands to `<name>:<cell>`
    blocks (see _build_grid_block). Cache-loaded proc blocks carry gen-time PCA — a 'pca' step in their
    proc is SKIPPED on load (proc there = post-load norm/whiten only)."""
    plan = {
        'agg':      {'origin': 'build', 'inputs': [{'source': 'agg'}], 'proc': []},
        # pix input reduces raw pixels at build (n_pix_pca); block proc = post-PCA steps.
        'pix':      {'origin': 'auto',
                     'inputs': [{'source': 'pix', 'reduce': 'raw', 'n_pix_pca': 300}],
                     'proc': [('norm', 'zscore')]},
        'motion':   {'origin': 'build', 'inputs': [{'source': 'motion'}], 'proc': []},
        'self_mot': {'origin': 'build', 'inputs': [{'source': 'self_mot'}], 'proc': []},
        'pix_mot':  {'origin': 'build', 'inputs': [{'source': 'pix_mot'}], 'proc': []},
        'beh':      {'origin': 'build', 'inputs': [{'source': 'beh'}], 'proc': []},
        # per-object terrain-visible fraction (needs render_terrain=True to build — the occlusion test).
        'obj_vis':  {'origin': 'build', 'inputs': [{'source': 'obj_vis'}], 'proc': []},
        'grid':     {'origin': 'auto',
                     'inputs': [{'source': 'grid', 'reduce': 'grid', 'grid': (12, 12),
                                 'stats': ('occupancy', 'mean', 'edge')}],
                     'proc': [('pca', 200)]},
        'flow':     {'origin': 'auto',
                     'inputs': [{'source': 'flow', 'reduce': 'grid', 'grid': (12, 12),
                                 'method': 'lucas_kanade', 'stats': ('speed', 'dir', 'div')}],
                     'proc': [('pca', 200)]},
        # grid gabor: grouped {gabM,gabP} PCA has no single f_proc form → one block PCA (add per-stat
        # input_proc later for per-group control).
        'gabor':    {'origin': 'auto',
                     'inputs': [{'source': 'gabor', 'reduce': 'grid', 'grid': (12, 12),
                                 'n_orient': 4, 'sf_cyc_deg': (0.13, 0.26),
                                 'stats': ('magnitude', 'phase')}],
                     'proc': [('pca', 100)]},
        'meng':     {'origin': 'auto',
                     'inputs': [{'source': 'meng', 'reduce': 'grid', 'grid': (12, 12),
                                 'n_orient': 4, 'sf_cyc_deg': (0.13,), 'tf_hz': (2.0,),
                                 'output': 'opponent'}],
                     'proc': [('pca', 100)]},
        # (gaborG / mengG inline grid banks removed 2026-11-12 — define banks with feat_sets / f_feature_plan)
    }
    for k, v in overrides.items():
        if v is None:
            plan.pop(k, None)
        elif k in plan and isinstance(v, dict):
            plan[k] = {**plan[k], **v}
        else:
            plan[k] = v
    return plan


def f_default_source(**overrides):
    """Resolver source-location kwargs → pass to f_resolve_features(**source). Bundles WHERE blocks
    come from + the autosave opts. Defaults set below; override a key by passing name=value."""
    mouse_id = overrides.pop('mouse_id', 'L')           # which mouse (default)
    feat_root = overrides.pop('feat_root', 'F:/VR/features/')   # features root (mouse appended)
    tag = 'gen_7_16_26'                 # preset cache/repo generation tag (bump on regen; physical-units repo)
    source = {
        'features_dir': feat_root + mouse_id,   # root holding cache/ and conv_repo/
        'mouse_id':     mouse_id,       # names the saved cache file
        'cache_tag':    tag,            # joblib cache tag (origin 'cache'/'auto')
        'conv_tag':     tag,            # conv_repo GEN_TAG (origin 'repo'/'auto')
        'cache_name':   None,           # exact/substring cache filename (overrides tag)
        'cache_idx':    0,              # sorted-file index (last-resort fallback)
        'save':         False,          # autosave the merged registry (build path only)
        'save_tag':     tag,            # tag baked into the saved filename
        'save_movie':   False,          # also store the raw movie in the cache (large)
    }
    source.update(overrides)
    return source


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _side_tags_from_build_cfg(build_cfg):
    side = (build_cfg or {}).get('side', 'both')
    return {'L': ['L'], 'R': ['R'], 'both': ['L', 'R']}.get(side, ['L', 'R'])


# Source taxonomy: which sources come from build_feature_blocks (cheap+pix), which are visual
# extractors (parameterised per-input), and which sources force a monitor render.
_VISUAL_SOURCES = ('grid', 'flow', 'gabor', 'meng', 'fft', 'fftglobal', 'wavelet')
_CHEAP_SOURCES = _ALWAYS_BUILT + tuple(_BUILD_FLAG)     # agg,pix,motion,self_mot,pix_mot,beh,per_obj,obj_vis
# obj_vis needs the render (objects + terrain depth) for the occlusion test — not render-free.
_RENDER_SOURCES = ('pix', 'pix_mot', 'obj_vis') + _VISUAL_SOURCES  # sources that require the monitor movie
_VISUAL_PARAM_KEYS = ('n_orient', 'odegs', 'wavelengths', 'stats', 'temporal_freqs', 'output',
                      'sigma_scale', 'gamma', 'temporal_support', 'temporal_sigma', 'presmooth_sigma',
                      'grid', 'method', 'n_sf_sub', 'n_ori_sub', 'patch_snap', 'edge_min_frac',
                      'family', 'decimate', 'n_levels', 'levels')      # wavelet

# Kinds with NO temporal axis: they carry `stats` and are emitted once per SF (no tf loop). 'fft' joins
# 'gabor' here — its patch size is tied to lambda, so the pyramid is one call per level, like gabor.
NO_TF_KINDS = ('gabor', 'fft', 'fftglobal', 'wavelet')


def _vk(block_name, idx):
    """Deterministic build_visual_blocks spec KEY for input `idx` of a block (so repeated sources at
    different params get separate raw extractions). The key names the block dict, not channels."""
    return f'__{block_name}__in{idx}'


def _input_label(inp):
    """Default channel-name prefix for an input (used when a block has >1 input or on explicit
    request): source + the discriminating reduction param (ds / grid)."""
    if inp.get('from_block'):                                 # input reuses a previous block
        return inp['from_block']
    src = inp['source']
    if inp.get('reduce') == 'sample' and 'ds' in inp:
        return f"{src}_ds{_ds_tag(inp['ds'])}"
    if inp.get('reduce') == 'grid' and 'grid' in inp:
        return f"{src}_g{inp['grid'][0]}x{inp['grid'][1]}"
    return src


def _input_transform(X, kind):
    """Optional per-input value transform BEFORE scaling/concat. `kind`:
      None            → unchanged.
      'signed_sqrt'   → sign(x)·√|x|  (ENERGY → amplitude units; meng ±8e9 → ~±9e4, matches gabor
                        magnitude; also variance-stabilizing, tames heavy tails).
      'sqrt'          → √max(x,0)     (one-sided energy → amplitude).
      'signed_log1p'  → sign(x)·log1p(|x|) ; 'log1p' → log1p(max(x,0))  (stronger compression)."""
    if not kind:
        return X
    X = np.asarray(X, np.float32)                       # float32 + in-place → half the memory of float64
    if kind == 'signed_sqrt':                           # (wide sampled meng blocks are many GB)
        a = np.abs(X); np.sqrt(a, out=a); return np.copysign(a, X, out=a)
    if kind == 'sqrt':
        return np.sqrt(np.clip(X, 0, None))
    if kind == 'signed_log1p':
        a = np.abs(X); np.log1p(a, out=a); return np.copysign(a, X, out=a)
    if kind == 'log1p':
        return np.log1p(np.clip(X, 0, None))
    raise ValueError(f"unknown transform {kind!r} "
                     "(use 'signed_sqrt'|'sqrt'|'signed_log1p'|'log1p'|None)")


def _zscore_cols(X):
    """NaN-safe per-column z-score (mirrors f_combine_blocks_pca's zscore tail)."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return np.nan_to_num((X - mu) / sd, nan=0.0)


def _input_scale_rank(X, max_frames=1500, tol=1e-9):
    """Scale + rank of one raw input BEFORE concatenation → dict(width, total_var, mean, std,
    eff_rank, hard_rank, n_frames, first_ev). `total_var` (Σ column variance = trace) is the SCALE;
    `mean`/`std` are the GLOBAL mean/std over all values; `eff_rank` = participation ratio (Σλ)²/Σλ²
    (effective dimensionality); `hard_rank` = # eigenvalues > tol·max. Spectrum estimated from ≤
    max_frames evenly-spaced frames to bound the SVD cost (scale/mean/std on the full data). NaN→0."""
    X = np.nan_to_num(np.asarray(X, float), nan=0.0)
    total_var = float(X.var(axis=0).sum())                 # SCALE — full data, cheap
    gmean = float(X.mean())                                 # global mean/std over all values
    gstd = float(X.std())
    T = X.shape[0]
    Xc = X[np.linspace(0, T - 1, max_frames).astype(int)] if T > max_frames else X
    Xc = Xc - Xc.mean(axis=0)
    first_ev = 0.0
    try:
        ev = np.linalg.svd(Xc, full_matrices=False, compute_uv=False) ** 2
        tot = float(ev.sum())
        pr = float(tot ** 2 / np.square(ev).sum()) if tot > 0 else 0.0
        hard = int((ev > tol * ev.max()).sum()) if ev.size and ev.max() > 0 else 0
        first_ev = float(ev.max()) if ev.size else 0.0     # top eigenvalue (MFA scale)
    except np.linalg.LinAlgError:
        pr, hard = float('nan'), -1
    return {'width': int(X.shape[1]), 'total_var': total_var, 'mean': gmean, 'std': gstd,
            'eff_rank': pr, 'hard_rank': hard, 'n_frames': int(Xc.shape[0]), 'first_ev': first_ev}


def _visual_spec_from_input(inp, deg_per_px=None, fs=None):
    """Translate a visual INPUT dict → a build_visual_blocks spec forced to RAW (n_pca=None); the
    block-level PCA is applied later by the combiner. `reduce` maps to pool/sample_ds/grid for
    gabor/meng; grid/flow ignore it. Always carries `type`=source (the spec key is a unique token).
    PHYSICAL units: an input's `sf_cyc_deg` → `wavelengths` (px) via `deg_per_px`, and `tf_hz` →
    `temporal_freqs` (cyc/frame) via `fs` — so the build path can be specified in cyc/deg + Hz too."""
    src = inp['source']
    p = {k: v for k, v in inp.items() if k in _VISUAL_PARAM_KEYS}   # keep recognized extractor params
    if 'sf_cyc_deg' in inp and deg_per_px:                 # cyc/deg → pixel wavelength
        p['wavelengths'] = tuple(1.0 / (float(s) * deg_per_px) for s in np.atleast_1d(inp['sf_cyc_deg']))
    if 'tf_hz' in inp and fs:                              # Hz → cycles/frame
        p['temporal_freqs'] = tuple(float(h) / fs for h in np.atleast_1d(inp['tf_hz']))
    if p.get('temporal_support') == 'auto':                # build path is ONE call → single support
        if fs and 'tf_hz' in inp:                          # size for the LOWEST tf (all tf resolvable)
            _tf_lo = min(float(h) for h in np.atleast_1d(inp['tf_hz']))
            p['temporal_support'] = f_auto_support(_tf_lo, fs, inp.get('n_cycles', 2.0),
                                                   inp.get('support_min', 7), inp.get('support_max', 91))
        else:
            p.pop('temporal_support', None)                # no fs → extractor default
    p['n_pca'] = None                                      # RAW; block PCA runs in the combiner
    p['type'] = src                                        # spec key is a unique token, not the kind
    if src in ('gabor', 'meng', 'fft'):                    # reduce → pool / sample_ds / grid
        #   fftglobal has no lattice and wavelet defines its own, so neither takes `reduce`
        if inp.get('reduce', 'grid') == 'sample':
            p['pool'] = 'sample'
            _ds = inp.get('ds', 1)
            if _ds == 'nyquist' or inp.get('ds_factor') is not None:   # per-SF Nyquist stride (build path)
                _wls = p.get('wavelengths')
                if not _wls:
                    raise ValueError("ds='nyquist' needs sf_cyc_deg (+deg_per_px) to size the stride")
                _wl = float(min(_wls))                     # finest SF → finest stride
                if len(_wls) > 1:
                    print(f"  [nyquist] multi-SF sample input — using finest λ={_wl:.1f}px stride for all")
                p['sample_ds'] = f_ds_from_wl(_wl, factor=float(inp.get('ds_factor', 0.5)))
            else:
                p['sample_ds'] = _ds
        else:
            p['pool'] = 'grid'
            p.setdefault('grid', inp.get('grid', (12, 12)))
    return p


def _block_sources(block):
    """Set of source-kinds across a block's inputs (for the repo-availability check). Inputs that
    reference a previous block (`from_block`) have no source-kind and are excluded."""
    return {inp['source'] for inp in block.get('inputs', []) if inp.get('source')}


def _block_has_from(block):
    """True if any input reuses a previously-built block via `from_block` (→ 'derived' origin)."""
    return any(isinstance(i, dict) and i.get('from_block') for i in (block.get('inputs') or []))


def _resolve_origin_for(name, block, cache_keys, repo_kinds):
    """Resolve one BLOCK's effective origin. Explicit origin passes through. 'auto' → 'cache' if the
    block name is IN the loaded cache bundle, else 'repo' if EVERY input source is a repo kind, else
    'build'. (Same-origin v1: the whole block shares one origin.) A block whose inputs reference other
    blocks (`from_block`) is 'derived' — assembled from those already-built blocks, no render/cache/repo."""
    if _block_has_from(block):
        return 'derived'
    origin = block.get('origin', 'auto')
    if origin != 'auto':
        return origin
    if name in cache_keys:
        return 'cache'                                     # cached under this exact block name
    srcs = _block_sources(block)
    if srcs and srcs <= set(repo_kinds):
        return 'repo'                                      # every input's kind is in the repo
    return 'build'                                         # render + extract


def _resolve_build(session, build_cfg, build_names, plan, render_verbose, prerendered=None, verbose=True):
    """Run ONE f_build_features for all origin='build' blocks. Returns a SHARED result dict keyed by
    cheap-source name (agg/pix/motion/…) AND by unique visual-input tokens (_vk), plus frame_t /
    side_tags / movie. The combiner selects each input's raw columns from this dict. Renders only if
    any input source needs the movie (pix/pix_mot/visual)."""
    cfg = dict(build_cfg or {})
    all_inputs = [(n, i, inp) for n in build_names for i, inp in enumerate(plan[n]['inputs'])]
    srcs = {inp['source'] for _, _, inp in all_inputs}
    for src, flag in _BUILD_FLAG.items():
        cfg[flag] = src in srcs                             # enable each optional cheap block requested
    # pix is in _ALWAYS_BUILT, but building it means flattening the whole movie to (T, H*W*n_side)
    # float32 + a PCA — ~60 s and ~12 GB at T=96k. Only pay it when a block actually asked for pix.
    cfg['build_pix'] = 'pix' in srcs
    cfg['build_agg'] = 'agg' in srcs                       # same for agg (1.5 s of each visual fetch)
    for _, _, inp in all_inputs:                            # lift per-input pix build params into cfg
        if inp['source'] == 'pix':
            if 'n_pix_pca' in inp:
                cfg['n_pix_pca'] = inp['n_pix_pca']
            if 'pix_pre_proc' in inp:
                cfg['pix_pre_proc'] = inp['pix_pre_proc']
            break
    # physical→pixel conversion geometry for any sf_cyc_deg / tf_hz visual inputs
    _lat = cfg.get('lat_samp') or cfg.get('num_samp', 101)
    _vert = cfg.get('vert_samp') or cfg.get('num_samp', 101)
    _dpp = f_deg_per_px(session['cam_params'], _lat, _vert)
    _ft = session.get('frame_t')
    _fs = (1.0 / float(np.median(np.diff(_ft)))) if _ft is not None and len(_ft) > 1 else None
    vis_specs = {_vk(n, i): _visual_spec_from_input(inp, deg_per_px=_dpp, fs=_fs)  # one RAW extraction/input
                 for n, i, inp in all_inputs if inp['source'] in _VISUAL_SOURCES}
    cfg['build_visual'] = bool(vis_specs)
    if vis_specs:
        cfg['visual_specs'] = vis_specs
    render_needed = any(inp['source'] in _RENDER_SOURCES for _, _, inp in all_inputs)
    built, frame_t, side_tags, movie = f_build_features(
        session, cfg, render_verbose=render_verbose, render=render_needed, prerendered=prerendered,
        verbose=verbose)
    return built, np.asarray(frame_t).ravel(), list(side_tags), movie


def _extract_input(inp, block_name, idx, origin, shared, verbose, prior=None):
    """Extract ONE input to RAW imaging-clock columns → (X, names, src_meta). `src_meta` is the source
    block (build) or a minimal {kind,params} (repo) used to carry metadata for single-input blocks.
    An input with `from_block` reuses an ALREADY-BUILT block's columns from `prior` (no extraction)."""
    if inp.get('from_block'):                                 # reuse a previously-built block's columns
        fb = inp['from_block']
        if not prior or fb not in prior:
            raise ValueError(f"block {block_name!r} input {idx}: from_block {fb!r} is not built yet "
                             f"(available: {list(prior or [])}) — list it earlier in the plan")
        b = prior[fb]
        X, names = np.asarray(b['X']), list(b['names'])
        sel = inp.get('groups')                               # optional: keep only matching PCA groups
        if sel:                                               # (grouped/pcasep block only), e.g. ['tf2','tf0']
            pg = b.get('pca_groups') or {}
            if not pg:
                raise ValueError(f"from_block {fb!r} has no pca_groups to select {sel} — the block must be "
                                 f"grouped (load the 'pcasep' variant, per-filter PCA)")
            cols = sorted(c for k, info in pg.items() if any(s in k for s in sel)
                          for c in range(*info['xslice']))
            if not cols:
                raise ValueError(f"from_block {fb!r}: groups {sel} matched none of {list(pg)}")
            X, names = X[:, cols], [names[c] for c in cols]
        return X, names, b
    src = inp['source']
    if origin == 'build':                                  # select raw columns from the shared render
        built = shared['build']
        key = _vk(block_name, idx) if src in _VISUAL_SOURCES else src
        if key not in built:
            raise ValueError(f"build did not produce source {src!r} (key {key!r}) for block "
                             f"{block_name!r}; produced {list(built)}")
        b = built[key]
        return np.asarray(b['X']), list(b['names']), b
    if origin == 'repo':                                   # sample / grid-average the dense conv maps
        rd = shared['repo']
        sel = _conv_filter(kinds=(src,), wavelengths=inp.get('wavelengths'),
                           odegs=inp.get('odegs'), tfreqs=inp.get('tfreqs'),
                           sf_cyc_deg=inp.get('sf_cyc_deg'), tf_hz=inp.get('tf_hz'))
        if inp.get('reduce', 'sample') == 'grid':
            grid = tuple(inp.get('grid', (12, 12)))
            X, names = f_gridavg_conv_to_block(rd['repo_dir'], grid, select=sel,
                                               side_tags=rd['side_tags'], verbose=verbose)
            params = {'pool': 'grid', 'grid': grid}
        else:
            _ds = inp.get('ds', 1)
            if _ds == 'nyquist' or inp.get('ds_factor') is not None:   # per-filter Nyquist stride
                _f = float(inp.get('ds_factor', 0.5))
                ds = (lambda e, _f=_f: f_ds_from_wl(e['wavelength'], factor=_f))
                params = {'pool': 'sample', 'sample_ds': 'nyquist', 'ds_factor': _f}
            else:
                ds = _norm_ds(_ds)
                params = {'pool': 'sample', 'sample_ds': ds}
            X, names = f_sample_conv_to_block(rd['repo_dir'], ds, select=sel,
                                              side_tags=rd['side_tags'], verbose=verbose)
        return np.asarray(X), list(names), {'kind': src, 'params': params}
    raise ValueError(f"origin {origin!r} has no per-input extractor (cache is pass-through)")


def _layer1_names(Xp, raw_nms, glabel):
    """Column names for one filter after layer1: the raw names if the width is unchanged, else
    '<glabel>_PC1..k' (layer1 reduced the filter)."""
    k = int(np.asarray(Xp).shape[1])
    return list(raw_nms) if k == len(raw_nms) else [f'{glabel}_PC{i + 1}' for i in range(k)]


# ── ONE grid-cell engine for every script (2026-11-09, PLAN_feature_builder_merge.md step 2–3) ──────────
# Used by f_feature_search.f_iter_grid_cells (search; raw = per-unit build + conv_autosave) and by
# _build_grid_block (cebra / encoding; raw = shared build or conv_repo). Exact layer-1 decomposition is
# fit ONCE per (unit, filter, pre-steps) and sliced, with the same k / identity rules as _pca_fit/_svd_fit.

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


def _layer1_apply(X, l1, cache, ckey):
    """LAYER-1 step-list on one filter's columns. norm-only pre steps → pca/svd from a decomposition fit
    ONCE per `ckey` (in `cache`) and sliced → trailing norm / whiten. Anything else → plain f_proc.
    Returns (Xp, fitted_steps)."""
    from f_feature_helpers import f_proc
    ir = next((i for i, st in enumerate(l1) if st[0] in ('pca', 'svd')), None)
    lead = l1[ir] if ir is not None else None
    if not (lead and lead[1] and X.shape[1] >= 2 and all(st[0] == 'norm' for st in l1[:ir])):
        return f_proc(X, l1)
    pre = l1[:ir]
    Xin = f_proc(X, pre)[0] if pre else X
    dk = ckey + (lead[0], repr(pre))
    if dk not in cache:
        cache[dk] = _l1_decomp(Xin, lead[0])
    Xp, p = _l1_nested(Xin, lead[0], lead[1], cache[dk])
    fit = [(lead[0], {'pca': p, 'eig': None if p is None else p.explained_variance_})]
    for st in l1[ir + 1:]:                                # trailing steps
        if st[0] == 'whiten':                             # ÷ √variance of each kept component
            if p is not None:
                ev = p.explained_variance_
                Xp = Xp / np.sqrt(np.where(ev > 0, ev, 1.0)).astype(np.float32)
            fit.append(('whiten', {'eig': None if p is None else p.explained_variance_}))
        else:
            Xp, rest = f_proc(Xp, [st])
            fit += rest
    return Xp, fit


def f_iter_feature_cells(inputs, grid, get_raw, *, origin='build', cache=None, release=None,
                         timing=False):
    """THE grid-cell engine (STREAMING generator) behind every feature-set build. `inputs` = one spec per
    conv-unit (one SF [× TF], all orientations); `grid` = {kind, marginalize, layer1_proc, layer2_proc}.
    Per unit: `get_raw(i, inp, label)` → (X (T, n), raw_names) already transformed → per filter (o, sf,
    tf) the layer-1 proc (a per-input `inp['proc']` overrides it) → filters routed to CELLS by the
    NON-marginalized axes → a cell is finalized (concat → layer-2 proc) and YIELDED as soon as its last
    contributing unit is done, then `release(i, inp, label)` frees that unit's raw. Peak RAM ≈ one unit's
    raw + the open cells. `cache` holds the layer-1 decompositions (reused across calls when shared).
    Yields (cell_name, block); cell names carry physical values (o<deg>, sf<cyc/deg>, tf<Hz>), 'all'
    when every axis is pooled. `origin='repo'` → orientation in the names is already DEGREES."""
    import re
    import time
    from collections import OrderedDict
    from f_feature_helpers import f_proc, f_num_tag
    from f_visual_features import f_pca_group_cols
    cache = {} if cache is None else cache
    marg = f_marg_axes(grid.get('marginalize', ()))
    l1_blk = list(grid.get('layer1_proc') or [])
    l2 = list(grid.get('layer2_proc') or [])
    units = []                                            # (i, label, inp, sfv, tfv)
    for i, inp in enumerate(inputs):
        inp = dict(inp)
        src = inp.get('source')
        sfv = float(np.atleast_1d(inp['sf_cyc_deg'])[0]) if inp.get('sf_cyc_deg') is not None else None
        tfv = float(np.atleast_1d(inp['tf_hz'])[0]) if inp.get('tf_hz') is not None else None
        label = (f'{src}_sf{f_num_tag(sfv)}' if sfv is not None else f'{src}_in{i}') \
            + ('' if (src in NO_TF_KINDS or tfv is None) else f'tf{f_num_tag(tfv)}')
        units.append((i, label, inp, sfv, tfv))

    def _later(key, u):                                   # does a unit after u feed this cell?
        kd = dict(key)
        fsf, ftf = kd.get('sf'), kd.get('tf')
        return any((fsf is None or sfv2 == fsf) and (ftf is None or tfv2 == ftf)
                   for (_i, _l, _p, sfv2, tfv2) in units[u + 1:])

    def _finalize(key, members):                          # concat → layer 2 → standard block
        cell = '_'.join(f'{ax}{f_num_tag(val) if isinstance(val, (int, float)) else val}'
                        for ax, val in key) or 'all'
        Xcat = np.nan_to_num(np.concatenate([X for _, X, _, _, _ in members], axis=1))
        raw_names = [f'{lbl}:{nm}' for lbl, _, nms, _, _ in members for nm in nms]
        Xp, fit = f_proc(Xcat, l2)
        _pca = next((p['pca'] for op, p in fit if op in ('pca', 'svd') and p.get('pca') is not None), None)
        names = ([f'{cell}_PC{k + 1}' for k in range(Xp.shape[1])] if _pca is not None
                 else list(raw_names))
        l1v = [(w, w * v) for _, _, _, w, v in members if v == v]              # skip NaN (no l1 pca)
        return cell, {'X': np.asarray(Xp, np.float32), 'names': names, 'pca': _pca, 'pca_groups': None,
                      'raw_names': raw_names, 'raw_dim': Xcat.shape[1],
                      'source_blocks': [lbl for lbl, _, _, _, _ in members], 'proc_fit': fit,
                      'l1_var_kept': (sum(v for _, v in l1v) / sum(w for w, _ in l1v) if l1v else np.nan),
                      'l1_dim_in': sum(w for _, _, _, w, _ in members), 'kind': cell}

    cells = OrderedDict()
    for u, (i, label, inp, sfv, tfv) in enumerate(units):
        t0 = time.perf_counter()
        if timing:
            print(f'      [{time.strftime("%H:%M:%S")}] → conv {label} [{u + 1}/{len(units)}]', flush=True)
        X, rn = get_raw(i, inp, label)
        X = np.asarray(X)
        if timing:
            print(f'      [{time.strftime("%H:%M:%S")}] ✓ conv {label} {time.perf_counter() - t0:.1f}s '
                  f'(raw {X.shape[1]} col)', flush=True)
        l1 = l1_blk if inp.get('proc') is None else list(inp['proc'])     # per-input layer-1 override
        odegs, n_ori = inp.get('odegs'), (inp.get('n_orient') or 1)
        for glabel, info in f_pca_group_cols(rn, 'none', None).items():  # one group per filter
            cols = info['cols']
            Xp, f1 = _layer1_apply(X[:, cols], l1, cache, ('l1fit', label, json.dumps(inp, sort_keys=True,
                                                                                      default=str), glabel))
            p1 = next((p['pca'] for op, p in f1 if op in ('pca', 'svd') and p.get('pca') is not None), None)
            v1 = (float(np.sum(p1.explained_variance_ratio_)) if p1 is not None
                  else (1.0 if l1 else np.nan))           # no pca fired → nothing dropped
            mo = re.search(r'_o(\d+)', glabel) or re.search(r'o(\d+)', glabel)
            io = int(mo.group(1)) if mo else 0
            if origin == 'repo':                          # repo names carry orientation DEGREES
                odeg = float(io)
            elif odegs is not None:
                odeg = float(np.atleast_1d(odegs)[io])
            else:
                odeg = io * 180.0 / n_ori
            key = []
            if 'ori' not in marg:
                key.append(('o', f'{odeg:g}'))
            if 'sf' not in marg and sfv is not None:
                key.append(('sf', sfv))                   # raw float (contributor match); formatted in name
            if 'tf' not in marg and tfv is not None:
                key.append(('tf', tfv))
            cells.setdefault(tuple(key), []).append(
                (f'{label}:{glabel}', Xp, _layer1_names(Xp, [rn[c] for c in cols], glabel), len(cols), v1))
        if release is not None:
            release(i, inp, label)
        del X
        for key in [k for k in cells if not _later(k, u)]:
            t1 = time.perf_counter()
            out = _finalize(key, cells.pop(key))
            if timing:
                print(f'      [{time.strftime("%H:%M:%S")}] ✓ finalize/PCA {out[0]} '
                      f'{time.perf_counter() - t1:.1f}s ({out[1]["X"].shape[1]} dim)', flush=True)
            yield out


def _build_grid_block(name, block, origin, shared, verbose, prior=None):
    """GRID block (cebra / encoding plan side) → {f'{name}:{cell}': block}. Thin wrapper over the shared
    engine f_iter_feature_cells (same engine as the feature search); raw per input comes from the shared
    build or the conv_repo via _extract_input, then _input_transform."""
    gb = shared.get('grid_build')

    def get_raw(i, inp, label):
        if origin == 'build' and gb is not None and inp.get('source') in _VISUAL_SOURCES:
            key = f'{name}#u{i}'                          # one conv-unit, built on its own (cache-keyed)
            built, _ft, _st, _mv = _resolve_build(gb['session'], gb['build_cfg'], [key],
                                                  {key: {'inputs': [inp]}}, gb['render_verbose'],
                                                  prerendered=gb['prerendered'], verbose=False)
            b = built[_vk(key, 0)]
            Xi, ni = b['X'], list(b['names'])
        else:
            Xi, ni, _meta = _extract_input(inp, name, i, origin, shared, verbose, prior=prior)
        return np.asarray(_input_transform(Xi, inp.get('transform'))), ni
    out = {}
    for cell, blk in f_iter_feature_cells(block['inputs'], block['grid'], get_raw, origin=origin):
        outname = f'{name}:{cell}'
        blk['kind'], blk['inputs'] = outname, block['inputs']
        out[outname] = blk
    return out


def _build_block_proc(name, block, origin, shared, verbose, prior=None):
    """Proc-based combiner (NEW design; routed here when a block has a `proc` key). Each input is
    extracted RAW → `transform` → per-input `input_proc` (f_proc, == grid layer1); inputs are
    concatenated → block `proc` (f_proc, == grid layer2). `proc`/`input_proc` are ordered f_proc
    step-lists — the SOLE processing (no n_pca/input_norm/zscore_joint). Cache origin passes the
    stored block through + any POST-ops (norm/whiten) in `proc` (a 'pca' step is skipped — the cache
    is already reduced). Mirrors _build_block's out-dict shape + single-input metadata inheritance."""
    from f_feature_helpers import f_proc
    proc = list(block.get('proc') or [])
    if origin == 'cache':
        bb = shared['cache']['built_blocks']
        if name not in bb:
            raise ValueError(f'cache lacks block {name!r}; has {list(bb)}')
        out = dict(bb[name])
        out['inputs'] = block.get('inputs')
        post = [s for s in proc if s[0] != 'pca']           # cache already PC'd → skip pca, keep norm/whiten
        if len(post) != len(proc) and verbose:
            print(f"  [proc {name!r}] cache block already reduced — skipping 'pca' step in proc")
        if post:
            out['X'] = np.asarray(f_proc(np.asarray(out['X'], float), post)[0], np.float32)
        return out
    inputs = block['inputs']
    single = len(inputs) == 1
    Xs, ni_list, metas = [], [], []
    for i, inp in enumerate(inputs):
        Xi, ni, meta = _extract_input(inp, name, i, origin, shared, verbose, prior=prior)
        Xi = _input_transform(Xi, inp.get('transform'))     # extraction-level value map (feature def)
        ip = list(inp.get('input_proc') or [])              # per-input f_proc (== grid layer1)
        lab = inp.get('label') or _input_label(inp)
        if ip:
            Xp, ifit = f_proc(np.asarray(Xi), ip)
            _ipca = next((p['pca'] for op, p in ifit if op in ('pca', 'svd') and p.get('pca') is not None), None)
            ni = ([f'{lab}:ipc{j}' for j in range(Xp.shape[1])] if _ipca is not None
                  else (ni if (single and not inp.get('label')) else [f'{lab}:{c}' for c in ni]))
            Xi = Xp
        elif not (single and not inp.get('label')):         # single-input, no label → NO prefix
            ni = [f'{lab}:{c}' for c in ni]
        Xs.append(np.asarray(Xi)); ni_list.append(ni); metas.append(meta)
    raw_names = [n for ni in ni_list for n in ni]
    X_cat = Xs[0] if single else np.concatenate(Xs, axis=1)
    raw_dim = int(np.asarray(X_cat).shape[1])
    Xp, fit = f_proc(np.asarray(X_cat), proc)               # block-level f_proc (== grid layer2)
    _pca = next((p['pca'] for op, p in fit if op in ('pca', 'svd') and p.get('pca') is not None), None)
    names = ([f'{name}_PC{i + 1}' for i in range(Xp.shape[1])] if _pca is not None else list(raw_names))
    X = np.asarray(Xp, np.float32)
    if single and metas[0] is not None:                     # inherit source metadata; refresh proc fields
        out = dict(metas[0])
        out.update({'X': X, 'names': names, 'pca': _pca, 'pca_groups': None,
                    'raw_names': raw_names, 'raw_dim': raw_dim, 'inputs': inputs, 'proc_fit': fit})
        out.setdefault('kind', inputs[0]['source'])
    else:
        out = {'X': X, 'names': names, 'pca': _pca, 'pca_groups': None, 'raw_names': raw_names,
               'raw_dim': raw_dim, 'kind': name, 'inputs': inputs, 'params': {}, 'proc_fit': fit}
    return out


def _build_block(name, block, origin, shared, verbose, prior=None):
    """Combine a block's inputs → the standard block dict. Cache origin passes the stored block
    through. Otherwise: extract each input RAW → (single-input: no name prefix; multi/labelled:
    'label:chan') → concat → optional zscore_joint → ONE block-level f_apply_block_pca. Single-input blocks
    inherit ALL source metadata (n_per_side / pca_pix / params / …) so downstream aliases stay intact.
    origin='derived' inputs reference already-built blocks in `prior` (from_block) — no extraction."""
    if 'proc' in block:                                     # NEW proc-based design (f_proc step-lists)
        return _build_block_proc(name, block, origin, shared, verbose, prior=prior)
    if origin == 'cache':
        bb = shared['cache']['built_blocks']
        if name not in bb:
            raise ValueError(f'cache lacks block {name!r}; has {list(bb)}')
        out = dict(bb[name])
        out['inputs'] = block.get('inputs')
        _cap_loaded_block(out, block.get('n_pca'), name, verbose)   # cap PCs to the plan's n_pca (upper bound)
        return out
    inputs = block['inputs']
    single = len(inputs) == 1
    # Per-input NORMALIZATION (multi-input only): put each input on a comparable footing before concat.
    # block['input_norm']:
    #   None          → raw (a huge-variance block dominates the joint PCA).
    #   'total_var'   → ÷ sqrt(Σ col-var): each input contributes EQUAL total variance (scalar).
    #   'first_sv'    → ÷ top singular value (MFA): no input's LEADING PC dominates (scalar).
    #   'standardize' → (X − global mean)/global std per input (scalar center + scale).
    #   'pca'         → PCA each input SEPARATELY, keep `n_init` comps, then concat + block PCA
    #                   (two-level). `pca_whiten` (default True): True → scores ÷ √eigenvalue = UNIT
    #                   variance (eigenvalue scaling REMOVED → equal weighting, block PCA finds
    #                   cross-input correlations); False → scores keep NATURAL variance = eigenvalues.
    #                   `n_init` = comps per input.
    # total_var/first_sv/standardize use SINGLE scalars → within-block structure preserved.
    norm = None if single else block.get('input_norm')
    _whiten = bool(block.get('pca_whiten', True))
    Xs, ni_list, metas = [], [], []

    if norm == 'pca':
        # STREAMED: extract → per-input PCA (n_init) → free the raw → next input, so the within-block
        # RAM peak is a SINGLE input's raw, not the sum over inputs. Identical result to extract-all-
        # then-PCA (same per-input PCA(random_state=42), same concat + block PCA afterwards).
        from sklearn.decomposition import PCA
        n_init = block.get('n_init')
        if verbose:
            print(f"  [combine {name!r}] per-input PCA (streamed, input_norm=pca, whiten={_whiten}):")
        for i, inp in enumerate(inputs):
            Xi, _, meta = _extract_input(inp, name, i, origin, shared, verbose, prior=prior)
            Xi = _input_transform(Xi, inp.get('transform'))
            lab = inp.get('label') or _input_label(inp)
            k = int(min(n_init if n_init else 10 ** 9, *np.asarray(Xi).shape))
            if verbose:
                dg = _input_scale_rank(Xi)
                print(f"    {lab:16s} shape={tuple(np.asarray(Xi).shape)}  mean={dg['mean']:.4g} "
                      f"std={dg['std']:.4g}  totvar={dg['total_var']:.4g}  eff_rank={dg['eff_rank']:.1f}  "
                      f"hard_rank={dg['hard_rank']}  ->pca({k}, whiten={_whiten})")
            Xf = np.nan_to_num(np.asarray(Xi, float)); del Xi   # one raw copy in flight at a time
            p = PCA(n_components=k, whiten=_whiten, random_state=42).fit(Xf)
            Xs.append(p.transform(Xf).astype(np.float32)); del Xf   # free raw before the next input
            ni_list.append([f'{lab}:ipc{j}' for j in range(k)])
            metas.append(meta)
    else:
        # extract every input RAW (scalar / no norm don't reduce dimensionality → all raw is held
        # anyway), then optional per-input scalar normalization before concat.
        for i, inp in enumerate(inputs):
            Xi, ni, meta = _extract_input(inp, name, i, origin, shared, verbose, prior=prior)
            Xi = _input_transform(Xi, inp.get('transform'))     # e.g. meng energy → signed_sqrt amplitude
            if not (single and not inp.get('label')):           # single-input, no label → NO prefix
                lab = inp.get('label') or _input_label(inp)
                ni = [f'{lab}:{c}' for c in ni]
            Xs.append(np.asarray(Xi)); ni_list.append(ni); metas.append(meta)

        shifts = [0.0] * len(Xs)                            # per-input mean subtracted ('standardize')
        scales = [1.0] * len(Xs)                            # per-input divisor (scalar norms)
        if norm:
            for i, Xi in enumerate(Xs):
                Xf = np.asarray(Xi, float)
                if norm in ('total_var', 'std', 'rms'):
                    v = float(np.nansum(np.nanvar(Xf, axis=0)))
                    scales[i] = float(np.sqrt(v)) if v > 0 else 1.0
                elif norm == 'first_sv':
                    fev = _input_scale_rank(Xi).get('first_ev', 0.0)
                    scales[i] = float(np.sqrt(fev)) if fev > 0 else 1.0
                elif norm == 'standardize':
                    shifts[i] = float(np.nanmean(Xf))
                    sd = float(np.nanstd(Xf))
                    scales[i] = sd if sd > 0 else 1.0
                else:
                    raise ValueError(f"unknown input_norm {norm!r} "
                                     "(use 'total_var'|'first_sv'|'standardize'|'pca'|None)")
        if not single and verbose:                          # analyze each input BEFORE concatenation
            print(f"  [combine {name!r}] per-input scale/rank (pre-concat, input_norm={norm}):")
            for inp, Xi, sh, sc in zip(inputs, Xs, shifts, scales):
                lab = inp.get('label') or _input_label(inp)
                dg = _input_scale_rank(Xi)
                _adj = f"scale/={sc:.4g}" + (f" mean-={sh:.3g}" if sh else "")
                print(f"    {lab:16s} shape={tuple(np.asarray(Xi).shape)}  mean={dg['mean']:.4g} "
                      f"std={dg['std']:.4g}  totvar={dg['total_var']:.4g}  eff_rank={dg['eff_rank']:.1f}  "
                      f"hard_rank={dg['hard_rank']}  {_adj}")
        if norm:                                            # apply (X − shift)/scale (float32-safe)
            _new = []
            for Xi, sh, sc in zip(Xs, shifts, scales):
                if sh == 0.0 and sc == 1.0:
                    _new.append(Xi); continue
                d = np.asarray(Xi).dtype
                Y = Xi if sh == 0.0 else (Xi - np.asarray(sh, dtype=d))
                _new.append(Y * np.asarray(1.0 / sc, dtype=d))
            Xs = _new

    names_all = [n for ni in ni_list for n in ni]           # flatten (post-norm, so 'pca' → ipc names)

    X_cat = Xs[0] if single else np.concatenate(Xs, axis=1)
    if block.get('zscore_joint'):
        X_cat = _zscore_cols(X_cat)
    if not single and verbose:                              # shape/rank of the ACTUAL matrix entering the PCA
        dc = _input_scale_rank(X_cat)
        print(f"    -> pre-PCA concat shape={tuple(X_cat.shape)}  mean={dc['mean']:.4g} "
              f"std={dc['std']:.4g}  totvar={dc['total_var']:.4g}  eff_rank={dc['eff_rank']:.1f}  "
              f"hard_rank={dc['hard_rank']}  (n_pca={block.get('n_pca')} zscore_joint={bool(block.get('zscore_joint'))})")
    raw_names, raw_dim = list(names_all), int(X_cat.shape[1])
    Xp, pns, pca, pg = f_apply_block_pca(X_cat, raw_names, name, block.get('n_pca'))
    if single and metas[0] is not None:                     # inherit source metadata; refresh PCA fields
        out = dict(metas[0])
        out.update({'X': Xp, 'names': pns, 'pca': pca, 'pca_groups': pg,
                    'raw_names': raw_names, 'raw_dim': raw_dim, 'inputs': inputs})
        out.setdefault('kind', inputs[0]['source'])
    else:
        out = {'X': Xp, 'names': pns, 'pca': pca, 'pca_groups': pg, 'raw_names': raw_names,
               'raw_dim': raw_dim, 'kind': name, 'inputs': inputs, 'params': {}}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The resolver
# ─────────────────────────────────────────────────────────────────────────────

def _trunc_pca(p, k):
    """Copy a fitted sklearn PCA truncated to its first k (variance-sorted) components (no re-fit)."""
    p2 = copy.copy(p)
    p2.components_ = p.components_[:k]
    p2.explained_variance_ = p.explained_variance_[:k]
    p2.explained_variance_ratio_ = p.explained_variance_ratio_[:k]
    if getattr(p, 'singular_values_', None) is not None:
        p2.singular_values_ = p.singular_values_[:k]
    p2.n_components_ = k
    return p2


def _cap_loaded_block(out, n_pca, name='', verbose=False):
    """Cap a LOADED PCA block to at most `n_pca` components, keeping the top (variance-sorted) PCs — so
    the plan's n_pca is an UPPER BOUND on a cached block: fewer saved → take all; more → truncate to
    n_pca. Only a scalar int n_pca caps (dict/float/tuple/None → keep the saved block as-is). Grouped
    (pcasep) blocks cap EACH group; single-PCA (pcacomb) blocks cap the whole block. Truncates X /
    names / pca / pca_groups consistently so whiten / inverse / the derived-composite path stay valid.
    Mutates `out` in place."""
    if not isinstance(n_pca, (int, np.integer)) or int(n_pca) <= 0:
        return out
    cap = int(n_pca)
    X = np.asarray(out.get('X')) if out.get('X') is not None else None
    if X is None or X.ndim != 2:
        return out
    pca, pg = out.get('pca'), out.get('pca_groups')
    if isinstance(pca, dict) and pg:                          # grouped: cap each group's PCs
        parts, names, new_pg, new_pca = [], [], {}, {}
        old_names = list(out.get('names') or [])
        n_cut = 0
        for pre in sorted(pg, key=lambda p: pg[p]['xslice'][0]):
            s0, s1 = pg[pre]['xslice']; g = s1 - s0
            p_g = pca.get(pre)
            k = min(cap, g) if p_g is not None else g         # only cap PCA'd groups (raw groups untouched)
            s = int(sum(pp.shape[1] for pp in parts))
            parts.append(X[:, s0:s0 + k])
            new_pg[pre] = {'src_cols': pg[pre]['src_cols'], 'xslice': (s, s + k)}
            names += (old_names[s0:s0 + k] if old_names else [f'{pre}_PC{i + 1}' for i in range(k)])
            new_pca[pre] = (_trunc_pca(p_g, k) if (p_g is not None and k < p_g.components_.shape[0]) else p_g)
            n_cut += g - k
        out['X'] = np.concatenate(parts, axis=1).astype(np.float32)
        out['names'], out['pca'], out['pca_groups'] = names, new_pca, new_pg
        if verbose and n_cut:
            print(f'  [cap {name!r}] grouped -> {out["X"].shape[1]} PCs (<= {cap}/group; dropped {n_cut})')
    elif pca is not None:                                     # single PCA: keep first cap PCs
        k = min(cap, X.shape[1])
        if k < X.shape[1]:
            out['X'] = np.asarray(X[:, :k], np.float32)
            if out.get('names'):
                out['names'] = list(out['names'])[:k]
            if k < pca.components_.shape[0]:
                out['pca'] = _trunc_pca(pca, k)
            if verbose:
                print(f'  [cap {name!r}] {X.shape[1]} -> {k} PCs (n_pca={cap})')
    return out


def _whiten_block(block):
    """Whiten a loaded block's PC scores to unit variance IN PLACE (scores ÷ √eigenvalue — matches
    sklearn PCA(whiten=True)), using the block's saved fitted `pca`. Handles a single PCA or a grouped
    `{prefix: PCA}` dict via `pca_groups` xslices. No-op if the block has no fitted PCA or is already
    whitened (tags `block['whitened']=True`). Lets feature-gen save UNWHITENED scores and toggle
    whitening at load."""
    if block.get('whitened') or block.get('pca') is None:
        return block
    X = np.asarray(block['X'], float)
    pca = block['pca']
    if isinstance(pca, dict):                                # grouped PCA → whiten per group slice
        pg = block.get('pca_groups') or {}
        for pre, p in pca.items():
            ev = getattr(p, 'explained_variance_', None)
            if p is None or ev is None or pre not in pg:
                continue
            s0, s1 = pg[pre]['xslice']
            X[:, s0:s1] = X[:, s0:s1] / np.sqrt(np.where(ev > 0, ev, 1.0))[: s1 - s0]
    else:
        ev = getattr(pca, 'explained_variance_', None)
        if ev is not None:
            X = X / np.sqrt(np.where(ev > 0, ev, 1.0))[: X.shape[1]]
    block['X'] = X.astype(np.float32)
    block['whitened'] = True
    return block


def f_combine_loaded_pca(built_blocks, groups, default_n_pca=500, default_balance=True, verbose=True):
    """Build joint composite blocks from ALREADY-LOADED per-input PCA blocks WITHOUT re-convolving.

    Each cached block stores UNWHITENED PC scores = an orthonormal projection of its raw features, so
    concatenating the scores and re-PCAing is mathematically EQUIVALENT (same retained variance and
    components) to reconstructing raw then re-PCAing — but far cheaper and needs no render/conv. Use
    this instead of a build 'gaborS+mengS' block when its inputs are the SAME already-loaded gaborS /
    mengS caches, just combined + re-PCA'd.

    `groups` accepts EITHER:
      - a plain LIST of composite names, e.g. ['gaborS+mengS'] — each is split on '+' to find its
        source blocks (must already be loaded), combined with default_n_pca / default_balance. This is
        the simple common case: one line, no dict.
      - a DICT {new_name: {'blocks':[…] (optional; default = name.split('+')), 'n_pca', 'balance',
        'zscore'}} — for per-composite overrides.
    `balance=True` whitens each source block's scores to unit per-input variance before the joint PCA
    (= a build composite's input_norm='pca'); else natural (eigenvalue) variance. `zscore` z-scores the
    concatenated columns first. Adds each block via f_combine_blocks_pca (single joint PCA)."""
    from f_visual_features import f_combine_blocks_pca
    if isinstance(groups, (list, tuple)):                     # simple list of names → default-spec dict
        groups = {nm: {} for nm in groups}
    for new_name, spec in groups.items():
        blocks = list(spec.get('blocks') or new_name.split('+'))    # default: split the name on '+'
        spec = {'n_pca': default_n_pca, 'balance': default_balance, **spec}
        missing = [b for b in blocks if b not in built_blocks]
        if missing:
            print(f"  [combine {new_name!r}] SKIP — missing loaded blocks {missing}")
            continue
        srcs = blocks
        if spec.get('balance'):                              # whiten temp copies → equal per-input var
            srcs = []
            for b in blocks:
                blk = dict(built_blocks[b])
                blk['X'] = np.asarray(blk['X'], float).copy()
                _whiten_block(blk)                           # no-op if already whitened at load
                key = f'__bal__{b}'
                built_blocks[key] = blk
                srcs.append(key)
        f_combine_blocks_pca(built_blocks, srcs, spec.get('n_pca', 500), name=new_name,
                             zscore=bool(spec.get('zscore', False)))
        built_blocks[new_name]['source_blocks'] = blocks     # report the ORIGINAL block names
        for k in [s for s in srcs if s.startswith('__bal__')]:
            built_blocks.pop(k, None)                         # drop temp balanced copies
        if verbose:
            bb = built_blocks[new_name]
            print(f"  [combine {new_name!r}] {blocks} -> X{tuple(np.asarray(bb['X']).shape)} "
                  f"(raw_dim={bb['raw_dim']}, balance={bool(spec.get('balance'))}, "
                  f"zscore={bool(spec.get('zscore'))}) - no reconvolution")
    return built_blocks


def f_resolve_features(session, plan, build_cfg=None, *, features_dir=None, mouse_id=None,
                       dset_name=None, only=None, cache_name=None, cache_tag=None, cache_idx=0,
                       conv_tag=None, save=False, save_tag='v1', save_movie=False,
                       want_movie=False, whiten=False, prerendered=None,
                       render_verbose=False, verbose=True):
    """Resolve a per-block feature `plan` → (built_blocks, frame_t, side_tags, movie).

    Groups blocks by (auto-resolved) origin and runs each shared op ONCE: all 'build' blocks share
    one render; all 'cache' blocks share one bundle load; all 'repo' blocks share one repo open.
    Every origin returns the standard block-dict schema, so downstream cells are unchanged.

    session     : f_load_session output (needed for 'build' / want_movie; may be None if every
                  block is cache/repo and want_movie is False).
    plan        : per-block dict (see f_default_feature_plan). Passed by value (deep-copied).
    build_cfg   : shared render/neural-side config (see f_default_build_cfg); also gen_config.
    features_dir/mouse_id/dset_name : cache + repo location (dset_name defaults to the session's).
    only        : restrict to these block names (subset load).
    cache_name/cache_tag/cache_idx  : cache selection (priority name > tag > index).
    conv_tag    : conv_repo GEN_TAG (folder <dset>_<conv_tag>) for origin='repo'/'auto'.
    save/save_tag/save_movie        : autosave the merged registry as a cache (build path only).
    want_movie  : ensure `movie` is returned (render if no build/cache movie is available).
    Returns (built_blocks ordered by plan, frame_t, side_tags, movie|None).
    """
    plan = copy.deepcopy(plan)
    build_cfg = dict(build_cfg or {})
    if dset_name is None:
        dset_name = (session.get('dset_name') if session else None) \
            or (session['est1']['dset_name'] if session else None)

    names = list(plan) if only is None else [n for n in plan if n in only]
    if only is not None:
        miss = [n for n in only if n not in plan]
        if miss:
            raise ValueError(f'only={only}: names not in plan ({list(plan)}): {miss}')

    # Pull in from_block dependencies: a 'derived' block reuses other blocks, so those must resolve
    # first (inserted just before the block that needs them). Iterate to closure (chained derives).
    _changed = True
    while _changed:
        _changed = False
        for n in list(names):
            for inp in (plan.get(n, {}).get('inputs') or []):
                fb = inp.get('from_block')
                if fb and fb in plan and fb not in names:
                    names.insert(names.index(n), fb)           # resolve the source before the derived block
                    _changed = True

    # Availability peek (once): what the cache bundle holds + which kinds the repo has, so an 'auto'
    # block resolves to a source that actually has it (esp. gaborS/mengS, absent from a block cache).
    _origins = [plan[n].get('origin', 'auto') for n in names]
    cache_bundle, cache_keys = None, set()
    if any(o in ('cache', 'auto') for o in _origins) and features_dir is not None:
        try:
            _cpath = f_resolve_feature_file(features_dir, dset_name, name=cache_name,
                                            tag=cache_tag, index=cache_idx)
            cache_bundle = f_load_features(_cpath)
            cache_keys = set(cache_bundle['built_blocks'])
        except Exception:
            cache_bundle = None                       # no cache; fine for 'auto', checked for explicit
    repo_kinds = set()
    if any(o in ('repo', 'auto') for o in _origins) and features_dir is not None and conv_tag:
        _rd = f_conv_repo_dir(features_dir, dset_name, conv_tag)
        if _rd.exists():
            try:
                repo_kinds = {e['kind'] for e in f_list_conv_filters(_rd)}
            except Exception:
                repo_kinds = set()

    for n in names:                                       # resolve 'auto' → build/cache/repo/derived per block
        plan[n]['origin'] = _resolve_origin_for(n, plan[n], cache_keys, repo_kinds)
    groups = {'build': [], 'cache': [], 'repo': [], 'derived': []}   # bucket by origin (shared op once each)
    for n in names:
        groups[plan[n]['origin']].append(n)
    # Will the movie be rendered? ONLY if a render-source block (pix/pix_mot/gabor/meng/grid/flow)
    # falls through to BUILD (i.e. isn't in cache/repo), OR want_movie forces it with none cached.
    _render_blocks = [n for n in groups['build']
                      if any(i.get('source') in _RENDER_SOURCES for i in plan[n].get('inputs', []))]
    if verbose:                                           # say plainly what LOADS, BUILDS, and RENDERS
        print(f"  [features] LOAD(cache)={groups['cache']}  LOAD(repo)={groups['repo']}  "
              f"BUILD={groups['build']}  DERIVE(from blocks)={groups['derived']}")
        if _render_blocks:
            print(f"  [features] -> RENDER + convolve WILL RUN for {_render_blocks} "
                  f"(render-source block(s) not in cache/repo)")
        elif groups['build']:
            print(f"  [features] -> building {groups['build']} (cheap, NO render)")
        else:
            print(f"  [features] -> all blocks loaded/derived — NO build, NO render")

    frame_t, side_tags, movie = None, None, None
    want_side_tags = _side_tags_from_build_cfg(build_cfg)
    shared = {}
    if build_cfg.get('conv_autosave') and not build_cfg.get('conv_autosave_root'):
        build_cfg = {**build_cfg, 'conv_autosave_root': features_dir}   # default the store root here,
    #                                                so callers need only set conv_autosave=True

    # Set up each origin's shared resource ONCE (one render / one bundle / one repo open). frame_t /
    # side_tags precedence: cache → build → repo (first non-None wins), matching the prior resolver.
    if groups['cache']:
        if cache_bundle is None:
            raise FileNotFoundError(
                f"origin='cache' for {groups['cache']} but no cache found in {features_dir} "
                f"for {dset_name!r} (tag={cache_tag!r})")
        shared['cache'] = cache_bundle
        frame_t = np.asarray(cache_bundle['frame_t']).ravel()
        side_tags = list(cache_bundle['side_tags'])
        if cache_bundle.get('movie') is not None:
            movie = cache_bundle['movie']
        if verbose:
            _gc = cache_bundle.get('gen_config') or {}
            _mt = cache_bundle.get('meta') or {}
            _prm = {k: _gc.get(k) for k in ('pca_gran', 'n_pca', 'pca_specs') if k in _gc}
            print(f'  [cache] {groups["cache"]} (tag={cache_tag!r})'
                  + (f'  params={_prm}' if _prm else ''))
            # per-block loaded dims (so params/shape are verifiable at a glance)
            for _n in groups['cache']:
                _b = cache_bundle['built_blocks'].get(_n, {})
                if 'X' in _b:
                    print(f'      {_n}: {np.asarray(_b["X"]).shape}'
                          + (f'  raw_dim={_b.get("raw_dim")}' if _b.get('raw_dim') else ''))

    # GRID blocks with origin='build' are built PER UNIT inside _build_grid_block (one conv-unit in RAM at
    # a time; conv_autosave hits need no pixels) with a LAZY movie: f_render_once renders only if a unit is
    # not cached (shapes recorded in <features_dir>/render_shapes.json). The shared build below serves the
    # remaining build blocks (agg / pix / self_mot / legacy) and reuses the same movie if it needs pixels.
    grid_build = [n for n in groups['build'] if 'grid' in plan[n]]
    if grid_build:
        if prerendered is None and any(i.get('source') in _RENDER_SOURCES
                                       for n in grid_build for i in plan[n].get('inputs', [])):
            from f_feature_search import f_render_once
            prerendered = f_render_once(session, build_cfg, verbose=verbose, shape_dir=features_dir)
        shared['grid_build'] = {'session': session, 'build_cfg': build_cfg, 'prerendered': prerendered,
                                'render_verbose': render_verbose}
    if groups['build']:
        built, ft, st, mv = _resolve_build(session, build_cfg,
                                           [n for n in groups['build'] if n not in grid_build], plan,
                                           render_verbose, prerendered=prerendered, verbose=verbose)
        if mv is None and grid_build and prerendered is not None:
            mv = prerendered[2]                               # lazy: pixels render on first use
        shared['build'] = built
        frame_t = ft if frame_t is None else frame_t
        side_tags = st if side_tags is None else side_tags
        if mv is not None:
            movie = mv
        if verbose:
            print(f'  [build] {groups["build"]}')

    if groups['repo']:
        if conv_tag is None:
            raise ValueError("origin='repo' needs conv_tag (the conv_repo GEN_TAG)")
        repo_dir = f_conv_repo_dir(features_dir, dset_name, conv_tag)
        if not repo_dir.exists():
            raise FileNotFoundError(f'conv_repo not found: {repo_dir}')
        repo_ft = np.asarray(f_conv_frame_t(repo_dir)).ravel()
        shared['repo'] = {'repo_dir': repo_dir, 'frame_t': repo_ft,
                          'side_tags': side_tags or want_side_tags}
        frame_t = repo_ft if frame_t is None else frame_t
        side_tags = (side_tags or want_side_tags) if side_tags is None else side_tags
        if verbose:
            print(f'  [repo] {groups["repo"]}')

    # Combine each block from its inputs (single source of truth for concat + block PCA). Non-derived
    # first; then derived blocks (from_block inputs) IN PLAN ORDER, reading the already-built blocks.
    # A grid block (a `grid` key) expands into MANY cell-blocks (name→[<name>:<cell>,…]); all others
    # produce ONE. `name_to_out` tracks each plan name's output block(s) so plan order is preserved.
    built_blocks, name_to_out = {}, {}

    def _emit(n, origin, prior=None):
        if 'grid' in plan[n]:
            d = _build_grid_block(n, plan[n], origin, shared, verbose, prior=prior)
            if verbose:
                print(f"  [grid {n!r}] {len(d)} cell-block(s): {list(d)}")
        else:
            d = {n: _build_block(n, plan[n], origin, shared, verbose, prior=prior)}
        built_blocks.update(d)
        name_to_out[n] = list(d)

    for n in names:
        if plan[n]['origin'] != 'derived':
            _emit(n, plan[n]['origin'])
    for n in names:
        if plan[n]['origin'] == 'derived':
            if verbose:
                _fb = [i.get('from_block') for i in plan[n].get('inputs', []) if i.get('from_block')]
                print(f"  [derive {n!r}] from blocks {_fb} (no reconvolution)")
            _emit(n, 'derived', prior=built_blocks)
    built_blocks = {o: built_blocks[o] for n in names for o in name_to_out[n]}    # restore plan order

    if groups['repo'] and frame_t is not None:                # subset session → truncate repo maps
        n_ref = len(frame_t)
        for n in groups['repo']:                              # map through name_to_out (grid → many cells)
            for o in name_to_out.get(n, [n]):
                if built_blocks[o]['X'].shape[0] > n_ref:
                    built_blocks[o]['X'] = built_blocks[o]['X'][:n_ref]

    if want_movie and movie is None:
        movie = prerendered[2] if prerendered is not None else \
            f_render_session(session, build_cfg, render_verbose=render_verbose)[2]

    for n, b in built_blocks.items():
        if b['X'].shape[0] != len(frame_t):
            raise ValueError(f"block {n!r} has {b['X'].shape[0]} rows but frame_t has "
                             f"{len(frame_t)} — clock mismatch across origins")

    if save and groups['build']:
        path = f_features_path(features_dir, mouse_id, dset_name, save_tag)
        exf, exi = f_pick_frames(movie, 8)
        f_save_features(path, built_blocks, frame_t, side_tags,
                        meta={'mouse_id': mouse_id, 'dset_name': dset_name},
                        gen_config={'build_cfg': build_cfg, 'plan': plan},
                        movie=movie if save_movie else None,
                        example_frames=exf, example_frame_idx=exi)
        if verbose:
            print(f'  [save] {save_tag} → {path}')

    if whiten:                                                # AFTER save → saved caches stay unwhitened
        for n in built_blocks:
            _whiten_block(built_blocks[n])
        if verbose:
            print(f'  [whiten] unit-variance PC scores for {[n for n in built_blocks]}')

    return built_blocks, frame_t, side_tags, movie
