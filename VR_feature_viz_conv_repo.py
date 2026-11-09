# -*- coding: utf-8 -*-
"""
VR_feature_viz_conv_repo.py — visualize / analyze a CONV_REPO (dense per-filter maps) (TODO #68).

A conv_repo (built by VR_feature_gen GEN_MODE='conv_repo') stores the DENSE convolved gabor /
motion-energy maps once per dataset (one uint16 .npy per side × orientation × wavelength [× TF]
+ manifest.json + frame_t.npy). This script:
  - loads the repo → sampled built_blocks (f_load_conv_blocks) for the per-orientation panel;
  - ACTIVITY diagnostic (f_conv_activity + f_plot_conv_activity): per-filter response magnitude
    over time (frame-mean over monitor space) + one number per filter → which orientations /
    wavelengths / temporal freqs the movie actually drives (how to CHOOSE them);
  - converts the repo → ONE browsable HDF5 (f_conv_repo_to_h5; open in HDFView / ImageJ).

For the reduced-block CACHE reconstruction viz (pix/grid/gabor/meng recon vs original, phase,
quiver) use VR_feature_viz_cache.py. Split out of VR_feature_viz.py on 2026-07-12. Spyder cells.
"""

import sys
from pathlib import Path

path1 = Path.home() / 'Desktop' / 'stuff'
sys.path.append(str(path1 / 'VR' / 'VR_analysis'))
sys.path.append(str(path1 / 'VR' / 'VR_analysis' / 'functions'))

import numpy as np

from f_conv_repo import (f_conv_repo_dir, f_load_conv_blocks, f_conv_repo_meta, f_conv_frame_t,
                         f_conv_activity, f_conv_repo_to_h5, f_conv_repo_retag,
                         f_conv_repo_scan, f_list_conv_filters, _conv_filter, _dequant_u16)
from f_feature_viz import (f_reconstruct_feature_movie, f_plot_movie_frames_compare,
                           f_pick_motion_frames, f_plot_conv_activity, f_block_activity,
                           f_plot_conv_hist, f_plot_movie_frames)

#%% Load a CONV_REPO — dense per-filter maps → sampled built_blocks + repo meta
mouse_id     = 'L'
features_dir = 'F:/VR/features/' + mouse_id
conv_dset    = '11_2_25'      # dataset name (repo subfolder is <conv_dset>_<conv_tag>)
conv_tag     = 'gen_7_11_26'
conv_ds_list = [8]             # sampling stride(s) → sampled blocks (gaborS_ds*/mengS_ds*)
conv_n_pca   = 50              # PCs per sampled block (panel uses inverse-PCA; None = raw)
# LIMIT what's loaded (None = load everything; lists = keep only these):
conv_kinds       = ('gabor',)  # e.g. ('gabor',) gabor-only | ('meng',) meng-only  (note the comma!)
conv_wavelengths = (6, 9, 12, 15, 18)        # e.g. (6,)      keep only these gabor/meng wavelengths (px)
conv_odegs       = (0)        # e.g. (0, 90)   keep only these orientations (degrees)
conv_tfreqs      = (0.025, 0.05, 0.1, 0.15, 0.25)        # e.g. (0.15, 0.25, 0.5)   keep only these meng temporal freqs (cyc/frame)

repo_dir = f_conv_repo_dir(features_dir, conv_dset, conv_tag)
built_blocks, side_tags = f_load_conv_blocks(
    repo_dir, conv_ds_list, n_pca=conv_n_pca, kinds=conv_kinds,
    wavelengths=conv_wavelengths, odegs=conv_odegs, tfreqs=conv_tfreqs)
frame_t  = np.asarray(f_conv_frame_t(repo_dir)).ravel()
_rm      = f_conv_repo_meta(repo_dir)
num_samp = _rm.get('num_samp') or 101
mon_hw   = (num_samp, num_samp)
_ds      = _rm.get('dset_name') or repo_dir.name
print(f'loaded conv_repo {repo_dir.name}: blocks={list(built_blocks)}  side={side_tags}  '
      f'T={frame_t.size}  num_samp={num_samp}')


#%% Conv_repo DIAGNOSTIC — per-filter value HISTOGRAMS + example frames (dynamic-range / outliers)
# Scans the .npy maps (mmap, every diag_frame_ds-th frame), dequantizes → per-filter value histograms
# + a stats table: std / p99.9|·| / absmax / tail (= absmax / median|·|). HIGH tail = heavy outlier
# tail (e.g. un-normalized motion energy) — what dominates a block's variance / PCA. Also shows
# example frames of ONE filter's dense map (spot a constant or outlier frame). Set diag_kinds to
# focus (('meng',) note the comma).
if 1:
    diag_kinds      = ('gabor',)     # which filters to scan (None = all)
    diag_waves      = None          # e.g. (15,) limit wavelengths
    diag_odegs      = None          # e.g. (0, 90) limit orientations (deg)
    diag_tfreqs     = None          # e.g. (0.1,) limit meng temporal freqs
    diag_frame_ds   = 8             # read every Nth frame (I/O speed knob)
    diag_max_panels = 16            # histogram panels (filters)
    _sel = _conv_filter(kinds=diag_kinds, wavelengths=diag_waves, odegs=diag_odegs, tfreqs=diag_tfreqs)
    scan_df, scan_samples = f_conv_repo_scan(repo_dir, select=_sel, frame_ds=diag_frame_ds)
    print(scan_df[['name', 'std', 'p50_abs', 'p999_abs', 'absmax', 'tail']].to_string(index=False))
    f_plot_conv_hist(scan_samples, df=scan_df, logy=True, max_panels=diag_max_panels,
                     suptitle=f'{repo_dir.name} — per-filter value histograms')
    # example frames of the FIRST matching filter's dense map (dequantized; watch frame 0 + outliers)
    _ents = [e for e in f_list_conv_filters(repo_dir) if _sel(e)]
    if _ents:
        _e = _ents[0]
        _q = np.load(repo_dir / _e['file'], mmap_mode='r')          # uint16 (T,H,W)
        _fi = np.linspace(0, _q.shape[0] - 1, 8).astype(int)
        _ex = _dequant_u16(np.asarray(_q[_fi]), _e['vmin'], _e['vmax'])
        f_plot_movie_frames(_ex, frame_idx=list(range(_fi.size)), n=8, cmap='magma',
                            titles=[f't={int(t)}' for t in _fi],
                            suptitle=f"{_e['file']} — example frames (dequantized)")


#%% (util) Convert this conv_repo → ONE browsable HDF5 (open in HDFView / ImageJ / h5py)
# dtype='float32' = dequantized real magnitudes (directly viewable); 'uint16' = half size
# (interpret via each dataset's vmin/vmax attrs). Writes <repo>/conv_maps.h5 (or conv_h5_out).
if 0:
    conv_h5_repo     = repo_dir       # ← or a Path/str to any other repo
    conv_h5_out      = None           # None → <repo>/conv_maps.h5 ; else an explicit .h5 path
    conv_h5_dtype    = 'float32'      # 'float32' (viewable real values) | 'uint16' (half size)
    conv_h5_compress = None           # None | 'gzip' | 'lzf'
    f_conv_repo_to_h5(conv_h5_repo, out_path=conv_h5_out, dtype=conv_h5_dtype,
                      compression=conv_h5_compress)


#%% (util) MIGRATE an OLD repo → lossless tf filename tags + fix index collisions
# Old repos used a 2-decimal tf filename tag that ROUNDED (0.025 → 'tf003' = 0.03) and, on
# incremental builds, reused itf indices (two tfreqs → the same tf channel token → the activity
# plot merged/dropped a freq). Reads already self-heal via f_list_conv_filters, but run this ONCE
# to also rename the .npy files (→ tf0025 etc.) + rewrite the manifest — do it BEFORE any further
# incremental build. Set retag_dry=False to actually rename. Safe/idempotent.
if 0:
    retag_repo = repo_dir       # ← or a Path/str to any other repo
    retag_dry  = True           # True → just print planned renames; False → rename + rewrite manifest
    f_conv_repo_retag(retag_repo, dry_run=retag_dry)


#%% Feature ACTIVITY — response magnitude over time + one number per orientation / SF / TF
# Which orientations / SFs / temporal freqs the movie actually drives → how to CHOOSE them.
#   act_source='blocks' — from the ALREADY-LOADED sampled blocks (f_block_activity): NO disk re-read,
#       honours the load filters, but keyed by SF/TF INDEX (isf/itf) + subsampled space (approx).
#   act_source='repo'   — re-reads the DENSE maps (f_conv_activity): exact full-res + wavelength/tf
#       VALUES + all filters; `act_frame_ds` reads every Nth frame to speed the I/O.
if 1:
    act_source   = 'blocks'         # 'blocks' (reuse loaded, fast) | 'repo' (re-read, exact)
    act_reduce   = 'absmean'        # 'absmean' (magnitude) | 'mean' (signed) | 'rms'
    act_frame_ds = 4                # ('repo' only) read every Nth frame — the I/O speed knob
    if act_source == 'blocks':      # pass repo_dir → real wavelength/tfreq VALUES (not indices)
        act_df, act_ts = f_block_activity(built_blocks, reduce=act_reduce, repo_dir=repo_dir)
    else:
        act_df, act_ts = f_conv_activity(repo_dir, reduce=act_reduce, frame_ds=act_frame_ds)
    print(act_df.groupby(['kind', 'odeg', 'wavelength', 'tfreq'], dropna=False)['scalar']
          .mean().round(4).to_string())
    # gabor: by orientation + wavelength(SF).  meng: by orientation + wavelength(SF) + temporal freq.
    f_plot_conv_activity(act_df, act_ts, by='odeg', kind='gabor',
                         suptitle='gabor — activity by orientation')
    f_plot_conv_activity(act_df, act_ts, by='wavelength', kind='gabor',
                         suptitle='gabor — activity by wavelength (which SFs to keep)')
    f_plot_conv_activity(act_df, act_ts, by='odeg', kind='meng',
                         suptitle='meng — activity by orientation')
    f_plot_conv_activity(act_df, act_ts, by='wavelength', kind='meng',
                         suptitle='meng — activity by wavelength (which SFs to keep)')
    f_plot_conv_activity(act_df, act_ts, by='tfreq', kind='meng',
                         suptitle='meng — activity by temporal freq (which speeds to keep)')


#%% Feature MAPS grid — COLUMNS = frames, ROWS = orientation | SF | TF (pick the row axis)
# Reconstructs the point-sampled block back to image space (grid inferred from names) — one ROW per
# value of `panel_by` (holding the other indices fixed), at high-motion frames as columns. So you
# SEE what each frequency / orientation looks like alongside the activity numbers. A conv_repo has
# no original movie → no comparison row.
if 1:
    import re as _re
    panel_blocks = None         # None → ALL loaded blocks (one figure each); or a list of names
    panel_stat   = 'auto'       # 'auto' (match block kind) | 'gabM' | 'mengopp' | 'gabEv' ...
    panel_by     = 'sf'         # ROW axis: 'orient' | 'sf' (spatial-freq idx) | 'tf' (meng temporal)
    panel_odeg   = 0            # fixed orientation (deg)  when NOT the row axis
    panel_sf     = 0            # fixed spatial-freq index when NOT the row axis
    panel_tf     = 0            # fixed temporal-freq index (meng) when NOT the row axis
    panel_npcs   = None         # inverse-PCA rank; None = all stored
    panel_nfr    = 8
    panel_tf_extra = True       # meng: ALSO emit a temporal-freq panel (rows=tf) at the MIDDLE
                                # spatial freq, in addition to the primary panel_by figure

    from f_conv_repo import f_list_conv_filters       # isf→wavelength, itf→tfreq (ACTUAL values)
    _filts = f_list_conv_filters(repo_dir)
    _wl_of = {(e['kind'], e['isf']): e['wavelength'] for e in _filts}
    _tf_of = {(e['kind'], e['itf']): e.get('tfreq') for e in _filts if e.get('itf') is not None}
    if not built_blocks:
        raise ValueError('no blocks loaded — check the load cell / conv_ds_list.')
    _names = list(built_blocks) if panel_blocks is None else list(panel_blocks)
    for _bn in _names:                                     # one figure per LOADED block
        _blk = built_blocks[_bn]
        _kind = _blk.get('kind', 'gabor')
        _stat = ('mengopp' if _kind == 'meng' else 'gabM') if panel_stat == 'auto' else panel_stat
        _is_meng = _stat.startswith('meng')
        _chan = lambda o, s, t: f'{_stat}_o{o}sf{s}' + (f'tf{t}' if _is_meng else '')
        _pat = _re.compile(rf'{_stat}_o(\d+)sf(\d+)' + (r'tf(\d+)' if _is_meng else ''))
        _recs = set()
        for n in _blk['raw_names']:
            m = _pat.match(n.split('_', 1)[1])
            if m:
                _recs.add((int(m.group(1)), int(m.group(2)),
                           int(m.group(3)) if _is_meng else None))
        # index → ACTUAL frequency labels: orientation already in degrees; sf idx → wavelength (px);
        # tf idx → temporal freq (cyc/frame), mapped via the repo manifest (fallback = index).
        _wl_lab = lambda s: (f'λ{_wl_of[(_kind, s)]:g}px' if (_kind, s) in _wl_of else f'sf{s}')
        _tf_lab = lambda t: (f'{_tf_of[(_kind, t)]:g}c/fr' if (_kind, t) in _tf_of else f'tf{t}')

        def _render(_by, sf_fix=panel_sf, tf_fix=panel_tf, odeg_fix=panel_odeg):
            if _by == 'orient':
                _vals = sorted({o for o, s, t in _recs if s == sf_fix and (t == tf_fix or not _is_meng)})
                _spec, _lab = [(v, sf_fix, tf_fix) for v in _vals], (lambda v: f'{v}°')
            elif _by == 'sf':
                _vals = sorted({s for o, s, t in _recs if o == odeg_fix and (t == tf_fix or not _is_meng)})
                _spec, _lab = [(odeg_fix, v, tf_fix) for v in _vals], _wl_lab
            else:                   # 'tf' (meng only)
                _vals = sorted({t for o, s, t in _recs if o == odeg_fix and s == sf_fix})
                _spec, _lab = [(odeg_fix, sf_fix, v) for v in _vals], _tf_lab
            if not _vals:
                print(f'  skip {_bn}: no {_stat} channels for by={_by} '
                      f'(odeg={odeg_fix}, sf={sf_fix}, tf={tf_fix})')
                return
            _pidx = f_pick_motion_frames(built_blocks, panel_nfr, mode='spread', block=_bn)
            _rows = [f_reconstruct_feature_movie(_blk, _kind, mon_hw, side_tags, n_pcs=panel_npcs,
                                                 channel=_chan(*spec), frame_idx=_pidx) for spec in _spec]
            _labels = [f'{_stat} {_lab(v)}' for v in _vals]
            _fixed = (f'{odeg_fix}° ' if _by != 'orient' else '') \
                     + (f'{_wl_lab(sf_fix)} ' if _by != 'sf' else '') \
                     + (f'{_tf_lab(tf_fix)}' if _is_meng and _by != 'tf' else '')
            f_plot_movie_frames_compare(_rows, _pidx, labels=_labels, cmaps=['gray'] * len(_rows),
                                        suptitle=f'{_bn} — {_stat} by {_by} '
                                                 f'(cols=frames; fixed {_fixed.strip()})')
            print(f'{_bn}: rows ({_by}) {_vals}  frames {_pidx}')

        _by0 = 'sf' if (panel_by == 'tf' and not _is_meng) else panel_by   # tf only for meng
        _render(_by0)
        # meng: ALSO plot across TEMPORAL freq (rows=tf), holding SF at the MIDDLE available value.
        if _is_meng and panel_tf_extra and _by0 != 'tf':
            _sfs = sorted({s for o, s, t in _recs if o == panel_odeg})
            if _sfs:
                _render('tf', sf_fix=_sfs[len(_sfs) // 2])
