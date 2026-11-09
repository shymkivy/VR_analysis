# -*- coding: utf-8 -*-
"""
f_feature_gen.py — batch feature-generation helpers for VR_feature_gen.py (TODO #66).

Compute-only helpers for the batch build+save script: build one dataset's feature
registry (thin wrapper over the shared f_pipeline.f_build_features) and resolve the
cfg['DSETS'] selector to a list of do_proc indices. The batch loop itself stays inline
in VR_feature_gen.py (editable per run).

Created 2026-06-29.
"""

import numpy as np

from f_pipeline import f_vectorize_session, f_build_features, f_render_session


def _terrain_render_keys(rc):
    """Pull the terrain surface/occlusion render keys out of a cfg['render'] sub-dict, with
    defaults, so every GEN_MODE forwards them to f_render_session identically."""
    return {'terrain_mode':       rc.get('terrain_mode', 'surface'),
            'terrain_shade':      rc.get('terrain_shade', 'none'),
            'terrain_stride':     rc.get('terrain_stride', 2),
            'terrain_point_size': rc.get('terrain_point_size', 1),
            'terrain_clip_len':   rc.get('terrain_clip_len'),
            'terrain_gray_level': rc.get('terrain_gray_level', 128),
            'terrain_close_iter': rc.get('terrain_close_iter', 2)}


def f_build_conv_repo_one_dataset(est1, bh, cam_params, cfg):
    """Convolved-REPO mode for one dataset (GEN_MODE='conv_repo'): vectorize → render the L/R
    movie → convolve each filter ONCE → save the dense uint16 maps to the repo (incremental).
    NO block build / PCA. Returns (manifest, n_written, n_skipped). Repo dir =
    f_conv_repo_dir(<features_dir>, dset_name, GEN_TAG); mirror of f_build_one_dataset."""
    from f_conv_repo import f_conv_repo_dir, f_conv_filter_specs, f_conv_filter_specs_phys, f_build_conv_repo
    from f_decoding import f_imaging_fs
    rc, cs = cfg['render'], cfg['conv_specs']
    session = {'est1': est1, 'bh': bh, 'cam_params': cam_params,
               **f_vectorize_session(est1, bh, cam_params, interp_step=rc['interp_step'])}
    frame_t = session['frame_t']
    lat = rc.get('lat_samp') or rc['num_samp']
    vert = rc.get('vert_samp') or rc['num_samp']
    # PHYSICAL conv_specs (gabor sf_cyc_deg / meng tf_hz) → convert to px/frame at the render geometry;
    # legacy px/cyc-frame specs pass through f_conv_filter_specs unchanged.
    _phys = any(isinstance(v, dict) and 'sf_cyc_deg' in v for v in cs.values())
    if _phys:
        specs = f_conv_filter_specs_phys(cs, cam_params, lat, vert, fs=f_imaging_fs(frame_t))
    else:
        specs = f_conv_filter_specs(cs)
    repo_dir = f_conv_repo_dir(cfg['_features_dir'], est1['dset_name'], cfg['GEN_TAG'])
    print(f'[conv_repo] {cfg.get("_mouse_id")}/{est1["dset_name"]} → {repo_dir}')
    print(f'  {len(specs)} filter specs × 2 sides = {len(specs) * 2} maps  '
          f'(units={"physical" if _phys else "pixel"}, overwrite={cfg.get("OVERWRITE", False)})')
    render_cfg = {'obj_size': rc['obj_size'], 'num_samp': rc['num_samp'],
                  'lat_samp': rc.get('lat_samp'), 'vert_samp': rc.get('vert_samp'),
                  'max_frames': rc['max_frames'], 'render_terrain': rc['render_terrain'],
                  'cam_params': cam_params, **_terrain_render_keys(rc)}
    print('  rendering L/R movie ...', flush=True)
    left, right, _ = f_render_session(session, render_cfg)
    side_tags, movies = ['L', 'R'], [left, right]
    print(f'  rendered {frame_t.size} frames ({left.shape[1]}×{left.shape[2]}/side) — '
          f'convolving + saving to repo ...', flush=True)
    meta = {'mouse': cfg.get('_mouse_id'), 'dset_name': est1['dset_name'],
            'gen_tag': cfg['GEN_TAG'], 'num_samp': rc['num_samp']}
    return f_build_conv_repo(movies, side_tags, specs, repo_dir, frame_t,
                             meta=meta, overwrite=cfg.get('OVERWRITE', False),
                             backend=cfg.get('conv_backend', 'cpu'),
                             device=cfg.get('conv_device', 'cuda'))


def f_build_one_dataset(est1, bh, cam_params, cfg):
    """Render + extract all feature blocks for one dataset. Thin wrapper over the
    shared f_pipeline.f_build_features (same code path as VR_ca_cebra.py): vectorize
    the dataset, flatten the nested cfg, build. Returns built_blocks, frame_t,
    side_tags, two_mon_frames."""
    rc, fb, vs = cfg['render'], cfg['feature_blocks'], cfg['visual_specs']
    session = {'est1': est1, 'bh': bh, 'cam_params': cam_params,
               **f_vectorize_session(est1, bh, cam_params, interp_step=rc['interp_step'])}
    flat_cfg = {**fb,
                'obj_size': rc['obj_size'], 'num_samp': rc['num_samp'],
                'lat_samp': rc.get('lat_samp'), 'vert_samp': rc.get('vert_samp'),
                'max_frames': rc['max_frames'], 'render_terrain': rc['render_terrain'],
                **_terrain_render_keys(rc),
                'cam_params': cam_params, 'build_visual': True, 'visual_specs': vs,
                # gabor/meng build convolution backend (blocks mode): 'gpu' (f_conv_gpu) | 'cpu'
                'conv_backend': cfg.get('conv_backend', 'gpu'),
                'conv_device': cfg.get('conv_device')}
    return f_build_features(session, flat_cfg, render_verbose=False)


def _pca_var_report(label, pca, n_pca, raw_dim, n_kept):
    """Print a per-block PCA variance summary: #components kept, cumulative variance explained, and —
    when n_pca is a (frac, cap) tuple — whether the cap was hit (variance short of the target) or the
    target fraction was reached. Grouped (per-filter/orient) pca → per-group var range + #capped."""
    def _limit(k, v):
        if isinstance(n_pca, (tuple, list)) and len(n_pca) == 2:
            frac, cap = float(n_pca[0]), int(n_pca[1])
            return (f' [CAPPED @ {cap}: {v:.1%} < {frac:.0%} target]' if (k >= cap and v < frac - 1e-9)
                    else f' [reached {frac:.0%} target in {k}]')
        return ''
    if pca is None:
        print(f'    PCA {label}: raw {raw_dim}d (no PCA)')
    elif isinstance(pca, dict):
        gs = [(p.n_components_, float(p.explained_variance_ratio_.sum()))
              for p in pca.values() if p is not None]
        if not gs:
            print(f'    PCA {label}: raw {raw_dim}d (no PCA)'); return
        vs = [v for _, v in gs]; tot = sum(k for k, _ in gs)
        ncap = 0
        if isinstance(n_pca, (tuple, list)) and len(n_pca) == 2:
            frac, cap = float(n_pca[0]), int(n_pca[1])
            ncap = sum(1 for k, v in gs if k >= cap and v < frac - 1e-9)
        print(f'    PCA {label}: {len(gs)} groups, {raw_dim}->{n_kept}d ({tot} PCs), '
              f'var {min(vs):.1%}-{max(vs):.1%}' + (f', {ncap}/{len(gs)} hit cap' if ncap else ''))
    else:
        v = float(pca.explained_variance_ratio_.sum())
        print(f'    PCA {label}: {raw_dim}->{n_kept}d, {int(pca.n_components_)} PCs, {v:.1%} var'
              + _limit(int(pca.n_components_), v))


def f_build_pca_feat_one_dataset(est1, bh, cam_params, cfg):
    """PCA-FEATURE mode (GEN_MODE='pca_feat'): render ONCE, build the RAW point-sampled gabor/meng
    blocks from cfg['pca_specs'] (f_resolve_features build path — handles the physical
    sf_cyc_deg→px / tf_hz→cyc-frame conversion + GPU conv), then PCA-compress at TWO granularities and
    save ONLY the small reduced outputs (no dense maps). Two independent knobs:
      cfg['SAVE_COMBINED']  → one PCA per INPUT (whole sampled block: all orientations, L‖R) →
        cache tagged GEN_TAG+'_pcacomb'.
      cfg['SAVE_SEPARATED'] → one PCA per FILTER (orientation×SF×TF, L‖R via f_pca_group_spec) →
        cache tagged GEN_TAG+'_pcasep'.
      cfg['SAVE_RAW']       → the sampled conv blocks WITHOUT PCA → cache tagged GEN_TAG+'_pcaraw'
        (larger — the full point-sampled response; still far below the dense conv_repo).
    n_pca knobs cfg['n_pca_combined'] / cfg['n_pca_separated'] (int | float(0,1) | (frac,cap)). The
    saved PCA blocks are UNWHITENED (whiten is a load-time toggle in f_resolve_features). Mirror of
    f_build_conv_repo_one_dataset. Returns (n_comb, n_sep, n_raw) block counts."""
    from f_feature_source import f_resolve_features, f_default_build_cfg
    from f_visual_features import f_apply_block_pca, f_pca_group_spec
    from f_feature_cache import f_save_features, f_features_path, f_pick_frames
    rc, fb = cfg['render'], cfg.get('feature_blocks', {})
    session = {'est1': est1, 'bh': bh, 'cam_params': cam_params,
               **f_vectorize_session(est1, bh, cam_params, interp_step=rc['interp_step'])}
    dname = est1['dset_name']
    save_movie = bool(cfg.get('SAVE_MOVIE', False))
    build_cfg = f_default_build_cfg(
        side=fb.get('side', 'both'), num_samp=rc['num_samp'],
        lat_samp=rc.get('lat_samp'), vert_samp=rc.get('vert_samp'),
        obj_size=rc['obj_size'], max_frames=rc['max_frames'], render_terrain=rc['render_terrain'],
        conv_backend=cfg.get('conv_backend', 'gpu'), conv_device=cfg.get('conv_device'),
        **_terrain_render_keys(rc))
    build_cfg['cam_params'] = cam_params

    # RAW sampled blocks (n_pca=None) — render + GPU conv + point-sample ONCE for all inputs.
    plan_raw = {name: {'origin': 'build', 'n_pca': None, **spec}
                for name, spec in cfg['pca_specs'].items()}
    if cfg.get('SAVE_PIX'):                              # cache the pix movie-PCA too (reuses the render →
        plan_raw['pix'] = {'origin': 'build', 'n_pca': None,   # no re-render on load). Passthrough below.
                           'inputs': [{'source': 'pix', 'reduce': 'raw',
                                       'n_pix_pca': cfg.get('n_pix_pca', 300)}]}
    built_raw, frame_t, side_tags, movie = f_resolve_features(
        session, plan_raw, build_cfg, dset_name=dname, only=list(plan_raw),
        want_movie=save_movie, render_verbose=False, verbose=True)
    exf, exi = (f_pick_frames(movie, 8) if movie is not None else (None, None))
    meta = {'mouse_id': cfg.get('_mouse_id'), 'dset_name': dname,
            'n_cells': int(est1['S'].shape[0]), 'n_frames': int(frame_t.size)}

    def _pca_and_save(gran, n_pca, tag_suffix):
        blocks = {}
        for name, b in built_raw.items():
            if name not in cfg['pca_specs']:            # non-conv block (pix): already reduced → passthrough
                blocks[name] = {**b}
                continue
            rn = b.get('raw_names') or b['names']
            raw_dim = int(np.asarray(b['X']).shape[1])
            n_spec = n_pca if gran == 'combined' else f_pca_group_spec(rn, gran, n_pca)
            Xp, pns, pca, pg = f_apply_block_pca(np.asarray(b['X']), rn, name, n_spec)
            _pca_var_report(f'{name} [{gran}]', pca, n_pca, raw_dim, Xp.shape[1])
            blocks[name] = {**b, 'X': Xp, 'names': pns, 'pca': pca, 'pca_groups': pg,
                            'raw_names': rn, 'raw_dim': raw_dim}
        path = f_features_path(cfg['_features_dir'], cfg.get('_mouse_id'), dname,
                               cfg['GEN_TAG'] + tag_suffix)
        f_save_features(path, blocks, frame_t, side_tags,
                        meta={**meta, 'pca_gran': gran, 'n_pca': n_pca},
                        gen_config={'pca_specs': cfg['pca_specs'], 'n_pca': n_pca,
                                    'pca_gran': gran, 'build_cfg': build_cfg},
                        movie=(movie if save_movie else None),
                        example_frames=exf, example_frame_idx=exi)
        return len(blocks), path

    n_comb = n_sep = n_raw = 0
    if cfg.get('SAVE_COMBINED', True):
        n_comb, _ = _pca_and_save('combined', cfg.get('n_pca_combined', 500), '_pcacomb')
    if cfg.get('SAVE_SEPARATED', True):
        n_sep, _ = _pca_and_save('filter', cfg.get('n_pca_separated', 50), '_pcasep')
    if cfg.get('SAVE_RAW', False):                       # sampled conv WITHOUT PCA (large; own tag)
        path = f_features_path(cfg['_features_dir'], cfg.get('_mouse_id'), dname,
                               cfg['GEN_TAG'] + '_pcaraw')
        f_save_features(path, built_raw, frame_t, side_tags,
                        meta={**meta, 'pca_gran': 'raw'},
                        gen_config={'pca_specs': cfg['pca_specs'], 'pca_gran': 'raw',
                                    'build_cfg': build_cfg},
                        movie=(movie if save_movie else None),
                        example_frames=exf, example_frame_idx=exi)
        n_raw = len(built_raw)
    return n_comb, n_sep, n_raw


def f_resolve_dsets(spec, vr_data2, mouse_id):
    """Resolve cfg['DSETS'] to a sorted list of do_proc-row indices for one mouse.
    spec: None/'all' = all; dict = per-mouse (look up mouse_id); a bare int/str or a
    list of them — ints = do_proc indices, str = dset_name (exact, else substring)."""
    if isinstance(spec, dict):
        spec = spec.get(mouse_id)
    names = list(vr_data2.dset_name)
    if spec is None or (isinstance(spec, str) and spec == 'all'):
        return list(range(len(names)))
    if isinstance(spec, (int, np.integer, str)):     # bare scalar → wrap in a list
        spec = [spec]
    out = []
    for s in spec:
        if isinstance(s, (int, np.integer)):
            out.append(int(s))
        else:
            hits = [i for i, n in enumerate(names) if str(s) == n] \
                or [i for i, n in enumerate(names) if str(s) in n]
            if not hits:
                raise ValueError(f'{mouse_id}: dataset {s!r} not in do_proc list {names}')
            out.extend(hits)
    return sorted(set(out))
