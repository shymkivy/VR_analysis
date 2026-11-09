# -*- coding: utf-8 -*-
"""
f_pipeline.py — cross-script orchestration glue (TODO #66 follow-up).

The "session prelude" — load calcium + behavior for one dataset, pick est1, and run
the CHEAP behavior/monitor vectorization (mov_data / lr_data / vec_data_l/r) — is
identical at the top of VR_ca_cebra.py / VR_ca_encoding.py / VR_ca_dimred.py and the
feature-inspection script. `f_load_session` factors it into one call so the scripts
stop copy-pasting it. It deliberately does NOT render or build feature blocks (that's
the exploratory part, kept inline / in f_build_features later) and does NOT touch the
neural target (each script smooths/sorts S its own way).

Created 2026-06-26.
"""

import numpy as np
import pandas as pd

from f_utils import f_load_caim_data
from f_functions import (f_load_bh_data_all, f_proc_movement, f_proc_lick_rew,
                         f_comp_FOV_adj, f_add_phase, f_get_monitor_coords,
                         f_get_monitor_coords_both)
from f_feature_cache import (f_save_features, f_load_features, f_features_path,
                             f_list_feature_files, f_resolve_feature_file, f_pick_frames)

# Rig defaults — single source of truth (was copy-pasted in each script).
DEFAULT_CAM_PARAMS = {
    'aspect':           16 / 9,        # 1920/1080
    'FOV_axis':         'horizontal',  # Unity sets horizontal; vertical derived via f_comp_FOV_adj
    'FOV_deg':          98.4,          # → hFOV 98.4°, vFOV 66.18° (measured rig)
    'cam_rotation_deg': 49.2,
    'clip_len':         60,
    'num_mon':          2,
    'cam_height':       0.6,           # Unity rb→camera-eye Y offset for this rig
}
DEFAULT_CAIM_KW = {
    'caiman_tag':     'results_cnmf.hdf5',
    'cuts_tag':       'h5cutsinfo',
    'frame_data_tag': 'framedata',
    'r_values_min':   0.5,
    'min_SNR':        1.5,
    'thresh_cnn_min': 0.8,
}


def f_vectorize_session(est1, bh, cam_params, interp_step=0.1):
    """Cheap behavior/monitor vectorization for ONE dataset — no calcium/behavior
    LOADING. Pass est1, the single dataset's bh dict (bh_data[n_dset]) and an
    FOV-adjusted cam_params (has cam_rotation_rad). Returns the mov_data / lr_data /
    vec_data_l/r / clock dict that f_build_features consumes. Factored out of
    f_load_session so the VR_feature_gen.py batch (load once, vectorize per dataset)
    can reuse it without reloading calcium."""
    mov_data = f_proc_movement(bh, frame_times=est1['frame_times'], do_interp=1,
                               interp_step=interp_step, plot_stuff=False)
    lr_data = f_proc_lick_rew(bh, mov_data, frame_times=est1['frame_times'],
                              plot_stuff=False)
    # Eye-shifted copy so object angles come from the camera position (not the rb
    # anchor): without this the vertical angles are off by ~arctan(cam_height/dist).
    mov_data_eye = dict(mov_data)
    mov_data_eye['y_pos'] = mov_data['y_pos'] + cam_params['cam_height']
    mov_data_eye['xyz'] = np.column_stack([mov_data['x_pos'], mov_data_eye['y_pos'],
                                           mov_data['z_pos']])
    phi, theta = mov_data_eye['phi'], mov_data_eye['theta']
    mon_r_phi = f_add_phase(phi,  cam_params['cam_rotation_rad'])
    mon_l_phi = f_add_phase(phi, -cam_params['cam_rotation_rad'])
    # Both monitors share the monitor-independent spherical conversion (computed once, not per side).
    vec_data_l, vec_data_r = f_get_monitor_coords_both(
        mov_data_eye, mon_l_phi, mon_r_phi, theta, bh, cam_params, remove_outside_objects=True)
    # Clock: mov_data is already on the imaging clock when delay_corrected → pulse_delay=0.
    pulse_delay = 0.0 if mov_data.get('delay_corrected') else bh['bh_pulse_delay']
    beh_t = mov_data['time']
    frame_t = np.asarray(est1['frame_times']).ravel()
    return {'mov_data': mov_data, 'lr_data': lr_data, 'mov_data_eye': mov_data_eye,
            'theta': theta, 'mon_l_phi': mon_l_phi, 'mon_r_phi': mon_r_phi,
            'vec_data_l': vec_data_l, 'vec_data_r': vec_data_r,
            'pulse_delay': pulse_delay, 'beh_t': beh_t,
            'frame_t': frame_t, 'T_img': frame_t.shape[0]}


def f_load_session(mouse_id, n_dset, data_dir, data_dir_bh, vr_data_xlsx,
                   cam_params=None, caim_kw=None, interp_step=0.1,
                   behavior=True, load_all=False, verbose=True):
    """Load one dataset + (optionally) its cheap behavior/monitor vectorization.

    Parameters
    ----------
    mouse_id : str
    n_dset : int                 index into the do_proc==1 rows for this mouse.
    data_dir : str               calcium proc dir (e.g. 'F:/VR/data_proc/L').
    data_dir_bh : str            behavior base dir; params at
                                 <data_dir_bh><mouse_id>_params.xlsx.
    vr_data_xlsx : str           VR_data.xlsx path (dset_name / bh_dset_name table).
    cam_params : dict | None     None → DEFAULT_CAM_PARAMS. f_comp_FOV_adj is applied
                                 (adds cam_rotation_rad etc.) if not already present.
    caim_kw : dict | None        None → DEFAULT_CAIM_KW (f_load_caim_data filters).
    interp_step : float          behavior interp step for f_proc_movement (s).
    behavior : bool              True (default) loads behavior + runs the monitor
                                 vectorization. False → calcium only (for the
                                 unsupervised neural sweep): bh_data / mov_data /
                                 lr_data / mov_data_eye / theta / mon_l/r_phi /
                                 vec_data_l/r / pulse_delay / beh_t come back None.
    load_all : bool              False (default) loads ONLY the n_dset dataset (fast) →
                                 data_ca / bh_data are 1-element lists. True loads every
                                 do_proc dset (slow). Either way session['est1'] /
                                 session['bh'] are the SELECTED dataset — use those,
                                 not data_ca[n_dset], downstream.

    Returns
    -------
    dict session with keys:
        mouse_id, n_dset, data_ca, bh_data, est1, dset_name, cam_params,
        mov_data, lr_data, mov_data_eye, theta, mon_l_phi, mon_r_phi,
        vec_data_l, vec_data_r, pulse_delay, beh_t, frame_t, T_img.
    Note: NO neural smoothing/sort and NO render — those stay per-script. data_ca is a
    1-element list unless load_all=True (est1 = data_ca[0] then).
    """
    caim_kw = {**DEFAULT_CAIM_KW, **(caim_kw or {})}
    cam_params = dict(cam_params if cam_params is not None else DEFAULT_CAM_PARAMS)
    if 'cam_rotation_rad' not in cam_params:
        cam_params = f_comp_FOV_adj(cam_params)

    # Dataset table.
    vr_data = pd.read_excel(vr_data_xlsx)
    vr_data2 = vr_data.loc[(vr_data.mouse_id == mouse_id)
                           & (vr_data.do_proc == 1)].reset_index(drop=True)
    _n_avail = len(vr_data2)
    _dset_name = str(vr_data2.dset_name.iloc[n_dset])
    if verbose:
        print(f'loading session {mouse_id} dset {n_dset + 1}/{_n_avail}: '
              f'{_dset_name}  (behavior={behavior}, load_all={load_all}) …', flush=True)

    # Load ONLY the selected dataset by default (fast); load_all=True loads every do_proc dset.
    if load_all:
        ca_names = list(vr_data2.dset_name)
        bh_names = list(vr_data2.bh_dset_name)
        _sel = n_dset                               # index of the selected dset within the loaded list
    else:
        ca_names = [_dset_name]                      # single calcium dataset
        bh_names = [str(vr_data2.bh_dset_name.iloc[n_dset])]
        _sel = 0
    import time
    _t = time.perf_counter()
    data_ca = f_load_caim_data(data_dir, ca_names, **caim_kw)
    if verbose:
        print(f'  ⏱ calcium load ({len(ca_names)} dset): {time.perf_counter()-_t:.2f}s', flush=True)
    est1 = data_ca[_sel]
    frame_t = np.asarray(est1['frame_times']).ravel()
    T_img = frame_t.shape[0]

    session = {'mouse_id': mouse_id, 'n_dset': n_dset, 'data_ca': data_ca,
               'bh_data': None, 'bh': None, 'est1': est1,
               'dset_name': est1['dset_name'], 'cam_params': cam_params,
               'frame_t': frame_t, 'T_img': T_img}
    # Behavior keys default to None (calcium-only path leaves them None).
    _beh_keys = ('mov_data', 'lr_data', 'mov_data_eye', 'theta', 'mon_l_phi',
                 'mon_r_phi', 'vec_data_l', 'vec_data_r', 'pulse_delay', 'beh_t')
    session.update({k: None for k in _beh_keys})

    if behavior:
        _t = time.perf_counter()
        params_xls = pd.read_excel(data_dir_bh + mouse_id + '_params.xlsx')
        bh_data = f_load_bh_data_all(data_dir_bh + '/' + mouse_id + '/',
                                     bh_names, params_xls, data_ca)
        if verbose:
            print(f'  ⏱ behavior load ({len(bh_names)} dset): {time.perf_counter()-_t:.2f}s',
                  flush=True)
        session['bh_data'] = bh_data
        session['bh'] = bh_data[_sel]
        _t = time.perf_counter()
        session.update(f_vectorize_session(est1, bh_data[_sel], cam_params, interp_step))
        if verbose:
            print(f'  ⏱ vectorize (mov/lr/monitor coords): {time.perf_counter()-_t:.2f}s',
                  flush=True)

    if verbose:
        _bh = (f'delay_corrected={session["mov_data"].get("delay_corrected")}' if behavior
               else 'behavior=False (calcium only)')
        print(f'session {mouse_id} dset{n_dset} {est1["dset_name"]}: '
              f'{est1["S"].shape[0]} cells, T_img={T_img}, {_bh}')

    return session


def f_acquire_features(source, dset_name, build_fn=None, features_dir=None,
                       mouse_id=None, cache_name=None, cache_tag=None, cache_idx=0,
                       save=False, save_tag='v1', save_movie=False,
                       gen_config=None, meta=None, n_example_frames=8, verbose=True):
    """Build-or-load the feature registry for one dataset — the cache junction.

    Owns the source switch, the cache resolve+load, and the autosave; the actual
    (exploratory) render+extract is supplied by the caller as `build_fn`, so the
    build params stay inline per script. Phase 3 will provide a shared `f_build_features`
    you can pass straight in as build_fn.

    Parameters
    ----------
    source : {'build', 'cache'}
    dset_name : str             est1['dset_name'] — names the cache file / scan filter.
    build_fn : callable | None  source='build' → called with no args, must return
                                (built_blocks, frame_t, side_tags, movie). movie may
                                be None (skip raw-movie caching).
    features_dir, mouse_id : str   cache location; filename
                                <mouse_id>_<dset_name>_<save_tag>_features.joblib.
    cache_name / cache_tag / cache_idx : cache selection (priority name > tag > index;
                                see f_resolve_feature_file).
    save : bool                 source='build' → also write the cache.
    save_tag : str              unique tag in the saved filename.
    save_movie : bool           include the raw movie in the saved cache.
    gen_config, meta : dict     provenance stored in the cache on save.

    Returns
    -------
    (built_blocks, frame_t, side_tags, movie). movie is None unless built/cached
    (cache only carries it when save_movie was set at write time).
    """
    if source == 'build':
        if build_fn is None:
            raise ValueError("source='build' requires a build_fn callable")
        built_blocks, frame_t, side_tags, movie = build_fn()
        if save:
            path = f_features_path(features_dir, mouse_id, dset_name, save_tag)
            # a few original frames stashed even when the full movie is skipped.
            exf, exi = f_pick_frames(movie, n_example_frames)
            f_save_features(path, built_blocks, frame_t, side_tags,
                            meta=meta or {}, gen_config=gen_config or {},
                            movie=movie if save_movie else None,
                            example_frames=exf, example_frame_idx=exi)
            if verbose:
                print(f'saved features [{save_tag}] → {path}')
        return built_blocks, frame_t, side_tags, movie

    if source == 'cache':
        avail = f_list_feature_files(features_dir, dset_name)
        if verbose:
            print(f'available caches for {dset_name!r}: {[p.name for p in avail]}')
        path = f_resolve_feature_file(features_dir, dset_name, name=cache_name,
                                      tag=cache_tag, index=cache_idx)
        bundle = f_load_features(path)
        frame_t = np.asarray(bundle['frame_t']).ravel()
        if verbose:
            print(f"loaded {path.name}: {list(bundle['built_blocks'])}  T={frame_t.size}")
        return (bundle['built_blocks'], frame_t, bundle['side_tags'],
                bundle.get('movie'))

    raise ValueError(f"source must be 'build' or 'cache', got {source!r}")


def _terrain_kwargs(cfg, lat_samp, vert_samp, max_frames, render_verbose):
    """Terrain render/occlusion kwargs from a flat cfg — shared by f_render_session (via
    f_add_terrain_to_monitor) and f_object_vis_blocks (via f_render_terrain). Same kwarg
    names on both so the terrain surface used for occlusion matches the composited movie."""
    return dict(lat_samp=lat_samp, vert_samp=vert_samp,
                stride=cfg.get('terrain_stride', 2),
                point_size=cfg.get('terrain_point_size', 1),
                clip_len=cfg.get('terrain_clip_len', None),
                terrain_mode=cfg.get('terrain_mode', 'surface'),
                terrain_shade=cfg.get('terrain_shade', 'none'),
                gray_level=cfg.get('terrain_gray_level', 128),
                terrain_close_iter=cfg.get('terrain_close_iter', 2),
                max_frames=max_frames, verbose=render_verbose)


def f_object_vis_blocks(session, cfg, side_tags, render_verbose=False, verbose=True):
    """Per-object terrain-visible FRACTION block ('obj_vis'). For each requested side, render each
    object's silhouette ALONE (f_angles_to_movie_v2 return_per_object) + the terrain depth surface
    (f_render_terrain), then f_object_visibility → (T, n_obj) coverage-weighted fraction of each
    object NOT occluded by terrain (partial: a hill occludes the object's lower pixels). Rows are on
    the imaging clock (the render is), so no resample. Absent objects → obj_vis_absent (default NaN).
    Returns a built_blocks-style dict {'obj_vis': {...}} (empty if render_terrain is off)."""
    import time
    import numpy as np
    from f_functions import f_angles_to_movie_v2, f_render_terrain, f_object_visibility
    if not cfg.get('render_terrain', False):
        if verbose:
            print('obj_vis: render_terrain is off → skipped (terrain needed for occlusion)')
        return {}
    cam_params = session['cam_params']
    mov_data = session['mov_data']
    obj_size = cfg.get('obj_size', {'x': 5, 'y': 20, 'z': 5, 'height': 2})
    num_samp = cfg.get('num_samp', 101)
    lat_samp = cfg.get('lat_samp') or num_samp
    vert_samp = cfg.get('vert_samp') or num_samp
    max_frames = cfg.get('max_frames', None)
    absent = cfg.get('obj_vis_absent', np.nan)
    bh = session['bh']
    total_n_obj = len(bh['object_data'])
    _tk = _terrain_kwargs(cfg, lat_samp, vert_samp, max_frames, render_verbose)
    mov_data_eye = session['mov_data_eye']
    theta = session['theta']
    mouse_xyz_eye = np.stack([mov_data_eye['x_pos'], mov_data_eye['y_pos'],
                              mov_data_eye['z_pos']], axis=1)
    _phi = {'L': session['mon_l_phi'], 'R': session['mon_r_phi']}
    _vec = {'L': session['vec_data_l'], 'R': session['vec_data_r']}

    _t = time.perf_counter()
    Xs, names = [], []
    for tag in side_tags:
        vec = _vec[tag]
        _obj, mon_per_obj, _depth = f_angles_to_movie_v2(
            vec, mov_data['time'], cam_params, obj_size, lat_samp=lat_samp, vert_samp=vert_samp,
            filled=True, antialias=True, chunk_t=None, return_per_object=True,
            total_n_obj=total_n_obj, return_depth=True, max_frames=max_frames, verbose=render_verbose)
        _terr, terrain_depth = f_render_terrain(
            bh['terrainData'], mouse_xyz_eye, _phi[tag], theta, cam_params,
            return_depth=True, **_tk)
        # obj_dist is in the USED-object space; scatter to the full object index space that
        # mon_per_obj's channels use (obj_dest_idx = vec['obj_used']). Unused channels have zero
        # coverage → visible fraction is `absent` regardless of distance.
        T = mon_per_obj.shape[0]
        obj_dist_full = np.full((T, total_n_obj), np.inf, dtype=np.float32)
        obj_dist_full[:, np.asarray(vec['obj_used'], dtype=int)] = vec['obj_dist'][:T]
        vis = f_object_visibility(mon_per_obj, obj_dist_full, terrain_depth, absent=absent)
        Xs.append(vis.astype(np.float32))
        names.extend([f'objvis_obj{k:02d}_{tag}' for k in range(total_n_obj)])
    X = np.concatenate(Xs, axis=1) if len(Xs) > 1 else Xs[0]
    if verbose:
        print(f'⏱  obj_vis ({total_n_obj} obj × {len(side_tags)} side): {time.perf_counter()-_t:.2f}s')
    return {'obj_vis': {'X': X, 'names': names, 'side_layout': True,
                        'kind': 'obj_vis', 'total_n_obj': total_n_obj}}


def f_render_session(session, cfg, render_verbose=False, verbose=True):
    """Render the L/R monitor movie (objects [+ optional terrain]) for one session —
    the render half of f_build_features, factored out so a cache-only consumer
    (VR_feature_viz.py) can REGENERATE the raw movie when the cache has none.

    session : f_load_session output (needs cam_params, vec_data_l/r, mov_data, bh;
              + mov_data_eye / mon_l/r_phi / theta when cfg['render_terrain']).
    cfg keys: obj_size, num_samp, max_frames, render_terrain +
              terrain_clip_len / terrain_stride / terrain_point_size.

    Returns (left_mon_frames, right_mon_frames, two_mon_frames).
    """
    import time
    from f_functions import f_angles_to_movie_v2, f_add_terrain_to_monitor

    cam_params = session['cam_params']
    vec_data_l, vec_data_r = session['vec_data_l'], session['vec_data_r']
    mov_data = session['mov_data']
    bh = session['bh']
    obj_size = cfg.get('obj_size', {'x': 5, 'y': 20, 'z': 5, 'height': 2})
    num_samp = cfg.get('num_samp', 101)
    # Anisotropic render: lat_samp (width) / vert_samp (height) default to num_samp (square). For
    # angularly-square pixels set lat:vert = hFOV:vFOV (≈1.49 for this rig, NOT 16:9 — the map is
    # linear in angle). Frames come out (T, vert_samp, lat_samp).
    lat_samp = cfg.get('lat_samp') or num_samp
    vert_samp = cfg.get('vert_samp') or num_samp
    max_frames = cfg.get('max_frames', None)

    _t = time.perf_counter()
    left_mon_frames, left_depth = f_angles_to_movie_v2(
        vec_data_l, mov_data['time'], cam_params, obj_size, lat_samp=lat_samp,
        vert_samp=vert_samp, filled=True, antialias=True, chunk_t=None,
        return_depth=True, max_frames=max_frames, verbose=render_verbose)
    right_mon_frames, right_depth = f_angles_to_movie_v2(
        vec_data_r, mov_data['time'], cam_params, obj_size, lat_samp=lat_samp,
        vert_samp=vert_samp, filled=True, antialias=True, chunk_t=None,
        return_depth=True, max_frames=max_frames, verbose=render_verbose)
    if verbose:
        print(f'⏱  render L+R monitors (f_angles_to_movie_v2): {time.perf_counter()-_t:.2f}s')

    if cfg.get('render_terrain', False):
        mov_data_eye = session['mov_data_eye']
        mon_l_phi, mon_r_phi, theta = session['mon_l_phi'], session['mon_r_phi'], session['theta']
        mouse_xyz_eye = np.stack([mov_data_eye['x_pos'], mov_data_eye['y_pos'],
                                  mov_data_eye['z_pos']], axis=1)
        _tk = _terrain_kwargs(cfg, lat_samp, vert_samp, max_frames, render_verbose)
        left_mon_frames = f_add_terrain_to_monitor(
            bh['terrainData'], mouse_xyz_eye, mon_l_phi, theta, cam_params,
            left_mon_frames, left_depth, **_tk)
        right_mon_frames = f_add_terrain_to_monitor(
            bh['terrainData'], mouse_xyz_eye, mon_r_phi, theta, cam_params,
            right_mon_frames, right_depth, **_tk)

    two_mon_frames = np.concatenate((left_mon_frames, right_mon_frames), axis=2)
    return left_mon_frames, right_mon_frames, two_mon_frames


def f_build_features(session, cfg, render_verbose=False, render=True, prerendered=None, verbose=True):
    """Render the monitor movie + build the full feature registry for one session.

    prerendered : (left, right, two_mon_frames) | None. When given (and render=True), REUSE this
        movie instead of calling f_render_session — the render is identical across feature configs,
        so a parameter search renders ONCE and passes it to every build (only the GPU conv re-runs).

    The shared render+extract (phase 3): used as f_acquire_features's build_fn in
    VR_ca_cebra.py / VR_ca_encoding.py and by the VR_feature_gen.py batch loop, so the
    render + build_feature_blocks + build_visual_blocks pipeline lives in ONE place.

    render : True (default) → render the L/R movie, build pix + visual blocks. False →
             render-free cheap path: build ONLY the Unity-derived blocks (agg / motion /
             self_mot / beh / per_obj) straight from session; no render, no pix, no visual
             (movie returned None). Lets f_resolve_features get the cheap blocks without
             paying a full-session render. Every existing caller uses the default (unchanged).

    session : f_load_session output (needs vec_data_l/r, mov_data, lr_data, cam_params,
              bh_data, n_dset, beh_t, frame_t, pulse_delay, T_img; + mov_data_eye /
              mon_l/r_phi / theta when cfg['render_terrain']).
    cfg : flat build config (also stored as gen_config). Keys (all optional, with
          defaults): side, obj_size, num_samp, max_frames, render_terrain +
          terrain_clip_len/terrain_stride/terrain_point_size, agg_mode, obj_collapse,
          nan_absent, n_pix_pca, build_motion/motion_smooth_sigma,
          build_self_motion/self_motion_smooth_sigma/self_motion_decomp,
          build_pix_motion/pix_motion_smooth_sigma,
          build_lick_reward/lick_reward_smooth_sigma, build_per_obj,
          build_visual/visual_specs.
    render_verbose : inner per-frame progress from the renderers.

    Returns (built_blocks, frame_t, side_tags, two_mon_frames).
    """
    import time
    from f_feature_helpers import build_feature_blocks
    from f_visual_features import build_visual_blocks

    cam_params = session['cam_params']
    vec_data_l, vec_data_r = session['vec_data_l'], session['vec_data_r']
    mov_data, lr_data = session['mov_data'], session['lr_data']
    bh = session['bh']           # single dataset's bh dict (bh_data[n_dset])
    beh_t, frame_t, pulse_delay = session['beh_t'], session['frame_t'], session['pulse_delay']
    T_img = session['T_img']

    side = cfg.get('side', 'both')
    obj_size = cfg.get('obj_size', {'x': 5, 'y': 20, 'z': 5, 'height': 2})
    max_frames = cfg.get('max_frames', None)

    if render:
        # Render L/R (objects [+ terrain]) — shared with VR_feature_viz.py. A search passes a
        # pre-rendered movie (identical across configs) so the render runs once, not per config.
        if prerendered is not None:
            left_mon_frames, right_mon_frames, two_mon_frames = prerendered
        else:
            left_mon_frames, right_mon_frames, two_mon_frames = f_render_session(
                session, cfg, render_verbose=render_verbose, verbose=verbose)
        # Side selection → vec_data / side_tags / pix movie / per-side movies.
        _sel = {'L':    ([vec_data_l],             ['L'],      left_mon_frames,  [left_mon_frames]),
                'R':    ([vec_data_r],             ['R'],      right_mon_frames, [right_mon_frames]),
                'both': ([vec_data_l, vec_data_r], ['L', 'R'], two_mon_frames,
                         [left_mon_frames, right_mon_frames])}
        if side not in _sel:
            raise ValueError(f'unknown side: {side!r}')
        vec_data_use, side_tags, pix_frames, mon_movies = _sel[side]
        if pix_frames.shape[0] != T_img:
            raise ValueError(
                f'pix_frames has {pix_frames.shape[0]} frames but behavior has {T_img} '
                f'(max_frames={max_frames}). Set max_frames=None to build features.')
    else:
        # Render-free cheap path: Unity-derived blocks only (no pix / no visual).
        _selnr = {'L':    ([vec_data_l],             ['L']),
                  'R':    ([vec_data_r],             ['R']),
                  'both': ([vec_data_l, vec_data_r], ['L', 'R'])}
        if side not in _selnr:
            raise ValueError(f'unknown side: {side!r}')
        vec_data_use, side_tags = _selnr[side]
        pix_frames, mon_movies, two_mon_frames = None, None, None

    # All feature blocks (agg, pix, motion, per_obj, self_mot, beh) in one call.
    _t = time.perf_counter()
    built_blocks = build_feature_blocks(
        vec_data_use=vec_data_use, side_tags=side_tags,
        beh_t=beh_t, frame_t=frame_t, pulse_delay=pulse_delay,
        agg_mode=cfg.get('agg_mode', 'angular'), obj_size=obj_size,
        clip_len=cam_params['clip_len'],
        obj_collapse=cfg.get('obj_collapse', 'salience_mean'),
        nan_absent=cfg.get('nan_absent', True),
        # needs the movie AND to be requested — a gabor/meng-only build must not pay the pix flatten
        build_pix=(pix_frames is not None and cfg.get('build_pix', True)),
        build_agg=cfg.get('build_agg', True),          # off when no block asked for agg
        pix_frames=pix_frames, n_pix_pca=cfg.get('n_pix_pca', 50),
        pix_pre_proc=cfg.get('pix_pre_proc', None),
        build_motion=cfg.get('build_motion', False),
        motion_smooth_sigma=cfg.get('motion_smooth_sigma', 2.0),
        build_per_obj=cfg.get('build_per_obj', False),
        total_n_obj=(len(bh['object_data'])
                     if cfg.get('build_per_obj', False) else None),
        build_self_motion=cfg.get('build_self_motion', False),
        mov_data=mov_data,
        self_motion_smooth_sigma=cfg.get('self_motion_smooth_sigma', 2.0),
        self_motion_decomp=cfg.get('self_motion_decomp', False),
        build_pix_motion=cfg.get('build_pix_motion', False),
        pix_motion_smooth_sigma=cfg.get('pix_motion_smooth_sigma', 2.0),
        build_lick_reward=cfg.get('build_lick_reward', False),
        lr_data=lr_data,
        lick_reward_smooth_sigma=cfg.get('lick_reward_smooth_sigma', 0.0),
    )
    if verbose:
        print(f'⏱  build_feature_blocks (agg/pix/motion/...): {time.perf_counter()-_t:.2f}s')

    # Visual-cortex-style monitor blocks merged into the SAME registry (render path only).
    if cfg.get('build_visual', False) and mon_movies is not None:
        visual_specs = cfg.get('visual_specs', {})
        _t = time.perf_counter()
        _as_ctx = None
        if cfg.get('conv_autosave'):          # opt-in SSD reuse of the pooled conv blocks
            import f_conv_autosave as _cas
            _as_ctx = _cas.f_autosave_ctx(cfg, session)
        built_blocks.update(build_visual_blocks(
            mon_movies, side_tags, beh_t, frame_t, pulse_delay, visual_specs,
            backend=cfg.get('conv_backend', 'cpu'), device=cfg.get('conv_device'), verbose=verbose,
            autosave=_as_ctx, edge_min_frac=cfg.get('edge_min_frac')))
        if verbose:
            print(f'⏱  build_visual_blocks ({list(visual_specs)}): {time.perf_counter()-_t:.2f}s')
            for _vn in visual_specs:
                _vb = built_blocks[_vn]
                _pca = _vb['pca']
                if isinstance(_pca, dict):
                    _ev = ' ' + ' '.join(f'{g}:EVR{p.explained_variance_ratio_.sum():.0%}'
                                         for g, p in _pca.items() if p is not None)
                elif _pca is not None:
                    _ev = f', PCA→{_vb["X"].shape[1]}'
                else:
                    _ev = ''
                print(f'  visual block {_vn!r}: {_vb["X"].shape}  (raw {_vb["raw_dim"]}d{_ev})')

    # Per-object terrain-visible fraction (render path only; needs render_terrain for occlusion).
    if render and cfg.get('build_obj_vis', False):
        built_blocks.update(f_object_vis_blocks(
            session, cfg, side_tags, render_verbose=render_verbose, verbose=verbose))

    return built_blocks, frame_t, side_tags, two_mon_frames
