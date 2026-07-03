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

from f_pipeline import f_vectorize_session, f_build_features


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
                'max_frames': rc['max_frames'], 'render_terrain': rc['render_terrain'],
                'cam_params': cam_params, 'build_visual': True, 'visual_specs': vs}
    return f_build_features(session, flat_cfg, render_verbose=False)


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
