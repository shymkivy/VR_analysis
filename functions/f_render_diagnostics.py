# -*- coding: utf-8 -*-
"""
Rendered-frame + terrain coord-system diagnostic helpers for VR_analysis.

Split out from f_functions.py on 2026-05-20 to keep the production
rendering toolkit (f_angles_to_movie_v2, f_render_terrain,
f_terrain_world_coords, f_composite_with_depth, loaders) separate from
the larger set of debug / sanity plots and exploratory diagnostics
that live here. Most callers import these via VR_video_recon.py.

For new diagnostics: add them here. Functions in this module may import
from f_functions but not vice versa (one-way dependency).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.spatial import cKDTree

from f_functions import (f_terrain_world_coords, f_spheric_to_cart,
                          f_add_phase, f_cart_to_spheric_np, f_comp_FOV_adj)


def f_plot_terrain_layout(terrain_data, mov_data, obj_data, **kwargs):
    # Top-down + side-view sanity plot. Verifies chunk_pitch + flip / swap
    # place the heightmap at the same world coords the mouse moves through.
    # `kwargs` forwarded to f_terrain_world_coords (chunk_pitch, chunk_centered,
    # flip_x, flip_z, swap_xz).
    w = f_terrain_world_coords(terrain_data, **kwargs)
    tx, tz, ty = w['tx'], w['tz'], w['ty']
    obj = np.stack([obj_data['ObjLocX'].values,
                    obj_data['ObjLocY'].values,
                    obj_data['ObjLocZ'].values], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    step = max(1, int(np.sqrt(len(tx) / 5000)))
    sub = (np.arange(len(tx)) % step == 0)
    sc_h = axes[0].scatter(tx[sub], tz[sub], c=ty[sub], s=4, cmap='terrain', alpha=0.6)
    plt.colorbar(sc_h, ax=axes[0], label='terrain height (world y)')
    axes[0].plot(mov_data['x_pos'], mov_data['z_pos'], color='red', lw=0.6, alpha=0.7, label='mouse trajectory')
    axes[0].scatter(obj[:, 0], obj[:, 2], facecolors='none', edgecolors='black', s=60, lw=1.5, label='objects')
    axes[0].set(xlabel='world x', ylabel='world z',
                title=f'top-down (terrain stride={step}, n_show={int(sub.sum())})')
    axes[0].set_aspect('equal')
    axes[0].legend(loc='best', fontsize=8)

    z_mid = np.median(tz)
    slab = max(20.0, (tz.max() - tz.min()) * 0.05)
    mid = np.abs(tz - z_mid) < slab
    axes[1].scatter(tx[mid], ty[mid], c='tab:green', s=4, alpha=0.4,
                    label=f'terrain (|z - {z_mid:.0f}| < {slab:.0f})')
    axes[1].plot(mov_data['x_pos'], mov_data['y_pos'], color='red', lw=0.6, alpha=0.7, label='mouse y')
    axes[1].scatter(obj[:, 0], obj[:, 1], facecolors='none', edgecolors='black', s=60, lw=1.5, label='objects')
    axes[1].set(xlabel='world x', ylabel='world y (height)', title='side view (mid-z slice)')
    axes[1].legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f'terrain x: [{tx.min():.1f}, {tx.max():.1f}]   z: [{tz.min():.1f}, {tz.max():.1f}]   y: [{ty.min():.2f}, {ty.max():.2f}]')
    print(f'mouse   x: [{mov_data["x_pos"].min():.1f}, {mov_data["x_pos"].max():.1f}]   z: [{mov_data["z_pos"].min():.1f}, {mov_data["z_pos"].max():.1f}]   y: [{mov_data["y_pos"].min():.2f}, {mov_data["y_pos"].max():.2f}]')
    print(f'objects x: [{obj[:, 0].min():.1f}, {obj[:, 0].max():.1f}]   z: [{obj[:, 2].min():.1f}, {obj[:, 2].max():.1f}]   y: [{obj[:, 1].min():.2f}, {obj[:, 1].max():.2f}]')
    print(f'n terrain samples: {len(tx)}   unique chunks: {terrain_data.groupby(["ChunkPosX","ChunkPosZ"]).ngroups}')
    return {'tx': tx, 'tz': tz, 'ty': ty}


def f_plot_terrain_mouse_alignment(terrain_data, mov_data, **kwargs):
    # Vertical alignment check: for each frame, look up terrain height at the
    # mouse's (x, z) and plot mouse_y vs terrain_y over time. If the mouse
    # walks on terrain at constant offset, delta = mouse_y - terrain_y is
    # flat — its median is a candidate `eye_height` for the renderer.
    # `kwargs` forwarded to f_terrain_world_coords.
    w = f_terrain_world_coords(terrain_data, **kwargs)
    tx, tz, ty, cs_x = w['tx'], w['tz'], w['ty'], w['cs_x']
    tree = cKDTree(np.stack([tx, tz], axis=1))
    mxz = np.stack([np.asarray(mov_data['x_pos']),
                    np.asarray(mov_data['z_pos'])], axis=1)
    dist_xy, idx_nn = tree.query(mxz, k=1)
    ty_at_mouse = ty[idx_nn]
    my = np.asarray(mov_data['y_pos'])
    delta = my - ty_at_mouse

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    t = np.asarray(mov_data['time'])
    axes[0].plot(t, my, color='red',  lw=0.6, label='mouse y')
    axes[0].plot(t, ty_at_mouse, color='tab:green', lw=0.6, alpha=0.8,
                 label='terrain y at mouse (x, z)')
    axes[0].set(xlabel='time (s)', ylabel='world y',
                title="mouse y vs terrain y at the mouse's (x, z)")
    axes[0].legend(loc='best', fontsize=8)
    axes[1].plot(t, delta, color='black', lw=0.6)
    axes[1].axhline(np.median(delta), color='red', lw=0.8, ls='--',
                    label=f'median = {np.median(delta):.2f}')
    axes[1].set(xlabel='time (s)', ylabel='mouse_y - terrain_y_at_mouse',
                title='delta over time — flat = mouse walks on terrain; median is a candidate eye_height')
    axes[1].legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(delta, bins=80, color='black')
    axes[0].axvline(np.median(delta), color='red', ls='--',
                    label=f'median = {np.median(delta):.2f}')
    axes[0].set(xlabel='mouse_y - terrain_y_at_mouse', ylabel='# frames',
                title='delta histogram')
    axes[0].legend(loc='best', fontsize=8)
    axes[1].hist(dist_xy, bins=80, color='tab:gray')
    axes[1].set(xlabel='NN horiz dist (mouse to terrain)', ylabel='# frames',
                title='NN xy distance')
    plt.tight_layout()
    plt.show()

    print(f'delta (mouse_y - terrain_y):  median={np.median(delta):.3f}  '
          f'mean={delta.mean():.3f}  std={delta.std():.3f}  '
          f'p5/p95={np.percentile(delta,5):.2f}/{np.percentile(delta,95):.2f}')
    print(f'NN horiz distance:  median={np.median(dist_xy):.3f}  '
          f'max={dist_xy.max():.3f}  (should be << cell_size={cs_x:.3f})')
    return {'delta': delta, 'dist_xy': dist_xy, 'ty_at_mouse': ty_at_mouse}


def f_plot_terrain_object_alignment(terrain_data, obj_data, obj_size, **kwargs):
    # Object alignment check: terrain y at each (ObjLocX, ObjLocZ) should sit
    # ~obj_size['height'] below ObjLocY. Independent of mouse behavior —
    # pure coordinate-system verification.
    w = f_terrain_world_coords(terrain_data, **kwargs)
    tx, tz, ty = w['tx'], w['tz'], w['ty']
    ox = obj_data['ObjLocX'].values
    oy = obj_data['ObjLocY'].values
    oz = obj_data['ObjLocZ'].values
    tree = cKDTree(np.stack([tx, tz], axis=1))
    dist_obj, idx_obj = tree.query(np.stack([ox, oz], axis=1), k=1)
    ty_at_obj = ty[idx_obj]
    delta_obj = oy - ty_at_obj
    expected = obj_size['height']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    step = max(1, int(np.sqrt(len(tx) / 5000)))
    sub = (np.arange(len(tx)) % step == 0)
    axes[0].scatter(tx[sub], tz[sub], c='lightgray', s=2, alpha=0.4)
    sc_h = axes[0].scatter(ox, oz, c=delta_obj, cmap='coolwarm',
                            vmin=expected - 5, vmax=expected + 5, s=80,
                            edgecolors='black', lw=0.7)
    plt.colorbar(sc_h, ax=axes[0], label=f'obj_y - terrain_y  (expected ≈ {expected})')
    axes[0].set(xlabel='world x', ylabel='world z',
                title='objects colored by delta')
    axes[0].set_aspect('equal')
    axes[1].hist(delta_obj, bins=30, color='black')
    axes[1].axvline(expected, color='tab:green', ls='--',
                    label=f"expected = obj_size['height'] = {expected}")
    axes[1].axvline(np.median(delta_obj), color='red', ls='--',
                    label=f'median = {np.median(delta_obj):.2f}')
    axes[1].set(xlabel='obj_y - terrain_y_at_obj_xz', ylabel='# objects',
                title='delta distribution across objects')
    axes[1].legend(loc='best', fontsize=8)
    axes[2].hist(dist_obj, bins=30, color='tab:gray')
    axes[2].set(xlabel='NN horiz dist (obj to nearest terrain)', ylabel='# objects',
                title='lookup quality')
    plt.tight_layout()
    plt.show()

    print(f'object delta:  median={np.median(delta_obj):.3f}  '
          f'mean={delta_obj.mean():.3f}  std={delta_obj.std():.3f}  '
          f'p5/p95={np.percentile(delta_obj,5):.2f}/{np.percentile(delta_obj,95):.2f}')
    print(f"expected ≈ obj_size['height'] = {expected}")
    print(f'n objects: {len(delta_obj)}   NN horiz dist median={np.median(dist_obj):.3f}  '
          f'max={dist_obj.max():.3f}')
    return {'delta_obj': delta_obj, 'dist_obj': dist_obj}


def f_terrain_coord_diagnostic(terrain_data, obj_data, obj_size,
                                chunk_pitch=122.0, chunk_centered=True,
                                cell_size=1.0):
    # Brute-force the 8 (flip_x, flip_z, swap_xz) combos and report
    # object-alignment delta stats. Combo with smallest std + |median-expected|
    # wins. Used to confirm orientation when adapting to a new VR setup.
    # cell_size : passed through to f_terrain_world_coords (default 1.0 —
    #     verify with the chunk-overlap diagnostic before trusting).
    ox = obj_data['ObjLocX'].values
    oy = obj_data['ObjLocY'].values
    oz = obj_data['ObjLocZ'].values
    expected = obj_size['height']

    print(f"brute-force coord diagnostic — expected delta ≈ obj_size['height'] = {expected}")
    print(f"{'flip_x':>7} {'flip_z':>7} {'swap_xz':>8} | "
          f"{'med':>7} {'std':>6} {'p5':>7} {'p95':>7} {'|med-exp|':>10}")
    print('-' * 70)
    best = None
    for swap_xz in (False, True):
        for flip_x in (False, True):
            for flip_z in (False, True):
                w = f_terrain_world_coords(terrain_data,
                                            chunk_pitch=chunk_pitch,
                                            chunk_centered=chunk_centered,
                                            flip_x=flip_x, flip_z=flip_z,
                                            swap_xz=swap_xz,
                                            cell_size=cell_size)
                tree = cKDTree(np.stack([w['tx'], w['tz']], axis=1))
                _, idx = tree.query(np.stack([ox, oz], axis=1), k=1)
                delta = oy - w['ty'][idx]
                row = (np.median(delta), delta.std(),
                       np.percentile(delta, 5), np.percentile(delta, 95),
                       abs(np.median(delta) - expected))
                print(f"{str(flip_x):>7} {str(flip_z):>7} {str(swap_xz):>8} | "
                      f"{row[0]:>7.2f} {row[1]:>6.2f} {row[2]:>7.2f} {row[3]:>7.2f} {row[4]:>10.2f}")
                score = row[1] + row[4]
                if best is None or score < best[0]:
                    best = (score, flip_x, flip_z, swap_xz, row)
    print('-' * 70)
    _, bfx, bfz, bsw, br = best
    print(f"best combo: flip_x={bfx}  flip_z={bfz}  swap_xz={bsw}  -> "
          f"median delta={br[0]:.2f}  std={br[1]:.2f}")
    return {'flip_x': bfx, 'flip_z': bfz, 'swap_xz': bsw,
            'median': br[0], 'std': br[1]}


def f_plot_mouse_vs_terrain(terrain_data, mov_data, terr_kw=None,
                              expected_delta=0.4, label='cam_y',
                              print_first_n=0):
    # 3-panel diagnostic: cam_y + terrain_y_at_cam_xz over time, delta
    # over time, delta histogram. NN-lookup against the full terrain via
    # f_terrain_world_coords. Used to verify the camera is above ground
    # in world coords (independent of any rendering orientation).
    #
    # Parameters
    # ----------
    # mov_data       : either raw mov_data (uses rb anchor, delta peaks
    #     near 0 or slightly negative for this rig) or pre-shifted
    #     mov_data_eye (uses camera position, delta peaks near +0.4 for
    #     this rig). Pick whichever the calling context expects.
    # terr_kw        : orientation kwargs forwarded to f_terrain_world_coords.
    # expected_delta : where the orange reference line is drawn (default
    #     +0.4 = "camera 0.4 above terrain"). Set to ~−0.2 if you're
    #     passing raw mov_data on this rig.
    # label          : caption used in legend / column headers (e.g.
    #     'cam_y' or 'rb_y').
    # print_first_n  : if > 0, also prints a table of the first N frames
    #     showing cam_x/cam_z/cam_y/terr_y/delta/nn_hors_dist.
    #
    # Returns dict {mouse_y, terrain_y, delta, nn_hors_dist}.
    if terr_kw is None:
        terr_kw = {}
    w = f_terrain_world_coords(terrain_data, **terr_kw)
    tree = cKDTree(np.column_stack([w['tx'], w['tz']]))
    cam_xz = np.column_stack([mov_data['x_pos'], mov_data['z_pos']])
    nn_dist, kd_idx = tree.query(cam_xz, k=1)
    terr_y_at_cam = w['ty'][kd_idx]
    delta = mov_data['y_pos'] - terr_y_at_cam

    T_total = len(delta)
    frame_idx = np.arange(T_total)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(frame_idx, mov_data['y_pos'], color='tab:blue',  lw=0.8, label=f'{label}')
    axes[0].plot(frame_idx, terr_y_at_cam,     color='tab:brown', lw=0.8, label=f'terrain_y at {label} xz')
    axes[0].set(xlabel='frame', ylabel='world y',
                title=f'{label} vs terrain_y over time')
    axes[0].legend(loc='upper right')

    axes[1].plot(frame_idx, delta, color='tab:green', lw=0.8)
    axes[1].axhline(0,              color='red',    lw=0.8, ls='--', label='ground (delta=0)')
    axes[1].axhline(expected_delta, color='orange', lw=0.8, ls=':',  label=f'expected {expected_delta:+.2f}')
    axes[1].set(xlabel='frame', ylabel=f'{label} − terrain_y',
                title='delta over time')
    axes[1].legend(loc='upper right')
    _post = delta[20:] if T_total > 20 else delta
    if len(_post) > 0:
        ymin = min(-1.0, np.percentile(_post, 1))
        ymax = max( 2.0, np.percentile(_post, 99))
        axes[1].set_ylim(ymin, ymax)

    axes[2].hist(delta, bins=60, color='tab:green', alpha=0.7)
    axes[2].axvline(0,              color='red',    lw=1.0, ls='--')
    axes[2].axvline(expected_delta, color='orange', lw=1.0, ls=':')
    axes[2].set(xlabel=f'{label} − terrain_y', ylabel='# frames',
                title=f'delta hist   median={np.median(delta):+.3f}   frac>0={np.mean(delta > 0):.3f}')

    plt.tight_layout()
    plt.show()

    if print_first_n > 0:
        n = min(int(print_first_n), T_total)
        print(f'frame | {"cam_x":>7} {"cam_z":>7} | {label:>7} {"terr_y":>7} '
              f'{"delta":>7} {"nn_hors":>8}')
        print('-' * 60)
        for i in range(n):
            print(f'{i:5d} | {mov_data["x_pos"][i]:7.2f} {mov_data["z_pos"][i]:7.2f} | '
                  f'{mov_data["y_pos"][i]:7.2f} {terr_y_at_cam[i]:7.2f} '
                  f'{delta[i]:+7.3f} {nn_dist[i]:8.3f}')

    return {'mouse_y': np.asarray(mov_data['y_pos']),
            'terrain_y': terr_y_at_cam, 'delta': delta, 'nn_hors_dist': nn_dist}


def f_diag_chunk_overlap(terrain_data, n_compare=3, z_for_compare=None):
    # Compare boundary cells of two x-adjacent chunks to detect the
    # overlap convention. Picks the z-row of chunks with the most
    # entries, takes the two leftmost x-adjacent chunks A and B, pivots
    # both to 2D heightmap grids, and computes the RMS-diff between A's
    # last n_compare rows and B's first n_compare rows over all shared z
    # indices.
    #
    # Interpretation:
    #   1-point overlap: zero appears only at (A_last, B_first)
    #   2-point overlap: zero on the anti-diagonal of the 2x2 submatrix
    #   3-point overlap: zero on the anti-diagonal of the 3x3 submatrix
    #                    (this rig — A[122..124] = B[0..2])
    # Returns the RMS-diff table + the (A_idx, B_idx) pair with smallest
    # rms per row (the empirical overlap pattern).
    chunk_keys = terrain_data.groupby(['ChunkPosX', 'ChunkPosZ']).size().index.tolist()
    chunks_by_z = {}
    for cpx, cpz in chunk_keys:
        chunks_by_z.setdefault(cpz, []).append(cpx)
    if z_for_compare is None:
        z_for_compare = max(chunks_by_z, key=lambda z: len(chunks_by_z[z]))
    xs = sorted(chunks_by_z[z_for_compare])
    if len(xs) < 2:
        print(f'no x-adjacent chunks at ChunkPosZ={z_for_compare}')
        return None
    cpx_A, cpx_B = xs[0], xs[1]
    print(f'Chunk A: ChunkPosX={cpx_A}, ChunkPosZ={z_for_compare}')
    print(f'Chunk B: ChunkPosX={cpx_B}, ChunkPosZ={z_for_compare}  (x-adjacent, pitch = {cpx_B - cpx_A})')

    chA = terrain_data[(terrain_data['ChunkPosX'] == cpx_A) & (terrain_data['ChunkPosZ'] == z_for_compare)]
    chB = terrain_data[(terrain_data['ChunkPosX'] == cpx_B) & (terrain_data['ChunkPosZ'] == z_for_compare)]
    hmA = chA.pivot(index='x', columns='z', values='height').values
    hmB = chB.pivot(index='x', columns='z', values='height').values
    print(f'chunk A heightmap shape: {hmA.shape}')
    print(f'chunk B heightmap shape: {hmB.shape}')

    n_A = hmA.shape[0]
    rms = np.zeros((n_compare, n_compare))
    print()
    header = f'{"":>14}' + ''.join([f' B[x={j}]'.rjust(10) for j in range(n_compare)])
    print(header)
    for ai in range(n_compare):
        i_A = n_A - n_compare + ai
        row = f'A[x={i_A:3d}]      '
        for i_B in range(n_compare):
            rms[ai, i_B] = float(np.sqrt(np.mean((hmA[i_A] - hmB[i_B])**2)))
            row += f'  {rms[ai, i_B]:8.4f}'
        print(row)
    # detect overlap pattern: anti-diagonal zeros
    print()
    print(f'sample heights at z = 0, 50, {hmA.shape[1]-1}:')
    for ai in range(n_compare):
        i_A = n_A - n_compare + ai
        print(f'  A[{i_A:3d}, z]: {hmA[i_A, [0, hmA.shape[1]//2, hmA.shape[1]-1]]}')
        i_B = ai
        print(f'  B[{i_B:3d}, z]: {hmB[i_B, [0, hmB.shape[1]//2, hmB.shape[1]-1]]}')
    return {'rms': rms, 'chunk_A_pos': (cpx_A, z_for_compare),
            'chunk_B_pos': (cpx_B, z_for_compare),
            'hmA': hmA, 'hmB': hmB}


def f_check_obj_coord_system(terrain_data, obj_data, mov_data=None,
                               world_coord_threshold=200.0):
    # Heuristic check: are ObjLocX/Y/Z world coords or chunk-relative?
    # Prints the ranges of ObjLoc, ChunkPos, terrain heights, and mouse
    # positions. If ObjLocX/Z absolute values exceed `world_coord_threshold`,
    # they're world coords (matches ChunkPosX/Z). Otherwise they may be
    # chunk-relative — needs explicit chunk offset before terrain lookup.
    print('--- Object-coord-system check ---')
    print(f"ObjLocX range  : [{obj_data['ObjLocX'].min():.1f}, {obj_data['ObjLocX'].max():.1f}]")
    print(f"ObjLocZ range  : [{obj_data['ObjLocZ'].min():.1f}, {obj_data['ObjLocZ'].max():.1f}]")
    print(f"ChunkPosX range: [{terrain_data['ChunkPosX'].min():.1f}, {terrain_data['ChunkPosX'].max():.1f}]")
    print(f"ChunkPosZ range: [{terrain_data['ChunkPosZ'].min():.1f}, {terrain_data['ChunkPosZ'].max():.1f}]")
    print(f"ObjLocY range  : [{obj_data['ObjLocY'].min():.2f}, {obj_data['ObjLocY'].max():.2f}]")
    print(f"terrain height : [{terrain_data['height'].min():.2f}, {terrain_data['height'].max():.2f}]")
    if mov_data is not None:
        print(f"mouse  x range : [{mov_data['x_pos'].min():.1f}, {mov_data['x_pos'].max():.1f}]")
        print(f"mouse  z range : [{mov_data['z_pos'].min():.1f}, {mov_data['z_pos'].max():.1f}]")
    obj_xz_world_like = (
        obj_data['ObjLocX'].min() < -world_coord_threshold or
        obj_data['ObjLocX'].max() >  world_coord_threshold or
        obj_data['ObjLocZ'].min() < -world_coord_threshold or
        obj_data['ObjLocZ'].max() >  world_coord_threshold
    )
    if obj_xz_world_like:
        print('  → ObjLocX/Z look like WORLD coords (range overlaps ChunkPosX/Z and mouse). '
              'Object alignment via direct KD-tree query is correct.')
    else:
        print('  → ObjLocX/Z values are SMALL — possibly chunk-relative. '
              'Then a chunk offset must be added before terrain lookup. '
              'Inspect object_data column meanings before trusting alignment.')
    return {'world_coords_likely': bool(obj_xz_world_like)}


def f_run_terrain_diagnostics(terrain_data, obj_data, mov_data, mov_data_eye=None,
                                obj_size=None, terr_kw=None,
                                expected_delta=0.4, run_chunk_overlap=True,
                                run_obj_coord_check=True, run_mouse_vs_terrain=True,
                                run_object_alignment=True):
    # Master wrapper that runs the canonical terrain coord-system
    # diagnostics in order. Use for a one-shot health check on a new
    # dataset (or after refactoring f_terrain_world_coords).
    #
    # Steps:
    #  1. Object-coord-system range check (cheap; clarifies whether
    #     ObjLocX/Z are world or chunk-relative).
    #  2. Chunk-overlap diagnostic (determines 1/2/3-point overlap by
    #     comparing boundary heights of adjacent chunks).
    #  3. Object-alignment plot (objects sit obj_size['height'] above
    #     terrain — within ±0.5 of expected).
    #  4. Mouse-vs-terrain plot — runs with both mov_data (rb anchor)
    #     AND mov_data_eye (camera) if both are provided.
    if run_obj_coord_check:
        print('\n==== 1) object-coord-system check ====')
        f_check_obj_coord_system(terrain_data, obj_data, mov_data=mov_data)
    if run_chunk_overlap:
        print('\n==== 2) chunk overlap diagnostic ====')
        f_diag_chunk_overlap(terrain_data)
    if run_object_alignment and obj_size is not None:
        print('\n==== 3) object alignment ====')
        kw = terr_kw if terr_kw is not None else {}
        f_plot_terrain_object_alignment(terrain_data, obj_data, obj_size, **kw)
    if run_mouse_vs_terrain:
        print('\n==== 4) mouse-vs-terrain (rb anchor) ====')
        f_plot_mouse_vs_terrain(terrain_data, mov_data, terr_kw=terr_kw,
                                  expected_delta=0.0, label='rb_y')
        if mov_data_eye is not None:
            print('\n==== 4b) camera-vs-terrain (rb anchor + cam_height) ====')
            f_plot_mouse_vs_terrain(terrain_data, mov_data_eye, terr_kw=terr_kw,
                                      expected_delta=expected_delta, label='cam_y')


def f_diag_chunk0_mouse_height(terrain_data, mov_data, max_frames=5000,
                                center=62, flip_z=True):
    # Hand-rolled minimal mouse-vs-terrain check, bypassing
    # f_terrain_world_coords entirely. Restricted to chunk (0, 0).
    # For each of the first max_frames mouse positions, finds the closest
    # cell in chunk 0 by Euclidean distance in (x_local - center,
    # z_local - center) and plots height vs mouse y.
    #
    # Use when you want a sanity check that doesn't share code with the
    # production lookup helpers. The mouse needs to actually traverse
    # chunk 0 for this to be meaningful — frames where the mouse is far
    # from chunk 0 will just snap to the chunk's boundary cell and read
    # an edge height.
    #
    # Parameters
    # ----------
    # terrain_data : full terrain DataFrame (will be filtered to ChunkPos
    #     0,0 inside).
    # mov_data     : mov_data dict from f_proc_movement (uses x_pos, y_pos,
    #     z_pos).
    # max_frames   : how many leading frames of the mouse trajectory to
    #     process. Default 5000.
    # center       : integer to subtract from local x and z indices to
    #     center the chunk around (0, 0). Default 62 (= cell_max / 2 for
    #     125-cell chunks).
    # flip_z       : if True, invert z index (z_world = center - z_local)
    #     to match this rig's flip_z convention. Default True; set False
    #     for a true raw "just subtract 62" lookup.
    #
    # Returns a dict {mouse_y, terrain_y_at_nn, delta, nn_dist, in_chunk}.
    # Also produces a figure with three panels: y over time, delta over
    # time, delta histogram.

    c0 = terrain_data[(terrain_data['ChunkPosX'] == 0) & (terrain_data['ChunkPosZ'] == 0)]
    if len(c0) == 0:
        print('no terrain rows in chunk (0, 0) — aborting')
        return None
    print(f'chunk-0 rows: {len(c0)}   (expected 125 x 125 = 15625 for full chunk)')

    tx = c0['x'].values.astype(np.float64) - float(center)
    if flip_z:
        # z_world = (cell_max - z_local) - center. With cell_max = 2*center
        # = 124 and center = 62, this is (124 - z) - 62 = 62 - z. Equivalent
        # to negating after centering: tz = -(z_local - center) = center - z.
        tz = float(center) - c0['z'].values.astype(np.float64)
    else:
        tz = c0['z'].values.astype(np.float64) - float(center)
    ty = c0['height'].values.astype(np.float64)

    mx_all = np.asarray(mov_data['x_pos'], dtype=np.float64)
    my_all = np.asarray(mov_data['y_pos'], dtype=np.float64)
    mz_all = np.asarray(mov_data['z_pos'], dtype=np.float64)
    n = min(int(max_frames), len(mx_all))
    mx = mx_all[:n]; my = my_all[:n]; mz = mz_all[:n]

    # vectorized NN by brute-force Euclidean: (n_mouse, n_terrain) distance
    # matrix. Float32 is enough; for n=5000, T=15625 → 312 MB; if too big,
    # falls back to a row-wise loop.
    n_terr = len(tx)
    bytes_full = n * n_terr * 4
    if bytes_full > 1_500_000_000:   # 1.5 GB
        terr_y_at_nn = np.zeros(n)
        nn_dist      = np.zeros(n)
        for i in range(n):
            dx = tx - mx[i]; dz = tz - mz[i]
            d2 = dx*dx + dz*dz
            j = int(np.argmin(d2))
            terr_y_at_nn[i] = ty[j]
            nn_dist[i]      = float(np.sqrt(d2[j]))
    else:
        dx = (tx[None, :] - mx[:, None]).astype(np.float32)
        dz = (tz[None, :] - mz[:, None]).astype(np.float32)
        d2 = dx*dx + dz*dz
        j_min = np.argmin(d2, axis=1)
        terr_y_at_nn = ty[j_min]
        nn_dist      = np.sqrt(d2[np.arange(n), j_min])

    delta = my - terr_y_at_nn
    # "in chunk" = mouse xz inside chunk-0 footprint [-center, +center]
    in_chunk = (np.abs(mx) <= center) & (np.abs(mz) <= center)
    n_in = int(in_chunk.sum())
    print(f'frames in chunk-0 footprint: {n_in} / {n}   '
          f'({100*n_in/max(n,1):.1f}%)')
    print(f'overall delta median = {np.median(delta):+.3f}   std = {delta.std():.3f}')
    if n_in > 0:
        print(f'in-chunk delta median = {np.median(delta[in_chunk]):+.3f}   '
              f'std = {delta[in_chunk].std():.3f}')

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fr = np.arange(n)
    axes[0].plot(fr, my,          color='tab:blue',  lw=0.7, label='mouse_y')
    axes[0].plot(fr, terr_y_at_nn,color='tab:brown', lw=0.7, label='terrain_y at NN')
    axes[0].set(xlabel='frame', ylabel='world y',
                title=f'chunk-0 hand-rolled NN — first {n} frames'
                      f'   (flip_z={flip_z}, center={center})')
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].plot(fr, delta, color='tab:green', lw=0.7, label='delta (all frames)')
    if n_in > 0:
        axes[1].plot(fr[in_chunk], delta[in_chunk], '.', color='black',
                     ms=3, alpha=0.6, label='in chunk-0 footprint')
    axes[1].axhline(0.0, color='red',    lw=0.8, ls='--')
    axes[1].axhline(0.4, color='orange', lw=0.8, ls=':', label='expected +0.4')
    axes[1].set(xlabel='frame', ylabel='mouse_y − terrain_y',
                title=f'delta over time (median={np.median(delta):+.3f})')
    axes[1].legend(loc='upper right', fontsize=8)

    axes[2].hist(delta, bins=80, color='tab:green', alpha=0.5, label='all frames')
    if n_in > 0:
        axes[2].hist(delta[in_chunk], bins=80, color='black', alpha=0.6,
                     label='in chunk-0 footprint')
    axes[2].axvline(0.0, color='red',    lw=1, ls='--')
    axes[2].axvline(0.4, color='orange', lw=1, ls=':')
    axes[2].axvline(np.median(delta), color='black', lw=1, ls='-',
                    label=f'median={np.median(delta):+.3f}')
    axes[2].set(xlabel='mouse_y − terrain_y', ylabel='# frames',
                title='delta histogram')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

    return {'mouse_y': my, 'terrain_y_at_nn': terr_y_at_nn,
            'delta': delta, 'nn_dist': nn_dist, 'in_chunk': in_chunk}


def f_diag_chunk0_mouse_height_bilinear(terrain_data, mov_data, max_frames=5000,
                                         center=62, flip_z=True,
                                         offset_x=0, offset_z=0):
    # Same diagnostic as f_diag_chunk0_mouse_height, but the terrain height
    # under the mouse is computed by BILINEAR INTERPOLATION across the 4
    # cells of the heightmap square the mouse is in — not nearest-neighbor.
    # This mimics what Unity's terrain mesh actually does between
    # heightmap vertices, so the delta = mouse_y − terrain_y_bilinear
    # should be much less noisy than the NN version (no slope-induced
    # snap to a neighboring vertex).
    #
    # Restricted to chunk (ChunkPosX=0, ChunkPosZ=0). For mouse frames
    # where the mouse is outside that chunk's footprint the bilinear
    # lookup is clipped to the chunk boundary (height at the nearest
    # edge); those frames are flagged in `in_chunk` and shown separately
    # in the plot.
    #
    # offset_x / offset_z : shift the assumed chunk center by these cell
    #     counts. Default (0, 0) = use center 62 as canonical. Used by the
    #     companion sweep functions to scan for an empirical optimum.

    c0 = terrain_data[(terrain_data['ChunkPosX'] == 0) & (terrain_data['ChunkPosZ'] == 0)]
    if len(c0) == 0:
        print('no chunk-0 data')
        return None

    # Build a regular (x_idx, z_idx) → height grid.
    H_grid = c0.pivot(index='x', columns='z', values='height').values.astype(np.float64)
    n_x, n_z = H_grid.shape
    cell_max_x = n_x - 1
    cell_max_z = n_z - 1

    mx_all = np.asarray(mov_data['x_pos'], dtype=np.float64)
    my_all = np.asarray(mov_data['y_pos'], dtype=np.float64)
    mz_all = np.asarray(mov_data['z_pos'], dtype=np.float64)
    n = min(int(max_frames), len(mx_all))
    mx = mx_all[:n]; my = my_all[:n]; mz = mz_all[:n]

    # Mouse world (mx, mz) → fractional cell indices (cx_f, cz_f).
    # world_x = (x_idx - (center + offset_x)) * cell_size   (cell_size = 1)
    #   ⇒ x_idx = mx + center + offset_x
    # world_z (flip_z=True) = ((center + offset_z) - z_idx)
    #   ⇒ z_idx = (center + offset_z) - mz
    cx_f = mx + (center + offset_x)
    if flip_z:
        cz_f = (center + offset_z) - mz
    else:
        cz_f = mz + (center + offset_z)

    in_chunk = (cx_f >= 0) & (cx_f <= cell_max_x) & (cz_f >= 0) & (cz_f <= cell_max_z)
    n_in = int(in_chunk.sum())

    # Bilinear: clamp to [0, cell_max - eps] so floor + 1 stays in bounds.
    eps = 1e-6
    cx_c = np.clip(cx_f, 0.0, cell_max_x - eps)
    cz_c = np.clip(cz_f, 0.0, cell_max_z - eps)
    ix_lo = np.floor(cx_c).astype(np.int64)
    iz_lo = np.floor(cz_c).astype(np.int64)
    ix_hi = ix_lo + 1
    iz_hi = iz_lo + 1
    fx = cx_c - ix_lo
    fz = cz_c - iz_lo

    H_ll = H_grid[ix_lo, iz_lo]
    H_hl = H_grid[ix_hi, iz_lo]
    H_lh = H_grid[ix_lo, iz_hi]
    H_hh = H_grid[ix_hi, iz_hi]
    terr_y = (H_ll * (1 - fx) * (1 - fz)
              + H_hl * fx       * (1 - fz)
              + H_lh * (1 - fx) * fz
              + H_hh * fx       * fz)

    delta = my - terr_y

    print(f'chunk-0 rows: {len(c0)} (pivoted to {n_x}x{n_z} grid).  '
          f'offset=({offset_x:+d},{offset_z:+d})  flip_z={flip_z}')
    print(f'frames in chunk-0 footprint: {n_in} / {n}   ({100*n_in/max(n,1):.1f}%)')
    print(f'overall delta median = {np.median(delta):+.3f}  std = {delta.std():.3f}')
    if n_in > 0:
        print(f'in-chunk delta median = {np.median(delta[in_chunk]):+.3f}  '
              f'std = {delta[in_chunk].std():.3f}')

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fr = np.arange(n)
    axes[0].plot(fr, my,     color='tab:blue',  lw=0.7, label='mouse_y')
    axes[0].plot(fr, terr_y, color='tab:brown', lw=0.7, label='terrain_y (bilinear)')
    axes[0].set(xlabel='frame', ylabel='world y',
                title=f'chunk-0 BILINEAR — first {n} frames  '
                      f'(offset=({offset_x:+d},{offset_z:+d}), flip_z={flip_z})')
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].plot(fr, delta, color='tab:green', lw=0.7, label='delta (all frames)')
    if n_in > 0:
        axes[1].plot(fr[in_chunk], delta[in_chunk], '.', color='black',
                     ms=3, alpha=0.6, label='in chunk-0 footprint')
    axes[1].axhline(0.0, color='red',    lw=0.8, ls='--')
    axes[1].axhline(0.4, color='orange', lw=0.8, ls=':', label='expected +0.4')
    axes[1].set(xlabel='frame', ylabel='mouse_y − terrain_y (bilinear)',
                title=f'delta over time   median={np.median(delta):+.3f}   '
                      f'std={delta.std():.3f}')
    axes[1].legend(loc='upper right', fontsize=8)

    axes[2].hist(delta, bins=80, color='tab:green', alpha=0.5, label='all frames')
    if n_in > 0:
        axes[2].hist(delta[in_chunk], bins=80, color='black', alpha=0.6,
                     label='in chunk-0 footprint')
    axes[2].axvline(0.0, color='red',    lw=1, ls='--')
    axes[2].axvline(0.4, color='orange', lw=1, ls=':')
    axes[2].axvline(np.median(delta), color='black', lw=1, ls='-',
                    label=f'median={np.median(delta):+.3f}')
    axes[2].set(xlabel='mouse_y − terrain_y (bilinear)', ylabel='# frames',
                title='delta histogram')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

    return {'mouse_y': my, 'terrain_y_bilinear': terr_y,
            'delta': delta, 'in_chunk': in_chunk,
            'cell_frac_x': fx, 'cell_frac_z': fz}


def f_diag_chunk0_offset_sweep(terrain_data, mov_data, max_frames=5000,
                                 center=62, flip_z=True,
                                 offsets=(-3, -2, -1, 0, 1, 2, 3),
                                 hist_bins=200, hist_range=None,
                                 smooth_sigma=2.0):
    # Sweep the chunk-0 center offset on EACH axis SEPARATELY and measure
    # how the mouse-vs-terrain delta histogram changes. Same hand-rolled
    # NN lookup as f_diag_chunk0_mouse_height, but repeated for every
    # offset in `offsets`. For each (x_offset, 0) and (0, z_offset), the
    # function reports:
    #   - peak_loc : delta value at which the SMOOTHED histogram is most
    #                populated (= argmax in delta units, with Gaussian
    #                smoothing applied to stabilize against bin noise).
    #   - std      : standard deviation of delta (lower = better
    #                alignment; the lookup is finding a single coherent
    #                surface instead of multiple).
    #
    # hist_bins / smooth_sigma : raw histogram resolution and Gaussian
    #     smoothing width (in bins) applied before argmax. Defaults
    #     200 bins / sigma=2 bins give a more stable peak than the raw
    #     argmax of a 80-bin histogram, especially for 5000 frames where
    #     individual bins are noisy. Set smooth_sigma=0 to disable smoothing.
    #
    # Default offsets sweep ±3 around the canonical 62. Pass a custom
    # iterable to widen / narrow the search.

    c0 = terrain_data[(terrain_data['ChunkPosX'] == 0) & (terrain_data['ChunkPosZ'] == 0)]
    if len(c0) == 0:
        print('no chunk-0 data')
        return None
    cx_idx = c0['x'].values.astype(np.float64)
    cz_idx = c0['z'].values.astype(np.float64)
    ty     = c0['height'].values.astype(np.float64)

    mx_all = np.asarray(mov_data['x_pos'], dtype=np.float64)
    my_all = np.asarray(mov_data['y_pos'], dtype=np.float64)
    mz_all = np.asarray(mov_data['z_pos'], dtype=np.float64)
    n = min(int(max_frames), len(mx_all))
    mx = mx_all[:n]; my = my_all[:n]; mz = mz_all[:n]

    def _delta(ox, oz):
        tx = cx_idx - (center + ox)
        if flip_z:
            tz = (center + oz) - cz_idx       # = -(cz_idx - (center + oz))
        else:
            tz = cz_idx - (center + oz)
        dx = (tx[None, :] - mx[:, None]).astype(np.float32)
        dz = (tz[None, :] - mz[:, None]).astype(np.float32)
        d2 = dx * dx + dz * dz
        j_min = np.argmin(d2, axis=1)
        return my - ty[j_min]

    offsets_arr = np.asarray(list(offsets))
    peak_loc_x = np.zeros(len(offsets_arr))
    stds_x     = np.zeros(len(offsets_arr))
    peak_loc_z = np.zeros(len(offsets_arr))
    stds_z     = np.zeros(len(offsets_arr))
    med_x      = np.zeros(len(offsets_arr))
    med_z      = np.zeros(len(offsets_arr))

    def _smoothed_peak(delta_vec):
        h, e = np.histogram(delta_vec, bins=hist_bins, range=hist_range)
        if smooth_sigma and smooth_sigma > 0:
            h = sc.ndimage.gaussian_filter1d(h.astype(np.float64), sigma=float(smooth_sigma))
        a = int(np.argmax(h))
        return 0.5 * (e[a] + e[a + 1])

    print(f'sweeping {len(offsets_arr)} offsets on each axis ({n} mouse frames)...')
    for i, off in enumerate(offsets_arr):
        d_x = _delta(int(off), 0)
        peak_loc_x[i] = _smoothed_peak(d_x)
        stds_x[i]     = d_x.std()
        med_x[i]      = np.median(d_x)

        d_z = _delta(0, int(off))
        peak_loc_z[i] = _smoothed_peak(d_z)
        stds_z[i]     = d_z.std()
        med_z[i]      = np.median(d_z)
        print(f'  offset {off:+d}:   X(peak_loc={peak_loc_x[i]:+.3f} std={stds_x[i]:.3f} med={med_x[i]:+.3f})   '
              f'Z(peak_loc={peak_loc_z[i]:+.3f} std={stds_z[i]:.3f} med={med_z[i]:+.3f})')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(offsets_arr, peak_loc_x, 'o-', color='tab:blue')
    axes[0, 0].axvline(0,   color='red',    ls='--', lw=0.8)
    axes[0, 0].axhline(0.0, color='gray',   ls=':',  lw=0.8, label='ground')
    axes[0, 0].axhline(0.4, color='orange', ls=':',  lw=0.8, label='expected +0.4')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].set(xlabel=f'x offset (rel. to center={center})',
                    ylabel='histogram peak location (delta value)',
                    title='X-axis sweep — peak location')

    axes[1, 0].plot(offsets_arr, stds_x, 'o-', color='tab:orange')
    axes[1, 0].axvline(0, color='red', ls='--', lw=0.8)
    axes[1, 0].set(xlabel=f'x offset (rel. to center={center})',
                    ylabel='delta std',
                    title='X-axis sweep — std')

    axes[0, 1].plot(offsets_arr, peak_loc_z, 'o-', color='tab:blue')
    axes[0, 1].axvline(0,   color='red',    ls='--', lw=0.8)
    axes[0, 1].axhline(0.0, color='gray',   ls=':',  lw=0.8, label='ground')
    axes[0, 1].axhline(0.4, color='orange', ls=':',  lw=0.8, label='expected +0.4')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].set(xlabel=f'z offset (rel. to center={center})',
                    ylabel='histogram peak location (delta value)',
                    title='Z-axis sweep — peak location')

    axes[1, 1].plot(offsets_arr, stds_z, 'o-', color='tab:orange')
    axes[1, 1].axvline(0, color='red', ls='--', lw=0.8)
    axes[1, 1].set(xlabel=f'z offset (rel. to center={center})',
                    ylabel='delta std',
                    title='Z-axis sweep — std')

    plt.tight_layout()
    plt.show()

    best_ox_std = int(offsets_arr[np.argmin(stds_x)])
    best_oz_std = int(offsets_arr[np.argmin(stds_z)])
    print(f'\nbest X offset (min std): {best_ox_std:+d}')
    print(f'best Z offset (min std): {best_oz_std:+d}')

    return {'offsets': offsets_arr,
            'peak_loc_x': peak_loc_x, 'stds_x': stds_x, 'medians_x': med_x,
            'peak_loc_z': peak_loc_z, 'stds_z': stds_z, 'medians_z': med_z,
            'best_ox_std': best_ox_std, 'best_oz_std': best_oz_std}


def f_diag_chunk0_offset_sweep_2d(terrain_data, mov_data, max_frames=5000,
                                    center=62, half_range=3, flip_z=True,
                                    offsets_x=None, offsets_z=None,
                                    hist_bins=200, hist_range=None,
                                    smooth_sigma=2.0,
                                    expected_delta=0.4,
                                    plot_hist_offsets=None):
    # 2D version of f_diag_chunk0_offset_sweep. Runs the chunk-0 hand-rolled
    # NN lookup for every (offset_x, offset_z) combination and reports:
    #   peak_loc : delta value at the histogram argmax (mode of mouse-y −
    #              terrain-y). Best = closest to expected_delta.
    #   std      : standard deviation of delta. Best = minimum.
    # Renders two heatmaps side by side (peak_loc + std) so any diagonal
    # alignment offset shows up directly.
    #
    # Parameters
    # ----------
    # center      : baseline cell index treated as the chunk middle
    #               before adding offsets. Default 62 for 125-cell chunks.
    # half_range  : sweep span. Generates symmetric offsets
    #               np.arange(-half_range, half_range+1) on each axis
    #               (default 3 → -3..+3 = 7 offsets per axis = 49 cells).
    # offsets_x   : explicit override for x offsets (sequence of ints).
    #               If None (default), uses half_range. Pass an iterable
    #               (e.g. [-5, 0, 5]) for non-symmetric or sparse sweeps.
    # offsets_z   : explicit override for z offsets, same convention.
    # expected_delta : target value for the peak_loc heatmap centering.
    #               Default 0.4 (camera 0.4 above terrain). Set 0 if you
    #               want to optimize for mouse-on-ground.
    # plot_hist_offsets : list / iterable of (ox, oz) tuples. If provided,
    #               renders a second figure showing the raw + smoothed
    #               delta histogram at each of these offsets, with peak,
    #               median, and expected_delta marked. Useful for
    #               inspecting what the distribution actually looks like
    #               at the empirical optimum (or far from it).

    c0 = terrain_data[(terrain_data['ChunkPosX'] == 0) & (terrain_data['ChunkPosZ'] == 0)]
    if len(c0) == 0:
        print('no chunk-0 data')
        return None
    cx_idx = c0['x'].values.astype(np.float64)
    cz_idx = c0['z'].values.astype(np.float64)
    ty     = c0['height'].values.astype(np.float64)

    mx_all = np.asarray(mov_data['x_pos'], dtype=np.float64)
    my_all = np.asarray(mov_data['y_pos'], dtype=np.float64)
    mz_all = np.asarray(mov_data['z_pos'], dtype=np.float64)
    n = min(int(max_frames), len(mx_all))
    mx = mx_all[:n]; my = my_all[:n]; mz = mz_all[:n]

    if offsets_x is None:
        offsets_x = np.arange(-int(half_range), int(half_range) + 1)
    if offsets_z is None:
        offsets_z = np.arange(-int(half_range), int(half_range) + 1)
    ox_arr = np.asarray(list(offsets_x))
    oz_arr = np.asarray(list(offsets_z))
    n_x = len(ox_arr); n_z = len(oz_arr)
    peak_loc = np.zeros((n_z, n_x))   # row = z, col = x (matches imshow orientation)
    std_2d   = np.zeros((n_z, n_x))
    med_2d   = np.zeros((n_z, n_x))

    print(f'2D sweep: {n_x} x_offsets x {n_z} z_offsets = {n_x*n_z} combinations '
          f'({n} mouse frames)...')
    for iz, oz in enumerate(oz_arr):
        for ix, ox in enumerate(ox_arr):
            tx = cx_idx - (center + ox)
            if flip_z:
                tz = (center + oz) - cz_idx
            else:
                tz = cz_idx - (center + oz)
            tree = cKDTree(np.column_stack([tx, tz]))
            _, j_min = tree.query(np.column_stack([mx, mz]), k=1)
            delta = my - ty[j_min]
            hist, edges = np.histogram(delta, bins=hist_bins, range=hist_range)
            if smooth_sigma and smooth_sigma > 0:
                hist = sc.ndimage.gaussian_filter1d(hist.astype(np.float64),
                                                     sigma=float(smooth_sigma))
            arg = int(np.argmax(hist))
            peak_loc[iz, ix] = 0.5 * (edges[arg] + edges[arg + 1])
            std_2d[iz, ix]   = delta.std()
            med_2d[iz, ix]   = np.median(delta)

    # Find argmin of std and argmin of |peak_loc - expected_delta|, plus
    # the argmin of |median - expected_delta| for the third panel.
    iz_s, ix_s = np.unravel_index(int(np.argmin(std_2d)), std_2d.shape)
    iz_p, ix_p = np.unravel_index(int(np.argmin(np.abs(peak_loc - expected_delta))),
                                    peak_loc.shape)
    iz_m, ix_m = np.unravel_index(int(np.argmin(np.abs(med_2d - expected_delta))),
                                    med_2d.shape)
    best_ox_std, best_oz_std = int(ox_arr[ix_s]), int(oz_arr[iz_s])
    best_ox_peak, best_oz_peak = int(ox_arr[ix_p]), int(oz_arr[iz_p])
    best_ox_med,  best_oz_med  = int(ox_arr[ix_m]), int(oz_arr[iz_m])

    # Heatmaps. Diverging cmap on peak_loc and median centered at
    # expected_delta so the target value shows up white and distance
    # from target reads off the color magnitude.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _half = max(abs(peak_loc.max() - expected_delta),
                abs(peak_loc.min() - expected_delta))
    im0 = axes[0].imshow(peak_loc,
                          origin='lower', aspect='auto', cmap='RdBu_r',
                          vmin=expected_delta - _half,
                          vmax=expected_delta + _half,
                          extent=[ox_arr[0]-0.5, ox_arr[-1]+0.5,
                                  oz_arr[0]-0.5, oz_arr[-1]+0.5])
    axes[0].set_xticks(ox_arr)
    axes[0].set_yticks(oz_arr)
    axes[0].axvline(0, color='gray', lw=0.5)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].plot(best_ox_peak, best_oz_peak, marker='*', ms=20,
                 color='lime', markeredgecolor='black',
                 label=f'closest to {expected_delta:.1f} @ ({best_ox_peak:+d},{best_oz_peak:+d})')
    axes[0].set(xlabel=f'x offset (rel. to center={center})',
                 ylabel=f'z offset (rel. to center={center})',
                 title=f'peak_loc (delta value at histogram mode; target={expected_delta:.1f})')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.colorbar(im0, ax=axes[0], label='delta peak (mouse_y − terrain_y)')

    im1 = axes[1].imshow(std_2d, origin='lower', aspect='auto', cmap='viridis',
                          extent=[ox_arr[0]-0.5, ox_arr[-1]+0.5,
                                  oz_arr[0]-0.5, oz_arr[-1]+0.5])
    axes[1].set_xticks(ox_arr)
    axes[1].set_yticks(oz_arr)
    axes[1].axvline(0, color='red', lw=0.5)
    axes[1].axhline(0, color='red', lw=0.5)
    axes[1].plot(best_ox_std, best_oz_std, marker='*', ms=20,
                 color='lime', markeredgecolor='black',
                 label=f'min std @ ({best_ox_std:+d},{best_oz_std:+d})')
    axes[1].set(xlabel=f'x offset (rel. to center={center})',
                 ylabel=f'z offset (rel. to center={center})',
                 title='delta std (lower = tighter lookup)')
    axes[1].legend(loc='upper right', fontsize=8)
    fig.colorbar(im1, ax=axes[1], label='std')

    _half_m = max(abs(med_2d.max() - expected_delta),
                  abs(med_2d.min() - expected_delta))
    im2 = axes[2].imshow(med_2d, origin='lower', aspect='auto', cmap='RdBu_r',
                          vmin=expected_delta - _half_m,
                          vmax=expected_delta + _half_m,
                          extent=[ox_arr[0]-0.5, ox_arr[-1]+0.5,
                                  oz_arr[0]-0.5, oz_arr[-1]+0.5])
    axes[2].set_xticks(ox_arr)
    axes[2].set_yticks(oz_arr)
    axes[2].axvline(0, color='gray', lw=0.5)
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].plot(best_ox_med, best_oz_med, marker='*', ms=20,
                 color='lime', markeredgecolor='black',
                 label=f'closest to {expected_delta:.1f} @ ({best_ox_med:+d},{best_oz_med:+d})')
    axes[2].set(xlabel=f'x offset (rel. to center={center})',
                 ylabel=f'z offset (rel. to center={center})',
                 title=f'median (50%; target={expected_delta:.1f}) — robust to noise')
    axes[2].legend(loc='upper right', fontsize=8)
    fig.colorbar(im2, ax=axes[2], label='delta median (mouse_y − terrain_y)')

    plt.tight_layout()
    plt.show()

    print(f'\nbest (ox, oz) by min std        : ({best_ox_std:+d}, {best_oz_std:+d})  '
          f'std={std_2d[iz_s, ix_s]:.3f}  peak_loc={peak_loc[iz_s, ix_s]:+.3f}  '
          f'med={med_2d[iz_s, ix_s]:+.3f}')
    print(f'best (ox, oz) by |peak−target|  : ({best_ox_peak:+d}, {best_oz_peak:+d})  '
          f'std={std_2d[iz_p, ix_p]:.3f}  peak_loc={peak_loc[iz_p, ix_p]:+.3f}  '
          f'med={med_2d[iz_p, ix_p]:+.3f}')
    print(f'best (ox, oz) by |median−target|: ({best_ox_med:+d}, {best_oz_med:+d})  '
          f'std={std_2d[iz_m, ix_m]:.3f}  peak_loc={peak_loc[iz_m, ix_m]:+.3f}  '
          f'med={med_2d[iz_m, ix_m]:+.3f}')

    # Optional: render the actual delta histogram at user-specified offsets.
    if plot_hist_offsets is not None and len(list(plot_hist_offsets)) > 0:
        pairs = list(plot_hist_offsets)
        n_pairs = len(pairs)
        n_cols = min(n_pairs, 4)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        fig_h, axes_h = plt.subplots(n_rows, n_cols,
                                      figsize=(4 * n_cols, 3 * n_rows),
                                      squeeze=False)
        for idx, (ox, oz) in enumerate(pairs):
            r, c = divmod(idx, n_cols)
            ax_h = axes_h[r, c]
            tx = cx_idx - (center + int(ox))
            if flip_z:
                tz = (center + int(oz)) - cz_idx
            else:
                tz = cz_idx - (center + int(oz))
            tree = cKDTree(np.column_stack([tx, tz]))
            _, j_min = tree.query(np.column_stack([mx, mz]), k=1)
            delta = my - ty[j_min]
            hist, edges = np.histogram(delta, bins=hist_bins, range=hist_range)
            hist_smooth = hist.astype(np.float64)
            if smooth_sigma and smooth_sigma > 0:
                hist_smooth = sc.ndimage.gaussian_filter1d(hist_smooth,
                                                            sigma=float(smooth_sigma))
            centers_bin = 0.5 * (edges[:-1] + edges[1:])
            arg = int(np.argmax(hist_smooth))
            peak_val = centers_bin[arg]
            med = float(np.median(delta))
            std = float(delta.std())
            ax_h.bar(centers_bin, hist, width=(edges[1]-edges[0]),
                     color='lightgray', edgecolor='none', label='raw hist')
            ax_h.plot(centers_bin, hist_smooth, color='tab:blue', lw=1.5,
                      label=f'smoothed (σ={smooth_sigma})')
            ax_h.axvline(peak_val,       color='tab:blue',   ls='--', lw=1,
                         label=f'peak={peak_val:+.3f}')
            ax_h.axvline(med,            color='black',      ls=':',  lw=1,
                         label=f'median={med:+.3f}')
            ax_h.axvline(expected_delta, color='orange',     ls=':',  lw=1,
                         label=f'target={expected_delta:.2f}')
            ax_h.axvline(0,              color='red',        ls='--', lw=0.8,
                         alpha=0.5)
            ax_h.set_xlabel('mouse_y − terrain_y')
            ax_h.set_ylabel('# frames')
            ax_h.set_title(f'offset (ox={ox:+d}, oz={oz:+d})   std={std:.3f}')
            ax_h.legend(loc='upper right', fontsize=7)
        # blank any unused subplots
        for k in range(n_pairs, n_rows * n_cols):
            r, c = divmod(k, n_cols)
            axes_h[r, c].axis('off')
        plt.tight_layout()
        plt.show()

    return {'offsets_x': ox_arr, 'offsets_z': oz_arr,
            'peak_loc': peak_loc, 'std': std_2d, 'medians': med_2d,
            'best_ox_std': best_ox_std, 'best_oz_std': best_oz_std,
            'best_ox_peak': best_ox_peak, 'best_oz_peak': best_oz_peak,
            'best_ox_med': best_ox_med, 'best_oz_med': best_oz_med}


def f_diag_chunk0_offset_sweep_2d_bilinear(terrain_data, mov_data, max_frames=5000,
                                             center=62, half_range=3, flip_z=True,
                                             offsets_x=None, offsets_z=None,
                                             hist_bins=200, hist_range=None,
                                             smooth_sigma=2.0,
                                             expected_delta=0.4,
                                             plot_hist_offsets=None):
    # Bilinear-interpolation version of f_diag_chunk0_offset_sweep_2d.
    # Instead of nearest-neighbor lookup at each mouse position, the
    # terrain height is bilinearly interpolated from the 4 cells of the
    # heightmap square the mouse currently sits in. Same 3-panel output
    # (peak_loc, std, median) so you can read the optimum offset directly.
    #
    # Parameters mirror the NN version. See f_diag_chunk0_mouse_height_bilinear
    # for the per-frame math.

    c0 = terrain_data[(terrain_data['ChunkPosX'] == 0) & (terrain_data['ChunkPosZ'] == 0)]
    if len(c0) == 0:
        print('no chunk-0 data')
        return None
    H_grid = c0.pivot(index='x', columns='z', values='height').values.astype(np.float64)
    n_x, n_z = H_grid.shape
    cell_max_x = n_x - 1
    cell_max_z = n_z - 1

    mx_all = np.asarray(mov_data['x_pos'], dtype=np.float64)
    my_all = np.asarray(mov_data['y_pos'], dtype=np.float64)
    mz_all = np.asarray(mov_data['z_pos'], dtype=np.float64)
    n = min(int(max_frames), len(mx_all))
    mx = mx_all[:n]; my = my_all[:n]; mz = mz_all[:n]

    if offsets_x is None:
        offsets_x = np.arange(-int(half_range), int(half_range) + 1)
    if offsets_z is None:
        offsets_z = np.arange(-int(half_range), int(half_range) + 1)
    ox_arr = np.asarray(list(offsets_x))
    oz_arr = np.asarray(list(offsets_z))
    n_xo = len(ox_arr); n_zo = len(oz_arr)
    peak_loc = np.zeros((n_zo, n_xo))
    std_2d   = np.zeros((n_zo, n_xo))
    med_2d   = np.zeros((n_zo, n_xo))

    eps = 1e-6

    def _bilinear_delta(ox, oz):
        cx_f = mx + (center + ox)
        if flip_z:
            cz_f = (center + oz) - mz
        else:
            cz_f = mz + (center + oz)
        cx_c = np.clip(cx_f, 0.0, cell_max_x - eps)
        cz_c = np.clip(cz_f, 0.0, cell_max_z - eps)
        ix_lo = np.floor(cx_c).astype(np.int64)
        iz_lo = np.floor(cz_c).astype(np.int64)
        ix_hi = ix_lo + 1
        iz_hi = iz_lo + 1
        fx = cx_c - ix_lo
        fz = cz_c - iz_lo
        H_ll = H_grid[ix_lo, iz_lo]
        H_hl = H_grid[ix_hi, iz_lo]
        H_lh = H_grid[ix_lo, iz_hi]
        H_hh = H_grid[ix_hi, iz_hi]
        terr_y = (H_ll * (1 - fx) * (1 - fz)
                  + H_hl * fx       * (1 - fz)
                  + H_lh * (1 - fx) * fz
                  + H_hh * fx       * fz)
        return my - terr_y

    print(f'2D BILINEAR sweep: {n_xo} x_offsets x {n_zo} z_offsets = '
          f'{n_xo*n_zo} combinations ({n} mouse frames)...')
    for iz, oz in enumerate(oz_arr):
        for ix, ox in enumerate(ox_arr):
            delta = _bilinear_delta(int(ox), int(oz))
            hist, edges = np.histogram(delta, bins=hist_bins, range=hist_range)
            if smooth_sigma and smooth_sigma > 0:
                hist = sc.ndimage.gaussian_filter1d(hist.astype(np.float64),
                                                     sigma=float(smooth_sigma))
            arg = int(np.argmax(hist))
            peak_loc[iz, ix] = 0.5 * (edges[arg] + edges[arg + 1])
            std_2d[iz, ix]   = delta.std()
            med_2d[iz, ix]   = np.median(delta)

    iz_s, ix_s = np.unravel_index(int(np.argmin(std_2d)), std_2d.shape)
    iz_p, ix_p = np.unravel_index(int(np.argmin(np.abs(peak_loc - expected_delta))),
                                    peak_loc.shape)
    iz_m, ix_m = np.unravel_index(int(np.argmin(np.abs(med_2d - expected_delta))),
                                    med_2d.shape)
    best_ox_std,  best_oz_std  = int(ox_arr[ix_s]), int(oz_arr[iz_s])
    best_ox_peak, best_oz_peak = int(ox_arr[ix_p]), int(oz_arr[iz_p])
    best_ox_med,  best_oz_med  = int(ox_arr[ix_m]), int(oz_arr[iz_m])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _half_p = max(abs(peak_loc.max() - expected_delta),
                  abs(peak_loc.min() - expected_delta))
    im0 = axes[0].imshow(peak_loc,
                          origin='lower', aspect='auto', cmap='RdBu_r',
                          vmin=expected_delta - _half_p,
                          vmax=expected_delta + _half_p,
                          extent=[ox_arr[0]-0.5, ox_arr[-1]+0.5,
                                  oz_arr[0]-0.5, oz_arr[-1]+0.5])
    axes[0].set_xticks(ox_arr); axes[0].set_yticks(oz_arr)
    axes[0].axvline(0, color='gray', lw=0.5)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].plot(best_ox_peak, best_oz_peak, marker='*', ms=20,
                 color='lime', markeredgecolor='black',
                 label=f'closest to {expected_delta:.1f} @ ({best_ox_peak:+d},{best_oz_peak:+d})')
    axes[0].set(xlabel=f'x offset (rel. to center={center})',
                 ylabel=f'z offset (rel. to center={center})',
                 title=f'BILINEAR peak_loc (delta at histogram mode; target={expected_delta:.1f})')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.colorbar(im0, ax=axes[0], label='delta peak (mouse_y − terrain_y)')

    im1 = axes[1].imshow(std_2d, origin='lower', aspect='auto', cmap='viridis',
                          extent=[ox_arr[0]-0.5, ox_arr[-1]+0.5,
                                  oz_arr[0]-0.5, oz_arr[-1]+0.5])
    axes[1].set_xticks(ox_arr); axes[1].set_yticks(oz_arr)
    axes[1].axvline(0, color='red', lw=0.5)
    axes[1].axhline(0, color='red', lw=0.5)
    axes[1].plot(best_ox_std, best_oz_std, marker='*', ms=20,
                 color='lime', markeredgecolor='black',
                 label=f'min std @ ({best_ox_std:+d},{best_oz_std:+d})')
    axes[1].set(xlabel=f'x offset (rel. to center={center})',
                 ylabel=f'z offset (rel. to center={center})',
                 title='BILINEAR delta std (lower = tighter)')
    axes[1].legend(loc='upper right', fontsize=8)
    fig.colorbar(im1, ax=axes[1], label='std')

    _half_m = max(abs(med_2d.max() - expected_delta),
                  abs(med_2d.min() - expected_delta))
    im2 = axes[2].imshow(med_2d, origin='lower', aspect='auto', cmap='RdBu_r',
                          vmin=expected_delta - _half_m,
                          vmax=expected_delta + _half_m,
                          extent=[ox_arr[0]-0.5, ox_arr[-1]+0.5,
                                  oz_arr[0]-0.5, oz_arr[-1]+0.5])
    axes[2].set_xticks(ox_arr); axes[2].set_yticks(oz_arr)
    axes[2].axvline(0, color='gray', lw=0.5)
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].plot(best_ox_med, best_oz_med, marker='*', ms=20,
                 color='lime', markeredgecolor='black',
                 label=f'closest to {expected_delta:.1f} @ ({best_ox_med:+d},{best_oz_med:+d})')
    axes[2].set(xlabel=f'x offset (rel. to center={center})',
                 ylabel=f'z offset (rel. to center={center})',
                 title=f'BILINEAR median (50%; target={expected_delta:.1f})')
    axes[2].legend(loc='upper right', fontsize=8)
    fig.colorbar(im2, ax=axes[2], label='delta median (mouse_y − terrain_y)')

    plt.tight_layout()
    plt.show()

    print(f'\nbest (ox, oz) by min std        : ({best_ox_std:+d}, {best_oz_std:+d})  '
          f'std={std_2d[iz_s, ix_s]:.3f}  peak_loc={peak_loc[iz_s, ix_s]:+.3f}  '
          f'med={med_2d[iz_s, ix_s]:+.3f}')
    print(f'best (ox, oz) by |peak−target|  : ({best_ox_peak:+d}, {best_oz_peak:+d})  '
          f'std={std_2d[iz_p, ix_p]:.3f}  peak_loc={peak_loc[iz_p, ix_p]:+.3f}  '
          f'med={med_2d[iz_p, ix_p]:+.3f}')
    print(f'best (ox, oz) by |median−target|: ({best_ox_med:+d}, {best_oz_med:+d})  '
          f'std={std_2d[iz_m, ix_m]:.3f}  peak_loc={peak_loc[iz_m, ix_m]:+.3f}  '
          f'med={med_2d[iz_m, ix_m]:+.3f}')

    if plot_hist_offsets is not None and len(list(plot_hist_offsets)) > 0:
        pairs = list(plot_hist_offsets)
        n_pairs = len(pairs)
        n_cols = min(n_pairs, 4)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        fig_h, axes_h = plt.subplots(n_rows, n_cols,
                                      figsize=(4 * n_cols, 3 * n_rows),
                                      squeeze=False)
        for idx, (ox, oz) in enumerate(pairs):
            r, c = divmod(idx, n_cols)
            ax_h = axes_h[r, c]
            delta = _bilinear_delta(int(ox), int(oz))
            hist, edges = np.histogram(delta, bins=hist_bins, range=hist_range)
            hist_smooth = hist.astype(np.float64)
            if smooth_sigma and smooth_sigma > 0:
                hist_smooth = sc.ndimage.gaussian_filter1d(hist_smooth,
                                                            sigma=float(smooth_sigma))
            centers_bin = 0.5 * (edges[:-1] + edges[1:])
            arg = int(np.argmax(hist_smooth))
            peak_val = centers_bin[arg]
            med = float(np.median(delta))
            std = float(delta.std())
            ax_h.bar(centers_bin, hist, width=(edges[1]-edges[0]),
                     color='lightgray', edgecolor='none', label='raw hist')
            ax_h.plot(centers_bin, hist_smooth, color='tab:blue', lw=1.5,
                      label=f'smoothed (σ={smooth_sigma})')
            ax_h.axvline(peak_val,       color='tab:blue', ls='--', lw=1,
                         label=f'peak={peak_val:+.3f}')
            ax_h.axvline(med,            color='black',    ls=':',  lw=1,
                         label=f'median={med:+.3f}')
            ax_h.axvline(expected_delta, color='orange',   ls=':',  lw=1,
                         label=f'target={expected_delta:.2f}')
            ax_h.axvline(0,              color='red',      ls='--', lw=0.8,
                         alpha=0.5)
            ax_h.set_xlabel('mouse_y − terrain_y (bilinear)')
            ax_h.set_ylabel('# frames')
            ax_h.set_title(f'offset (ox={ox:+d}, oz={oz:+d})   std={std:.3f}')
            ax_h.legend(loc='upper right', fontsize=7)
        for k in range(n_pairs, n_rows * n_cols):
            r, c = divmod(k, n_cols)
            axes_h[r, c].axis('off')
        plt.tight_layout()
        plt.show()

    return {'offsets_x': ox_arr, 'offsets_z': oz_arr,
            'peak_loc': peak_loc, 'std': std_2d, 'medians': med_2d,
            'best_ox_std': best_ox_std, 'best_oz_std': best_oz_std,
            'best_ox_peak': best_ox_peak, 'best_oz_peak': best_oz_peak,
            'best_ox_med': best_ox_med, 'best_oz_med': best_oz_med}

def f_add_orientation_stripes(frames, top_intensity=255, bottom_intensity=128,
                                stripe_width=3, in_place=False):
    # Paint two debug stripes onto a (T, vert, lat) frames array so the
    # vertical orientation in default image viewers can be read at a glance.
    #   top of array (rows 2..2+stripe_width)         : top_intensity (default 255)
    #   bottom of array (rows -1-stripe_width..-1)    : bottom_intensity (default 128)
    # In default ImageJ / matplotlib display (row 0 at top of screen):
    #   - bright stripe at top of display  → small pix = top of display (standard)
    #   - bright stripe at bottom          → image was flipped vs. array order
    # Returns the (in-place modified or copied) array.
    if not in_place:
        frames = frames.copy()
    frames[:,  2:2+stripe_width,  :] = top_intensity
    frames[:, -1-stripe_width:-1, :] = bottom_intensity
    return frames


def f_plot_monitor_frame(two_mon_frames, num_samp, n_fr=0,
                          show_mon_labels=True, show_horizon=True,
                          title=None, ax=None, figsize=(10, 5)):
    # Quick-look matplotlib plot of one frame from a two-monitor stack
    # (left + right concatenated along the lat axis). Adds a cyan vertical
    # divider and LEFT / RIGHT text labels at the top of each half.
    # show_horizon : draw a horizontal dashed line at the camera horizon
    #     (vert_pix = (vert_samp-1)/2, i.e. vert_angle = 0). Useful for
    #     checking that the terrain horizon and object centers fall on the
    #     expected side — with the current vert mapping (negated), the
    #     terrain horizon should sit at this line; terrain below camera
    #     should appear BELOW it (= larger row index = closer to bottom
    #     of display in default origin='upper').
    # If orientation debug stripes were painted into the array via
    # f_add_orientation_stripes earlier, they'll show up as bright / gray
    # bands at the top / bottom of the image.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(two_mon_frames[n_fr], cmap='gray', vmin=0, vmax=255,
              interpolation='nearest', aspect='equal')
    if show_mon_labels:
        ax.axvline(num_samp - 0.5, color='cyan', lw=1.5, alpha=0.7)
        ax.text(num_samp * 0.25, num_samp * 0.05, 'LEFT', color='cyan',
                fontsize=18, ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='cyan'))
        ax.text(num_samp * 1.25, num_samp * 0.05, 'RIGHT', color='cyan',
                fontsize=18, ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='cyan'))
    if show_horizon:
        horizon_row = (two_mon_frames.shape[1] - 1) / 2.0
        ax.axhline(horizon_row, color='orange', lw=1.2, ls='--', alpha=0.7)
        ax.text(2, horizon_row - 1, 'horizon (vert_angle=0)',
                color='orange', fontsize=10, ha='left', va='bottom')
    if title is None:
        title = f'frame {n_fr} — two_mon_frames[{n_fr}] shape={two_mon_frames[n_fr].shape}'
    ax.set_title(title)
    ax.set_xlabel('lat pix (col)  —  left mon | right mon')
    ax.set_ylabel('vert pix (row)')
    return fig, ax


def f_plot_obj_terrain_heights(two_mon_frames, vec_data_l, vec_data_r,
                                obj_data, terrain_data, cam_params,
                                num_samp, n_fr=0, terr_kw=None,
                                show_mon_labels=True, ax=None,
                                figsize=(12, 6),
                                label_color='red', label_bg='yellow',
                                label_fontsize=16):
    # For each object visible in frame `n_fr` on either monitor, look up
    # the terrain height at the object's (ObjLocX, ObjLocZ) and overlay
    # the value as a colored label on the rendered frame. Useful for
    # checking placement: labels should land on the rendered cylinder
    # silhouettes.
    #
    # Pixel mapping below MUST match f_angles_to_movie_v2 (otherwise the
    # labels drift off the silhouettes). Update both together if the
    # convention changes.
    #
    # two_mon_frames : (T, vert, lat_l + lat_r) uint8 — left+right concat.
    # vec_data_l / vec_data_r : dicts from f_get_monitor_coords for the
    #     two monitors. Must include obj_used, obj_mon_idx, obj_lat_angle,
    #     obj_vert_angle.
    # obj_data       : DataFrame with ObjLocX, ObjLocY, ObjLocZ.
    # terrain_data   : DataFrame with ChunkPosX, ChunkPosZ, x, z, height.
    # cam_params     : dict (uses hFOV_rad, vFOV_rad).
    # num_samp       : per-monitor lat / vert sample count (assumed square).
    # n_fr           : frame index to display.
    # terr_kw        : kwargs forwarded to f_terrain_world_coords
    #                  (chunk_pitch, chunk_centered, flip_x/z, swap_xz).
    if terr_kw is None:
        terr_kw = {}

    terr_w = f_terrain_world_coords(terrain_data, **terr_kw)
    xz_tree = cKDTree(np.column_stack([terr_w['tx'], terr_w['tz']]))
    obj_xz = np.column_stack([obj_data['ObjLocX'].to_numpy(),
                              obj_data['ObjLocZ'].to_numpy()])
    _, kd_idx = xz_tree.query(obj_xz, k=1)
    terrain_h_at_obj = terr_w['ty'][kd_idx]   # indexed by global object id

    def _proj(vec_data):
        out = []
        for k, glob_idx in enumerate(vec_data['obj_used']):
            if not vec_data['obj_mon_idx'][n_fr, k]:
                continue
            lat_a  = vec_data['obj_lat_angle'][n_fr, k]
            vert_a = vec_data['obj_vert_angle'][n_fr, k]
            lat_pix  = (-lat_a  + cam_params['hFOV_rad']/2) / cam_params['hFOV_rad'] * (num_samp - 1)
            vert_pix = (-vert_a + cam_params['vFOV_rad']/2) / cam_params['vFOV_rad'] * (num_samp - 1)
            out.append((lat_pix, vert_pix, terrain_h_at_obj[glob_idx]))
        return out

    left_lbl  = _proj(vec_data_l)
    right_lbl = _proj(vec_data_r)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(two_mon_frames[n_fr], cmap='gray', vmin=0, vmax=255,
              interpolation='nearest', aspect='equal')
    if show_mon_labels:
        ax.axvline(num_samp - 0.5, color='cyan', lw=1.5, alpha=0.7)
        ax.text(num_samp * 0.25, num_samp * 0.05, 'LEFT', color='cyan',
                fontsize=18, ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='cyan'))
        ax.text(num_samp * 1.25, num_samp * 0.05, 'RIGHT', color='cyan',
                fontsize=18, ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='cyan'))
    for (lat_p, vert_p, th) in left_lbl:
        ax.text(lat_p, vert_p, f'{th:.1f}',
                color=label_color, fontsize=label_fontsize,
                ha='center', va='center',
                bbox=dict(facecolor=label_bg, alpha=0.7, edgecolor='none'))
    for (lat_p, vert_p, th) in right_lbl:
        ax.text(lat_p + num_samp, vert_p, f'{th:.1f}',
                color=label_color, fontsize=label_fontsize,
                ha='center', va='center',
                bbox=dict(facecolor=label_bg, alpha=0.7, edgecolor='none'))
    ax.set_title(f'frame {n_fr} — labels = terrain height at each visible object')
    return fig, ax


def f_plot_monitor_outline(mouse_xyz, mon_phi, mon_theta, cam_params, axis=None, color_cent = 'gray', color_edge = 'blue'):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis
    
    x_pt = 0
    z_pt = 2

    mon_dir_cart = f_spheric_to_cart(mon_phi, mon_theta)
    mon_edge_r = f_spheric_to_cart(f_add_phase(mon_phi, cam_params['hFOV_rad']/2), mon_theta)
    mon_edge_l = f_spheric_to_cart(f_add_phase(mon_phi, -cam_params['hFOV_rad']/2), mon_theta)
    
    h_adj = cam_params['clip_len']
    
    ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mon_dir_cart[x_pt]*cam_params['clip_len']], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mon_dir_cart[z_pt]*cam_params['clip_len']], 'o-', color=color_cent)
    ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mon_edge_r[x_pt]*h_adj], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mon_edge_r[z_pt]*h_adj], 'o-', color=color_edge)
    ax1.plot([mouse_xyz[x_pt], mouse_xyz[x_pt] + mon_edge_l[x_pt]*h_adj], [mouse_xyz[z_pt], mouse_xyz[z_pt] + mon_edge_l[z_pt]*h_adj], 'o-', color=color_edge)

def f_plot_lateral_over_time(ovj_vec_data, time, axis=None, ylabel=True, xlabel=True):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis
    
    for n_obj in range(ovj_vec_data['obj_used'].shape[0]):
        in_fov_idx = ovj_vec_data['obj_mon_idx'][:,n_obj].astype(bool)
        ax1.plot(time[in_fov_idx], ovj_vec_data['obj_lat_angle'][:,n_obj][in_fov_idx], '.')
    
    if xlabel:
        ax1.set_xlabel('time (sec)')
    if ylabel:
        ax1.set_ylabel('lateral') 

def f_plot_vertical_over_time(ovj_vec_data, time, cam_params, axis=None, ylabel=True, xlabel=True):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis
    
    for n_obj in range(ovj_vec_data['obj_used'].shape[0]):
        in_fov_idx = ovj_vec_data['obj_mon_idx'][:,n_obj].astype(bool)
        ax1.plot(time[in_fov_idx], ovj_vec_data['obj_vert_angle'][:,n_obj][in_fov_idx], '.')
    
    cam_params = f_comp_FOV_adj(cam_params)
    ax1.set_ylim([-cam_params['vFOV_rad']/2*1.1, cam_params['vFOV_rad']/2*1.1])
    if xlabel:
        ax1.set_xlabel('time (sec)')
    if ylabel:
        ax1.set_ylabel('vertical')
    
def f_plot_dist_over_time(ovj_vec_data, time, axis=None, ylabel=True, xlabel=True):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis
    
    for n_obj in range(ovj_vec_data['obj_used'].shape[0]):
        in_fov_idx = ovj_vec_data['obj_mon_idx'][:,n_obj].astype(bool)
        ax1.plot(time[in_fov_idx], ovj_vec_data['obj_dist'][:,n_obj][in_fov_idx], '.')
    
    if xlabel:
        ax1.set_xlabel('time (sec)')
    if ylabel:
        ax1.set_ylabel('distance')

def f_plot_lateral_over_time2(mouse_xyz, mon_phi, mon_theta, obj_locs, time, cam_params, axis=None):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis

    for n_pt3 in range(obj_locs.shape[0]):
        spher_vec_objs = f_cart_to_spheric_np(obj_locs[n_pt3,:] - mouse_xyz)
        lat_angle = f_add_phase(mon_phi, - spher_vec_objs[:,2])
        vert_angle = f_add_phase(mon_theta, - spher_vec_objs[:,1])
        
        in_fov_idx = (spher_vec_objs[:,0] < cam_params['clip_len']) & (np.abs(lat_angle) < cam_params['hFOV_rad']/2) & (np.abs(vert_angle) < cam_params['vFOV_rad']/2)
        if np.sum(in_fov_idx):
            ax1.plot(time[in_fov_idx], lat_angle[in_fov_idx], '.')
    #plt.ylim([-FOV_rad_adj/2, FOV_rad_adj/2])
    #ax1.set_title('lateral translation over time')
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('lateral') 
   
def f_plot_vertical_over_time2(mouse_xyz, mon_phi, mon_theta, obj_locs, time, cam_params, axis=None):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis

    for n_pt3 in range(obj_locs.shape[0]):
        spher_vec_objs = f_cart_to_spheric_np(obj_locs[n_pt3,:] - mouse_xyz)
        spher_vec_objs[:,2] = spher_vec_objs[:,2]%(2*np.pi)
        
        in_fov_idx = (spher_vec_objs[:,0] < cam_params['clip_len']) & (np.abs(mon_phi - spher_vec_objs[:,2]) < cam_params['hFOV_rad']/2) & (np.abs(mon_theta - spher_vec_objs[:,1]) < cam_params['vFOV_rad']/2)
        if np.sum(in_fov_idx):
            ax1.plot(time[in_fov_idx], mon_theta[in_fov_idx] - spher_vec_objs[in_fov_idx,1], '.')
    ax1.set_ylim([-cam_params['vFOV_rad']/2*1.1, cam_params['vFOV_rad']/2*1.1])
    #ax1.set_title('vertical translation over time')
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('vertical')
    
    
def f_plot_dist_over_time2(mouse_xyz, mon_phi, mon_theta, obj_locs, time, cam_params, axis=None):
    if axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = axis

    for n_pt3 in range(obj_locs.shape[0]):
        spher_vec_objs = f_cart_to_spheric_np(obj_locs[n_pt3,:] - mouse_xyz)
        spher_vec_objs[:,2] = spher_vec_objs[:,2]%(2*np.pi)
        
        in_fov_idx = (spher_vec_objs[:,0] < cam_params['clip_len']) & (np.abs(mon_phi - spher_vec_objs[:,2]) < cam_params['hFOV_rad']/2) & (np.abs(mon_theta - spher_vec_objs[:,1]) < cam_params['vFOV_rad']/2)
        if np.sum(in_fov_idx):
            ax1.plot(time[in_fov_idx], spher_vec_objs[in_fov_idx,0], '.')
    #plt.ylim([-FOV_rad_adj/2, FOV_rad_adj/2])
    #ax1.set_title('object distance over time')
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('distance')
