# -*- coding: utf-8 -*-
"""
Feature-engineering helpers for VR / calcium-imaging analysis scripts.
Generic (not CEBRA-specific) — used by any pipeline that needs to turn
per-frame behavior into decoder-ready feature matrices aligned to the
imaging clock.

Functions:
    f_aggregate_monitor_features         — per-frame collapsed scalars per
                                           monitor (X_agg style; loses identity)
    f_aggregate_monitor_features_per_obj — per-frame per-object scalars
                                           (X_per_obj style; preserves identity)
    f_motion_features                    — per-frame velocities per monitor
                                           (X_mot style; d_lat, d_vert, etc.)
    f_resample_to_imaging                — behavior-clock → imaging-clock interp
    f_zscore                             — column z-score with std=0 guard
    build_feature_blocks                 — orchestrator: build several blocks at
                                           once and return the registry dict

Moved out of f_cebra_helpers.py on 2026-05-21 — these are not CEBRA-specific;
CEBRA-specific helpers (model fits, supervision wrappers) live elsewhere.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def f_aggregate_monitor_features(vec_data, mode='raw', obj_size=None, clip_len=None,
                                  obj_collapse='nearest'):
    # Per-frame summary features for one monitor.
    # When no object is in FOV: all per-side values = 0 (build cell may convert
    # to NaN downstream).
    #
    # `obj_collapse` controls how multiple simultaneously-in-FOV objects are
    # collapsed into a single (lat, vert, dist) per frame:
    #   'nearest'       — pick the single nearest in-FOV obj. Cheap and stable
    #                     but causes step discontinuities when which-object-is-
    #                     nearest changes (e.g. nearest exits FOV, another
    #                     object overtakes it). Other in-FOV objects ignored.
    #   'salience_mean' — angular-size-weighted mean across all in-FOV objects.
    #                     weight_i = 2·arctan2(obj_size_x, dist_i) — i.e. each
    #                     object weighted by how big it looks on the monitor.
    #                     Smooth across object swaps; reports the "visual
    #                     center of mass" rather than a single object's coords.
    #                     Requires obj_size.
    #
    # `mode` controls how the distance channel is parameterized — addresses
    # the "object is far → tiny on monitor" nonlinearity that the raw pix
    # representation gets for free via projection:
    #   'raw'              → [pres, lat, vert, dist]                 (4 dims)
    #   'angular'          → [pres, lat, vert, ang_size]             (4 dims)
    #                        ang_size = 2·arctan2(obj_size_x, dist),
    #                        the monitor angular subtense of the object.
    #   'salience'         → [pres, lat, vert, dist, salience]       (5 dims)
    #                        salience = max(0, 1 − dist/clip_len), a soft
    #                        "how prominent" channel alongside raw dist.
    #   'angular_salience' → [pres, lat, vert, ang_size, salience]   (5 dims)
    obj_mon  = vec_data['obj_mon_idx']        # (T_beh, n_obj) bool
    obj_dist = vec_data['obj_dist']           # (T_beh, n_obj)
    obj_lat  = vec_data['obj_lat_angle']
    obj_vert = vec_data['obj_vert_angle']
    T = obj_mon.shape[0]
    presence = obj_mon.any(axis=1)

    if obj_collapse == 'nearest':
        # mask out-of-FOV distances → +inf so argmin picks an in-FOV obj
        obj_dist_masked = np.where(obj_mon, obj_dist, np.inf)
        nearest = np.argmin(obj_dist_masked, axis=1)
        rows = np.arange(T)
        lat       = np.where(presence, obj_lat [rows, nearest], 0.0)
        vert      = np.where(presence, obj_vert[rows, nearest], 0.0)
        dist_raw  = np.where(presence, obj_dist[rows, nearest], 0.0)
    elif obj_collapse == 'salience_mean':
        if obj_size is None:
            raise ValueError("obj_collapse 'salience_mean' requires obj_size")
        # angular-size weights per obj per frame; zero weight for out-of-FOV
        # objs (and clip dist to avoid div-by-zero on absent objs)
        dist_safe_all = np.maximum(obj_dist, 1e-6)
        w = 2.0 * np.arctan2(obj_size['x'], dist_safe_all)
        w = np.where(obj_mon, w, 0.0)
        w_sum = w.sum(axis=1)
        # weighted mean of (lat, vert, dist) across in-FOV objs.
        # presence=False rows get 0 here; downstream NaN-mask handles them.
        with np.errstate(divide='ignore', invalid='ignore'):
            lat      = np.where(presence, (w * obj_lat ).sum(axis=1) / w_sum, 0.0)
            vert     = np.where(presence, (w * obj_vert).sum(axis=1) / w_sum, 0.0)
            dist_raw = np.where(presence, (w * obj_dist).sum(axis=1) / w_sum, 0.0)
    else:
        raise ValueError(f'unknown obj_collapse: {obj_collapse!r}')

    if mode in ('angular', 'angular_salience'):
        if obj_size is None:
            raise ValueError("mode '%s' requires obj_size" % mode)
        # arctan2(obj_size_x, dist) — clip dist_raw to a small +ve so the
        # formula stays finite where presence==0 (we zero it out below).
        dist_safe = np.where(presence, dist_raw, 1.0)
        dist_chan = np.where(presence,
                             2.0*np.arctan2(obj_size['x'], dist_safe),
                             0.0)
    else:
        dist_chan = dist_raw

    feats = [presence.astype(float), lat, vert, dist_chan]

    if mode in ('salience', 'angular_salience'):
        if clip_len is None:
            raise ValueError("mode '%s' requires clip_len" % mode)
        salience = np.where(presence,
                            np.maximum(0.0, 1.0 - dist_raw/clip_len),
                            0.0)
        feats.append(salience)

    return np.stack(feats, axis=1)   # (T_beh, 4) or (T_beh, 5)


def f_aggregate_monitor_features_per_obj(vec_data, mode='raw', obj_size=None, clip_len=None, total_n_obj=None):
    # Per-object scalar features for one monitor — preserves identity and count
    # instead of collapsing across objects the way f_aggregate_monitor_features
    # does. Output shape (T_beh, n_obj_out, d):
    #   d = 4 ('raw'/'angular')           → [pres, lat, vert, dist_chan]
    #   d = 5 ('salience'/'angular_salience') → adds salience as 5th channel
    # `mode` semantics match f_aggregate_monitor_features.
    #
    # `total_n_obj` controls the channel space:
    #   None  → n_obj_out = vec_data['obj_used'].shape[0]; channels are local to
    #           this monitor (left.ch3 ≠ right.ch3).
    #   N     → n_obj_out = N; channel k = original object id k via
    #           vec_data['obj_used']. Pass len(bh_data[n_dset]['object_data'])
    #           if you want left/right to share a channel space (lets you
    #           stack / compare the same object across monitors).
    # Objects with pres=0 on a frame get lat/vert/dist_chan/salience = 0 —
    # downstream NaN-mask logic handles the empty-frame ambiguity (task #12).
    obj_mon  = vec_data['obj_mon_idx']         # (T_beh, n_obj_local) bool
    obj_dist = vec_data['obj_dist']            # (T_beh, n_obj_local)
    obj_lat  = vec_data['obj_lat_angle']
    obj_vert = vec_data['obj_vert_angle']
    obj_used = np.asarray(vec_data['obj_used'], dtype=int)
    T, n_local = obj_mon.shape

    if total_n_obj is None:
        n_out = n_local
        dest = np.arange(n_local, dtype=int)
    else:
        n_out = int(total_n_obj)
        dest = obj_used

    pres = np.zeros((T, n_out), dtype=bool)
    pres[:, dest] = obj_mon

    lat      = np.zeros((T, n_out), dtype=float)
    vert     = np.zeros((T, n_out), dtype=float)
    dist_raw = np.zeros((T, n_out), dtype=float)
    lat[:, dest]      = np.where(obj_mon, obj_lat,  0.0)
    vert[:, dest]     = np.where(obj_mon, obj_vert, 0.0)
    dist_raw[:, dest] = np.where(obj_mon, obj_dist, 0.0)

    if mode in ('angular', 'angular_salience'):
        if obj_size is None:
            raise ValueError("mode '%s' requires obj_size" % mode)
        dist_safe = np.where(pres, dist_raw, 1.0)
        dist_chan = np.where(pres, 2.0*np.arctan2(obj_size['x'], dist_safe), 0.0)
    else:
        dist_chan = dist_raw

    chans = [pres.astype(float), lat, vert, dist_chan]

    if mode in ('salience', 'angular_salience'):
        if clip_len is None:
            raise ValueError("mode '%s' requires clip_len" % mode)
        salience = np.where(pres, np.maximum(0.0, 1.0 - dist_raw/clip_len), 0.0)
        chans.append(salience)

    return np.stack(chans, axis=-1)   # (T_beh, n_obj_out, d)


def f_motion_features(vec_data, mode='raw', obj_size=None, clip_len=None,
                       obj_collapse='salience_mean', smooth_sigma=2.0, beh_dt=0.1,
                       beh_t=None):
    # Per-frame velocity features for one monitor — time derivatives of the
    # nearest-or-salience-weighted-mean position channels. Companion to
    # f_aggregate_monitor_features (which gives positions).
    #
    # Output (T_beh, d) where d = 6 (raw/salience mode) or 7 (angular mode):
    #   0: d_lat        — horizontal screen velocity (rad/s)
    #   1: d_vert       — vertical screen velocity (rad/s)
    #   2: d_dist       — depth velocity (units/s)
    #  [3: d_ang_size]  — angular-size growth rate (rad/s) — looming-cell
    #                     correlate; only present when mode in 'angular' or
    #                     'angular_salience'.
    #   next: motion_mag      — sqrt(d_lat² + d_vert²) (rad/s)
    #   next: motion_dir_sin  — sin(arctan2(d_vert, d_lat))  ∈ [-1, 1]
    #   next: motion_dir_cos  — cos(arctan2(d_vert, d_lat))  ∈ [-1, 1]
    #
    # Direction is emitted as a sin/cos PAIR so linear decoders can read
    # circular structure without the ±π wrap discontinuity.
    #
    # `obj_collapse`:
    #   'salience_mean' (default) — derivative of angular-size-weighted-mean
    #                               position; smooth across object swaps.
    #                               Recommended.
    #   'nearest'                 — derivative of nearest-object trace;
    #                               causes one-frame velocity SPIKES at
    #                               object swaps. Those samples are
    #                               NaN-masked (extended by smooth_sigma
    #                               window). A warning is emitted.
    #
    # `smooth_sigma`: Gaussian smoothing of lat/vert/dist (behavior samples)
    # before differentiating. Reduces noise. 0 disables smoothing.
    #
    # `beh_dt`: behavior sampling interval (s). Converts per-sample gradient
    # to per-second velocity. With mov_data['time'] from f_proc_movement
    # (interp_step=0.1), this is 0.1.
    #
    # NaN where presence=0: this function returns 0 for absent frames; build
    # cell can NaN-mask post-resampling, paralleling the X_agg pattern.
    obj_mon  = vec_data['obj_mon_idx']        # (T_beh, n_obj) bool
    obj_dist = vec_data['obj_dist']
    obj_lat  = vec_data['obj_lat_angle']
    obj_vert = vec_data['obj_vert_angle']
    T = obj_mon.shape[0]
    presence = obj_mon.any(axis=1)

    # Compute (lat, vert, dist_raw) using same logic as f_aggregate_monitor_features
    nearest = None    # only set under 'nearest' (used for swap detection)
    if obj_collapse == 'nearest':
        warnings.warn("f_motion_features: obj_collapse='nearest' causes velocity "
                      "spikes at object swaps; those samples are NaN-masked. "
                      "Consider obj_collapse='salience_mean' for smoother motion.")
        obj_dist_masked = np.where(obj_mon, obj_dist, np.inf)
        nearest = np.argmin(obj_dist_masked, axis=1)
        rows = np.arange(T)
        lat       = np.where(presence, obj_lat [rows, nearest], 0.0)
        vert      = np.where(presence, obj_vert[rows, nearest], 0.0)
        dist_raw  = np.where(presence, obj_dist[rows, nearest], 0.0)
    elif obj_collapse == 'salience_mean':
        if obj_size is None:
            raise ValueError("obj_collapse='salience_mean' requires obj_size")
        dist_safe_all = np.maximum(obj_dist, 1e-6)
        w = 2.0 * np.arctan2(obj_size['x'], dist_safe_all)
        w = np.where(obj_mon, w, 0.0)
        w_sum = w.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lat      = np.where(presence, (w * obj_lat ).sum(axis=1) / w_sum, 0.0)
            vert     = np.where(presence, (w * obj_vert).sum(axis=1) / w_sum, 0.0)
            dist_raw = np.where(presence, (w * obj_dist).sum(axis=1) / w_sum, 0.0)
    else:
        raise ValueError(f'unknown obj_collapse: {obj_collapse!r}')

    # Smooth before differentiating (reduces high-frequency noise).
    if smooth_sigma and smooth_sigma > 0:
        lat_s      = gaussian_filter1d(lat,      smooth_sigma, mode='nearest')
        vert_s     = gaussian_filter1d(vert,     smooth_sigma, mode='nearest')
        dist_raw_s = gaussian_filter1d(dist_raw, smooth_sigma, mode='nearest')
    else:
        lat_s, vert_s, dist_raw_s = lat, vert, dist_raw

    # Velocities (per-second). np.gradient(x, beh_t) is exact on non-uniform grids.
    if beh_t is not None:
        d_lat  = np.gradient(lat_s,      beh_t)
        d_vert = np.gradient(vert_s,     beh_t)
        d_dist = np.gradient(dist_raw_s, beh_t)
    else:
        d_lat  = np.gradient(lat_s)      / beh_dt
        d_vert = np.gradient(vert_s)     / beh_dt
        d_dist = np.gradient(dist_raw_s) / beh_dt

    chans = [d_lat, d_vert, d_dist]

    if mode in ('angular', 'angular_salience'):
        if obj_size is None:
            raise ValueError(f"mode '{mode}' requires obj_size")
        dist_safe = np.maximum(dist_raw_s, 1e-6)
        ang_size = 2.0 * np.arctan2(obj_size['x'], dist_safe)
        d_ang_size = (np.gradient(ang_size, beh_t) if beh_t is not None
                      else np.gradient(ang_size) / beh_dt)
        chans.append(d_ang_size)

    # Polar motion: magnitude + direction as sin/cos pair (linear-decoder-safe).
    motion_mag     = np.sqrt(d_lat**2 + d_vert**2)
    motion_dir     = np.arctan2(d_vert, d_lat)
    motion_dir_sin = np.sin(motion_dir)
    motion_dir_cos = np.cos(motion_dir)
    chans.extend([motion_mag, motion_dir_sin, motion_dir_cos])

    out = np.stack(chans, axis=1)   # (T_beh, d)

    # Zero out absent frames (build cell NaN-masks post-resampling for parallelism
    # with X_agg).
    out = np.where(presence[:, None], out, 0.0)

    # Object-swap masking — only in 'nearest' mode.
    if nearest is not None:
        swap = np.zeros(T, dtype=bool)
        swap[1:] = (nearest[1:] != nearest[:-1]) & presence[1:] & presence[:-1]
        # Smoothing spreads the spike; mask a window of half-width ≈ 2σ around each swap.
        half_w = int(np.ceil(2 * smooth_sigma)) + 1 if smooth_sigma else 1
        mask_idx = np.zeros(T, dtype=bool)
        for k in range(-half_w, half_w + 1):
            shifted = np.roll(swap, k)
            if k > 0:
                shifted[:k] = False
            elif k < 0:
                shifted[k:] = False
            mask_idx |= shifted
        out[mask_idx, :] = np.nan

    return out


def f_lick_reward_features(lr_data, smooth_sigma=0.0):
    # Mouse lick + reward event traces, on the behavior clock.
    #
    # Input
    # -----
    # lr_data : dict from f_proc_lick_rew with keys 'lick_trace', 'rew_trace'.
    #     Each is (T_beh,) integer event counts per behavior frame (mostly
    #     0, +1 at event times, occasionally +2 if events bin into same frame).
    #
    # smooth_sigma : Gaussian σ (behavior samples) applied to each trace.
    #     0 (default) → raw event counts. Each event is a single +1 spike at
    #         its frame. Decoder sees a sparse signal — good for "is there a
    #         lick at THIS frame?" decoding, hard for low-temporal-precision
    #         decoders.
    #     >0 → continuous rate-like signal. e.g. σ=5 at 10 Hz behavior clock
    #         ≈ 500 ms smoothing → "lick rate over a half-second window".
    #         Easier target for Ridge-style decoders; loses single-event timing.
    #
    # Output (T_beh, 2)
    #   col 0: lick   — count or smoothed-rate per behavior frame
    #   col 1: rew    — count or smoothed-rate per behavior frame
    #
    # No NaN-masking — events are defined for every behavior frame (0 when no
    # event). Parallels the self_mot block in being "mouse-level" (not per
    # monitor); side_layout=False at the registry level.
    lick = np.asarray(lr_data['lick_trace'], dtype=np.float32)
    rew  = np.asarray(lr_data['rew_trace'],  dtype=np.float32)

    if smooth_sigma and smooth_sigma > 0:
        lick = gaussian_filter1d(lick, smooth_sigma, mode='nearest')
        rew  = gaussian_filter1d(rew,  smooth_sigma, mode='nearest')

    return np.stack([lick, rew], axis=1)   # (T_beh, 2)


def f_self_motion_features(mov_data, smooth_sigma=2.0, beh_dt=0.1, beh_t=None):
    # Mouse-self motion (world-frame). Independent of any monitor / object —
    # captures vestibular / head-direction-style signals that visual-motion
    # channels miss.
    #
    # Output (T_beh, 2):
    #   0: self_d_dist — linear speed of mouse in xz plane         (units/s)
    #                    = √((dx/dt)² + (dz/dt)²); always ≥ 0
    #   1: self_d_phi  — yaw angular velocity                      (rad/s)
    #                    np.unwrap-d before differentiating to avoid spikes
    #                    at ±π wraps.
    #
    # `smooth_sigma`: Gaussian σ (behavior samples) applied to x_pos/z_pos/phi
    # before differentiating. 0 disables smoothing.
    # `beh_dt`: behavior sampling interval (s). Default 0.1 = 10 Hz.
    #
    # No NaN-masking: the mouse is always present, so these channels are
    # defined for every behavior frame.
    x   = np.asarray(mov_data['x_pos'], dtype=float)
    z   = np.asarray(mov_data['z_pos'], dtype=float)
    phi = np.asarray(mov_data['phi'],   dtype=float)

    if smooth_sigma and smooth_sigma > 0:
        x   = gaussian_filter1d(x,   smooth_sigma, mode='nearest')
        z   = gaussian_filter1d(z,   smooth_sigma, mode='nearest')
        # Unwrap before smoothing so ±π discontinuities don't get smeared.
        phi = gaussian_filter1d(np.unwrap(phi), smooth_sigma, mode='nearest')
    else:
        phi = np.unwrap(phi)

    if beh_t is not None:
        dx = np.gradient(x, beh_t)
        dz = np.gradient(z, beh_t)
        d_phi = np.gradient(phi, beh_t)
    else:
        dx = np.gradient(x) / beh_dt
        dz = np.gradient(z) / beh_dt
        d_phi = np.gradient(phi) / beh_dt
    speed = np.sqrt(dx**2 + dz**2)

    return np.stack([speed, d_phi], axis=1)   # (T_beh, 2)


def f_resample_to_imaging(beh_t, beh_feats, frame_t, pulse_delay, kind='linear'):
    # Map behavior clock to imaging clock and interpolate features to frame_t.
    # `beh_feats` can be (T_beh,) or (T_beh, d). Out-of-range frames get filled
    # with 0 (e.g. before first or after last behavior sample).
    beh_t_aligned = np.asarray(beh_t) - pulse_delay
    fi = interp1d(beh_t_aligned, beh_feats, axis=0, kind=kind,
                  bounds_error=False, fill_value=0.0, assume_sorted=False)
    return fi(np.asarray(frame_t))


def f_zscore(M):
    # Column-wise z-score with std==0 guard.
    s = M.std(axis=0, keepdims=True)
    s[s == 0] = 1
    return (M - M.mean(axis=0, keepdims=True)) / s


def f_detrend_col(y, sigma_frames):
    """NaN-aware high-pass: subtract a normalized-Gaussian-smoothed copy.

    Removes session-spanning drift (anything slower than ~sigma_frames) while
    keeping fast structure, so a blocked-CV decoder can't latch onto slow
    nonstationarity that doesn't generalize across time blocks (task #8/#10).
    NaNs in `y` are preserved as NaN in the output (masked frames stay masked).

    Parameters
    ----------
    y : ndarray (T,)
        Single target column. May contain NaN.
    sigma_frames : float or None
        Gaussian σ in samples. Falsy / <=0 → returns `y` unchanged.

    Returns
    -------
    ndarray (T,) : y minus its low-pass trend.
    """
    if not sigma_frames or sigma_frames <= 0:
        return y
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return y
    yf  = np.where(mask, y, 0.0).astype(float)
    m   = mask.astype(float)
    num = gaussian_filter1d(yf, sigma_frames, mode='nearest')
    den = gaussian_filter1d(m,  sigma_frames, mode='nearest')
    trend = np.where(den > 1e-6, num / den, 0.0)
    return y - trend     # NaN where y is NaN


def f_resolve_detrend_sigma(block_name, detrend_blocks, detrend_sigma_sec, fs):
    """Resolve the high-pass σ (in FRAMES) to use for one feature block.

    Lets a caller detrend SOME blocks and not others — important here because
    slow behavioral targets (agg position/distance/angle) carry real signal in
    their slow component, while pix-PC1/2's slow component is session-spanning
    drift (the #8 confound). A global detrend can't treat them differently;
    this resolver does.

    Parameters
    ----------
    block_name : str
        Registry block name of the column(s) being detrended ('agg', 'pix', ...).
    detrend_blocks : None | dict | list/set/tuple
        None → fall back to `detrend_sigma_sec` for EVERY block (global, the
               original behavior).
        dict → per-block {block: σ_seconds}. Blocks ABSENT from the dict are
               left raw. A value of None / 0 also means "off" for that block.
        list/set/tuple of block names → detrend only those blocks, each at the
               global `detrend_sigma_sec`.
    detrend_sigma_sec : float | None
        Global σ (seconds) used for the None / list forms. None → off.
    fs : float
        Imaging sample rate (Hz), to convert seconds → frames.

    Returns
    -------
    float σ in frames, or None if this block should not be detrended.
    """
    if detrend_blocks is None:
        sig_sec = detrend_sigma_sec
    elif isinstance(detrend_blocks, dict):
        sig_sec = detrend_blocks.get(block_name, None)
    else:                                   # list/set/tuple of block names
        sig_sec = detrend_sigma_sec if block_name in detrend_blocks else None
    if not sig_sec or sig_sec <= 0:
        return None
    return sig_sec * fs


def build_feature_blocks(
    vec_data_use, side_tags,
    beh_t, frame_t, pulse_delay,
    *,
    # agg knobs
    agg_mode='angular', obj_size=None, clip_len=None,
    obj_collapse='salience_mean', nan_absent=True,
    # pix knobs
    pix_frames=None, n_pix_pca=20,
    # motion knobs (block included iff build_motion=True)
    build_motion=False, motion_smooth_sigma=2.0,
    # per-object knobs (block included iff build_per_obj=True)
    build_per_obj=False, total_n_obj=None,
    # self-motion knobs (block included iff build_self_motion=True)
    build_self_motion=False, mov_data=None, self_motion_smooth_sigma=2.0,
    # pix-motion knobs (block included iff build_pix_motion=True)
    build_pix_motion=False, pix_motion_smooth_sigma=2.0,
    # lick+reward knobs (block included iff build_lick_reward=True)
    build_lick_reward=False, lr_data=None, lick_reward_smooth_sigma=0.0,
):
    """Build feature blocks for one dataset, aligned to imaging-frame clock.

    Always builds 'agg' and 'pix' blocks. Optionally builds 'motion' and
    'per_obj' (gated by build_motion / build_per_obj flags).

    Parameters
    ----------
    vec_data_use : list of dict
        Per-side vec_data dicts (output of f_get_monitor_coords). Length 1 or 2.
    side_tags : list of str
        Per-side string tags, e.g. ['R'] or ['L', 'R']. Same length as vec_data_use.
    beh_t : ndarray (T_beh,)
        Behavior-clock timestamps.
    frame_t : ndarray (T_img,)
        Imaging-clock timestamps to resample onto.
    pulse_delay : float
        bh_data[n_dset]['bh_pulse_delay'].
    agg_mode, obj_size, clip_len, obj_collapse, nan_absent : agg knobs
        See f_aggregate_monitor_features. nan_absent NaN-masks continuous
        channels where the side's presence==0 (parallels X_agg pattern).
    pix_frames : ndarray (T_beh, vert_samp, lat_samp[*2])
        Rendered pixel movie for the chosen side(s). Caller picks which.
    n_pix_pca : int
        Number of pix-PCA components to keep.
    build_motion, motion_smooth_sigma : motion knobs (see f_motion_features).
    build_per_obj, total_n_obj : per-object knobs (see f_aggregate_monitor_features_per_obj).

    Returns
    -------
    blocks : dict[str, dict]
        Per-block record with keys:
          'X'           : (T_img, d) ndarray aligned to frame_t
          'names'       : list of d column names
          'side_layout' : True if block is per-side structured
          'kind'        : block kind ('agg', 'pix', 'motion', 'per_obj')
        Plus per-block metadata (e.g. 'n_per_side', 'pres_cols', 'pca_pix').
    """
    from sklearn.decomposition import PCA   # local import: this helper is
                                            # decoupled from any importing script

    blocks = {}
    n_sides = len(side_tags)
    T_img = np.asarray(frame_t).shape[0]

    # ─── 'agg' block ──────────────────────────────────────────────────────
    aggs = [f_aggregate_monitor_features(v, mode=agg_mode, obj_size=obj_size,
                                          clip_len=clip_len, obj_collapse=obj_collapse)
            for v in vec_data_use]
    n_per_side = aggs[0].shape[1]
    agg_beh = np.concatenate(aggs, axis=1)
    pres_cols = [s * n_per_side for s in range(n_sides)]
    cont_cols = [c for c in range(agg_beh.shape[1]) if c not in pres_cols]

    pres_at_img = f_resample_to_imaging(beh_t, agg_beh[:, pres_cols], frame_t, pulse_delay, kind='nearest')
    cont_at_img = f_resample_to_imaging(beh_t, agg_beh[:, cont_cols], frame_t, pulse_delay, kind='linear')
    X_agg = np.empty((T_img, agg_beh.shape[1]), dtype=np.float32)
    X_agg[:, pres_cols] = pres_at_img
    X_agg[:, cont_cols] = cont_at_img

    if nan_absent:
        for s, p_col in enumerate(pres_cols):
            absent = X_agg[:, p_col] == 0
            for c in range(s*n_per_side + 1, (s+1)*n_per_side):
                X_agg[absent, c] = np.nan

    side_feat_names_single = ['pres', 'lat', 'vert',
                              ('ang_size' if agg_mode in ('angular', 'angular_salience') else 'dist')]
    if n_per_side == 5:
        side_feat_names_single.append('salience')
    agg_feat_names = [f'{nm}_{tag}' for tag in side_tags for nm in side_feat_names_single]

    blocks['agg'] = {
        'X':                       X_agg,
        'names':                   list(agg_feat_names),
        'side_layout':             True,
        'kind':                    'agg',
        'n_per_side':              n_per_side,
        'pres_cols':               list(pres_cols),
        'side_feat_names_single':  list(side_feat_names_single),
    }

    # ─── 'pix' block ──────────────────────────────────────────────────────
    if pix_frames is None:
        raise ValueError('build_feature_blocks: pix_frames is required for the pix block')
    pix = pix_frames.reshape(pix_frames.shape[0], -1).astype(np.float32)
    pca_pix = PCA(n_components=n_pix_pca, svd_solver='randomized', random_state=42)
    pix_pc = pca_pix.fit_transform(pix)
    X_pix = f_resample_to_imaging(beh_t, pix_pc, frame_t, pulse_delay, kind='linear').astype(np.float32)

    blocks['pix'] = {
        'X':                            X_pix,
        'names':                        [f'pix_PC{i+1}' for i in range(X_pix.shape[1])],
        'side_layout':                  False,
        'kind':                         'pix',
        'pca_pix':                      pca_pix,
        'explained_variance_ratio_cum': float(np.cumsum(pca_pix.explained_variance_ratio_)[-1]),
    }

    # ─── 'pix_mot' block (optional) ───────────────────────────────────────
    # Time derivative of each pix-PC trace. The natural parallel to the
    # 'motion' block but operating on the visual scene's variance modes —
    # captures "rate of change of each pixel-PCA component". E.g., d_pix_PC1
    # ≈ "rate of change of overall on-screen content"; d_pix_PC2 ≈ "rate of
    # lateral motion across the monitor"; etc.
    if build_pix_motion:
        beh_dt = float(np.median(np.diff(beh_t)))
        if pix_motion_smooth_sigma and pix_motion_smooth_sigma > 0:
            pix_pc_s = gaussian_filter1d(pix_pc, pix_motion_smooth_sigma,
                                          axis=0, mode='nearest')
        else:
            pix_pc_s = pix_pc
        pix_pc_dot = np.gradient(pix_pc_s, beh_t, axis=0)
        X_pix_mot = f_resample_to_imaging(beh_t, pix_pc_dot, frame_t, pulse_delay,
                                           kind='linear').astype(np.float32)

        blocks['pix_mot'] = {
            'X':            X_pix_mot,
            'names':        [f'd_pix_PC{i+1}' for i in range(X_pix_mot.shape[1])],
            'side_layout':  False,
            'kind':         'pix_mot',
            'beh_dt':       beh_dt,
            'smooth_sigma': pix_motion_smooth_sigma,
        }

    # ─── 'motion' block (optional) ────────────────────────────────────────
    if build_motion:
        beh_dt = float(np.median(np.diff(beh_t)))
        mots = [f_motion_features(v, mode=agg_mode, obj_size=obj_size, clip_len=clip_len,
                                   obj_collapse=obj_collapse,
                                   smooth_sigma=motion_smooth_sigma, beh_dt=beh_dt,
                                   beh_t=beh_t)
                for v in vec_data_use]
        n_per_side_mot = mots[0].shape[1]
        mot_beh = np.concatenate(mots, axis=1)
        X_mot = f_resample_to_imaging(beh_t, mot_beh, frame_t, pulse_delay, kind='linear').astype(np.float32)

        if nan_absent:
            for s, p_col in enumerate(pres_cols):
                absent = X_agg[:, p_col] == 0
                for c in range(s*n_per_side_mot, (s+1)*n_per_side_mot):
                    X_mot[absent, c] = np.nan

        mot_chans_single = ['d_lat', 'd_vert', 'd_dist']
        if agg_mode in ('angular', 'angular_salience'):
            mot_chans_single.append('d_ang_size')
        mot_chans_single.extend(['motion_mag', 'motion_dir_sin', 'motion_dir_cos'])
        mot_feat_names = [f'{nm}_{tag}' for tag in side_tags for nm in mot_chans_single]

        blocks['motion'] = {
            'X':            X_mot,
            'names':        list(mot_feat_names),
            'side_layout':  True,
            'kind':         'motion',
            'n_per_side':   n_per_side_mot,
            'beh_dt':       beh_dt,
            'smooth_sigma': motion_smooth_sigma,
        }

    # ─── 'per_obj' block (optional) ───────────────────────────────────────
    if build_per_obj:
        if total_n_obj is None:
            raise ValueError('build_per_obj=True requires total_n_obj')
        per_obj_sides = [f_aggregate_monitor_features_per_obj(
                            v, mode=agg_mode, obj_size=obj_size, clip_len=clip_len,
                            total_n_obj=total_n_obj)
                         for v in vec_data_use]
        d_per_obj = per_obj_sides[0].shape[2]
        per_obj_beh = np.concatenate([s.reshape(s.shape[0], -1) for s in per_obj_sides], axis=1)

        pres_cols_po = []
        for s in range(n_sides):
            side_off = s * total_n_obj * d_per_obj
            for k in range(total_n_obj):
                pres_cols_po.append(side_off + k * d_per_obj)
        cont_cols_po = [c for c in range(per_obj_beh.shape[1]) if c not in pres_cols_po]

        pres_at_img_po = f_resample_to_imaging(beh_t, per_obj_beh[:, pres_cols_po], frame_t, pulse_delay, kind='nearest')
        cont_at_img_po = f_resample_to_imaging(beh_t, per_obj_beh[:, cont_cols_po], frame_t, pulse_delay, kind='linear')
        X_per_obj = np.empty((T_img, per_obj_beh.shape[1]), dtype=np.float32)
        X_per_obj[:, pres_cols_po] = pres_at_img_po
        X_per_obj[:, cont_cols_po] = cont_at_img_po

        if nan_absent:
            for p_col in pres_cols_po:
                absent = X_per_obj[:, p_col] == 0
                for off in range(1, d_per_obj):
                    X_per_obj[absent, p_col + off] = np.nan

        chan_names_po = ['pres', 'lat', 'vert',
                         ('ang_size' if agg_mode in ('angular', 'angular_salience') else 'dist')]
        if d_per_obj == 5:
            chan_names_po.append('salience')
        per_obj_feat_names = [f'{nm}_obj{k:02d}_{tag}'
                              for tag in side_tags
                              for k in range(total_n_obj)
                              for nm in chan_names_po]

        blocks['per_obj'] = {
            'X':            X_per_obj,
            'names':        list(per_obj_feat_names),
            'side_layout':  True,
            'kind':         'per_obj',
            'total_n_obj':  total_n_obj,
            'd_per_obj':    d_per_obj,
            'pres_cols':    list(pres_cols_po),
        }

    # ─── 'self_mot' block (optional) ──────────────────────────────────────
    if build_self_motion:
        if mov_data is None:
            raise ValueError('build_self_motion=True requires mov_data')
        beh_dt = float(np.median(np.diff(beh_t)))
        sm = f_self_motion_features(mov_data, smooth_sigma=self_motion_smooth_sigma,
                                    beh_dt=beh_dt, beh_t=beh_t)
        X_self_mot = f_resample_to_imaging(beh_t, sm, frame_t, pulse_delay,
                                            kind='linear').astype(np.float32)
        blocks['self_mot'] = {
            'X':            X_self_mot,
            'names':        ['self_d_dist', 'self_d_phi'],
            'side_layout':  False,
            'kind':         'self_mot',
            'beh_dt':       beh_dt,
            'smooth_sigma': self_motion_smooth_sigma,
        }

    # ─── 'beh' block (optional) — lick + reward events ────────────────────
    if build_lick_reward:
        if lr_data is None:
            raise ValueError('build_lick_reward=True requires lr_data '
                             '(output of f_proc_lick_rew)')
        lr = f_lick_reward_features(lr_data, smooth_sigma=lick_reward_smooth_sigma)
        # Use nearest-neighbor interp so raw event counts don't get smeared
        # across frames by linear interpolation (would turn a +1 spike into
        # 0.5 / 0.5 in adjacent frames). When smooth_sigma > 0 the trace is
        # already a smooth rate-like signal so the kind doesn't matter much.
        X_beh = f_resample_to_imaging(beh_t, lr, frame_t, pulse_delay,
                                       kind='nearest').astype(np.float32)
        blocks['beh'] = {
            'X':            X_beh,
            'names':        ['lick', 'rew'],
            'side_layout':  False,
            'kind':         'beh',
            'smooth_sigma': lick_reward_smooth_sigma,
        }

    return blocks
