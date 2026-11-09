# -*- coding: utf-8 -*-
"""
f_conv_gpu.py — torch (GPU) gabor / motion-energy convolution, CPU-parity with f_conv_repo.

The CPU convolution (f_conv_repo.f_gabor_map / f_meng_map, via scipy.signal.fftconvolve) is the
bottleneck for a WIDE feature scan. This module reimplements the SAME math on the GPU with torch
conv2d/conv1d so a wide gabor/motion-energy grid can be convolved fast, then reduced to the
whole-monitor tuning regressor (the reduce lives in f_feature_scan). It reuses the numpy kernel
builders `f_visual_features._gabor_pair` / `_temporal_quad` as the single source of truth for the
kernel MATH, so only the convolution engine differs.

Parity notes (verified by allclose vs the CPU maps within uint16 quant tolerance):
  * scipy.fftconvolve is TRUE convolution (kernel flipped); torch conv2d/conv1d is cross-correlation.
    We FLIP every kernel before the torch op so the two agree. This matters for the meng opponent
    map (|R|²−|L|²): a correlation-based path would swap R↔L and negate the map. Magnitude
    √(even²+odd²) is flip-invariant, but we flip anyway for uniformity.
  * mode='same' ↔ padding = kernel//2 (all kernels are odd-length: spatial 2n+1, temporal support 11).
  * meng shares the SPATIAL gabor conv across temporal frequencies — spatial conv once, one temporal
    conv1d per tfreq on the shared complex field (the CPU f_meng_map recomputes spatial per tfreq).

Spatial engine (2026-07-24): every spatial conv goes through `_spatial_conv`, an exact drop-in for
F.conv2d that (a) CROPS kernel rows/cols which can only ever hit the zero pad (_crop_kernel — our
low-SF gabors are bigger than the frame, so this is a free exact win) and (b) picks direct conv2d for
small kernels or an FFT path for large (_fft_conv2d, cost independent of K, one shared forward rfft2
per frame-block). Direct conv2d costs K^2 per pixel, which the CPU path never paid (it used
scipy.fftconvolve); at sigma=0.5*lambda a cyc_fov=1 gabor is 451x451 on a 101x150 render, i.e. ~200x
more arithmetic than the transform. Crossover knob FFT_MIN_K; force with backend='direct'|'fft'.
The temporal conv1d is unchanged (support is short).

Created 2026-07-16.
"""

import numpy as np
import torch
import torch.nn.functional as F

from f_visual_features import _gabor_pair, _temporal_quad, _gauss_env, f_grid_bounds, _sample_idx


def pick_device(device=None):
    """Resolve a device string. None/'auto' → 'cuda' if available else 'cpu'."""
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def as_movie_tensor(movie, device=None, dtype=torch.float32):
    """(T,H,W) array/tensor → float32 torch tensor on `device` (matches the CPU float cast)."""
    device = pick_device(device)
    if torch.is_tensor(movie):
        return movie.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(movie), dtype=dtype, device=device)


def _spatial_kernels(theta, wavelength, sigma_scale, gamma, device, dtype):
    """Even/odd gabor kernels as (k,k) torch tensors, FLIPPED for conv-via-correlation."""
    ev, od = _gabor_pair(theta, wavelength, sigma_scale * wavelength, gamma)   # numpy (k,k)
    ev = torch.as_tensor(ev[::-1, ::-1].copy(), dtype=dtype, device=device)
    od = torch.as_tensor(od[::-1, ::-1].copy(), dtype=dtype, device=device)
    return ev, od


def _temporal_kernels(tfreq, support, sigma, device, dtype):
    """Even/odd temporal kernels as (L,) torch tensors, FLIPPED for conv-via-correlation."""
    te, to = _temporal_quad(tfreq, support, sigma)                            # numpy (L,)
    te = torch.as_tensor(te[::-1].copy(), dtype=dtype, device=device)
    to = torch.as_tensor(to[::-1].copy(), dtype=dtype, device=device)
    return te, to


# ─────────────────────────────────────────────────────────────────────────────
# Spatial convolution engine — direct conv2d (small kernels) | FFT (large kernels)
# ─────────────────────────────────────────────────────────────────────────────

FFT_MIN_K = 48              # kernel side (after cropping) at/above which the FFT path is used
FFT_BUDGET_BYTES = 1.5e9    # frequency-domain working set per call → sets the internal frame sub-chunk
FFT_WITH_PHASE = False      # allow the FFT path when gabor PHASE (Pc/Ps) is requested — see below

# Why FFT_WITH_PHASE defaults OFF. Phase is re/mag gated by an ABSOLUTE test (`mag > 1e-6`, in
# f_gabor_features). Over a BLACK screen region the true response is exactly 0, so direct conv2d
# returns exact zeros (gate closed → phase 0) while ANY FFT engine returns ~1e-5 rounding (gate opens →
# arbitrary unit vector). Measured on a render-like movie: magnitude/even/luminance agree to <1e-4
# either way (no division → no amplification), but POINT-SAMPLED phase drops to corr 0.86. So phase
# stays on the direct path — values are unchanged from every cache built so far. Note the CPU reference
# (scipy.fftconvolve) has the FFT behaviour, so cpu and gpu-direct ALREADY disagree on phase
# (corr 0.9917): the gate, not the engine, is the underlying problem. Fixing it (scale-relative gate)
# is a feature-semantics change — do it deliberately when picking up TODO #70/#71, then set this True.


def _fast_len(n):
    """Smallest length >= n factorable into 2/3/5/7 (cuFFT-friendly transform size)."""
    n = int(max(1, n))
    if n <= 8:
        return n
    m = n
    while True:
        r = m
        for f in (2, 3, 5, 7):
            while r % f == 0:
                r //= f
        if r == 1:
            return m
        m += 1


def _crop_kernel(w, H, W, ph, pw):
    """Drop kernel rows/cols that can NEVER multiply a real pixel of an (H,W) frame. EXACT, not an
    approximation: with zero padding `ph`, output row i reads kernel row u only for u in
    [ph-H+1, ph+H-1], so a kernel taller than 2H-1 (wider than 2W-1) carries rows/cols that only ever
    land on the zero pad. The low-SF gabors here are LARGER than the frame (sigma=0.5*lambda, support
    6*sigma+1 → 451x451 on a 101x150 render), so this cuts both the direct MAC count and the FFT
    transform size for free. Returns (w, ph, pw)."""
    Kh, Kw = w.shape[-2], w.shape[-1]
    lo_h, hi_h = max(0, ph - H + 1), min(Kh, ph + H)
    lo_w, hi_w = max(0, pw - W + 1), min(Kw, pw + W)
    if (lo_h, hi_h, lo_w, hi_w) == (0, Kh, 0, Kw):
        return w, ph, pw
    return w[..., lo_h:hi_h, lo_w:hi_w].contiguous(), ph - lo_h, pw - lo_w


def _fft_conv2d(x, w, ph, pw):
    """conv2d-EQUIVALENT (cross-correlation, zero-padded 'same') via the frequency domain. Cost is
    INDEPENDENT of kernel size, so it wins hard on the low-SF gabors where direct conv2d pays K^2
    (451^2 = 203k MACs/px vs ~2k for the transform). ONE forward rfft2 per frame-block is SHARED by
    every filter in `w` (all filters of a conv-unit share a size), then one pointwise multiply +
    inverse each. `w` arrives in conv2d convention (already flipped by _spatial_kernels), so it is
    flipped BACK here: a frequency multiply is TRUE convolution while conv2d is correlation — without
    this the antisymmetric ODD gabor (and the meng opponent) would come out negated. Sub-chunks over
    frames internally: padding to a linear-convolution size inflates the pixel count (101x150 →
    ~315x448), so the caller's chunk_t would blow up VRAM. ~1e-5 vs direct (fp32 FFT rounding)."""
    b, _, H, W = x.shape
    n, _, Kh, Kw = w.shape
    fh, fw = _fast_len(H + Kh - 1), _fast_len(W + Kw - 1)
    oh, ow = Kh - 1 - ph, Kw - 1 - pw                    # centered crop of the full linear convolution
    kf = torch.fft.rfft2(torch.flip(w[:, 0], (-2, -1)), s=(fh, fw)).unsqueeze(0)  # (1,n,fh,fw//2+1)
    per = n * fh * (fw // 2 + 1) * 8 + n * fh * fw * 4   # complex product + real inverse, per frame
    cb = max(1, min(b, int(FFT_BUDGET_BYTES // max(1, per))))
    out = torch.empty((b, n, H, W), dtype=x.dtype, device=x.device)
    for s in range(0, b, cb):
        e = min(s + cb, b)
        xf = torch.fft.rfft2(x[s:e, 0], s=(fh, fw)).unsqueeze(1)     # (bc,1,fh,fw//2+1) — shared
        y = torch.fft.irfft2(xf * kf, s=(fh, fw))                    # (bc,n,fh,fw)
        out[s:e] = y[..., oh:oh + H, ow:ow + W]
        del xf, y
    return out


def _spatial_conv(x, w, pad, backend='auto'):
    """The module's spatial convolution: EXACT drop-in for `F.conv2d(x, w, padding=pad)`. Crops kernel
    rows/cols that only ever hit the zero pad (_crop_kernel), then picks the engine — direct conv2d for
    small kernels, FFT for large (crossover FFT_MIN_K, where K^2 overtakes the transform cost).
    backend='direct'|'fft' forces one (parity testing / benchmarking). x (b,1,H,W), w (n,1,Kh,Kw),
    pad int | (ph,pw) → (b,n,H,W)."""
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = ((int(pad), int(pad)) if isinstance(pad, (int, np.integer))
              else (int(pad[0]), int(pad[1])))
    w, ph, pw = _crop_kernel(w, H, W, ph, pw)
    if backend == 'fft' or (backend == 'auto' and max(w.shape[-2], w.shape[-1]) >= FFT_MIN_K):
        return _fft_conv2d(x, w, ph, pw)
    return F.conv2d(x, w, padding=(ph, pw))


def f_gabor_map_gpu(movie, theta, wavelength, sigma_scale=0.5, gamma=1.0,
                    chunk_t=2000, device=None):
    """GPU gabor MAGNITUDE map √(even²+odd²) → (T,H,W) float32 NUMPY, built on CPU and STREAMED
    chunk-by-chunk (movie stays on CPU; the full map is never resident on the GPU). Matches
    f_conv_repo.f_gabor_map."""
    dev = pick_device(device)
    T, H, W = movie.shape[0], movie.shape[1], movie.shape[2]
    ev, od = _spatial_kernels(theta, wavelength, sigma_scale, gamma, dev, torch.float32)
    ws = torch.stack([ev, od]).unsqueeze(1); pad = ev.shape[-1] // 2
    out = np.empty((T, H, W), np.float32)
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        x = _to_chunk(movie, s, e, dev).unsqueeze(1)
        r = _spatial_conv(x, ws, pad)
        out[s:e] = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2).cpu().numpy()
        del x, r
    return out


def _temporal_conv(S, kt):
    """Convolve (T,H,W) real tensor with 1D kernel `kt` (L,) along time → (T,H,W). 'same', flipped."""
    T, H, W = S.shape
    L = kt.shape[0]
    pad = L // 2
    x = S.reshape(T, H * W).transpose(0, 1).unsqueeze(1)  # (H*W, 1, T)
    w = kt.reshape(1, 1, L)                                # already flipped by _temporal_kernels
    y = F.conv1d(x, w, padding=pad)                        # (H*W,1,T)  ('same' for odd L)
    return y.squeeze(1).transpose(0, 1).reshape(T, H, W)


def f_meng_maps_gpu(movie, theta, wavelength, tfreqs, temporal_support=11, temporal_sigma=None,
                    sigma_scale=0.5, gamma=1.0, output='opponent', chunk_t=2000, device=None):
    """GPU Adelson–Bergen motion energy for a group of temporal freqs (shared spatial gabor) → dense
    (T,H,W) float32 NUMPY per tfreq (opponent |R|²−|L|², or (eR,eL) for output!='opponent'). STREAMED
    over time with a temporal HALO — the movie stays on CPU; each chunk (+halo) is convolved on the
    GPU and the valid interior written to a CPU array, so the full-time field is never on the GPU
    (bounds VRAM for long sessions; the earlier full-time version OOM'd). Per-chunk it recomputes the
    cheap spatial conv. `temporal_support` = scalar | {tf: support}; `temporal_sigma` scalar|dict|None.
    Matches f_conv_repo.f_meng_map per tfreq."""
    dev = pick_device(device)
    T, H, W = movie.shape[0], movie.shape[1], movie.shape[2]
    ev, od = _spatial_kernels(theta, wavelength, sigma_scale, gamma, dev, torch.float32)
    ws = torch.stack([ev, od]).unsqueeze(1); pad_s = ev.shape[-1] // 2
    out = {}
    for tf in np.atleast_1d(tfreqs):
        tf = float(tf)
        sup = int(temporal_support[tf] if isinstance(temporal_support, dict) else temporal_support)
        sig = (temporal_sigma[tf] if isinstance(temporal_sigma, dict) else temporal_sigma) or sup / 4.0
        te, to = _temporal_kernels(tf, sup, sig, dev, torch.float32)
        halo = sup // 2
        opp = np.empty((T, H, W), np.float32) if output == 'opponent' else None
        eRo = None if output == 'opponent' else np.empty((T, H, W), np.float32)
        eLo = None if output == 'opponent' else np.empty((T, H, W), np.float32)
        for s in range(0, T, chunk_t):
            e = min(s + chunk_t, T)
            lo, hi = max(0, s - halo), min(T, e + halo)
            x = _to_chunk(movie, lo, hi, dev).unsqueeze(1)
            r = _spatial_conv(x, ws, pad_s)
            re, ro = r[:, 0], r[:, 1]
            re_te = _temporal_conv(re, te); re_to = _temporal_conv(re, to)
            ro_te = _temporal_conv(ro, te); ro_to = _temporal_conv(ro, to)
            eR = (re_te - ro_to) ** 2 + (re_to + ro_te) ** 2
            eL = (re_te + ro_to) ** 2 + (ro_te - re_to) ** 2
            sl = slice(s - lo, s - lo + (e - s))
            if output == 'opponent':
                opp[s:e] = (eR - eL)[sl].cpu().numpy()
            else:
                eRo[s:e] = eR[sl].cpu().numpy(); eLo[s:e] = eL[sl].cpu().numpy()
            del x, r, re, ro, re_te, re_to, ro_te, ro_to, eR, eL
        out[tf] = opp if output == 'opponent' else (eRo, eLo)
    return out


def _reduce_space(x, reduce):
    """Reduce the last two (spatial) dims → whole-monitor scalar per frame. 'absmean' (default) =
    mean(|x|) (correct for signed meng) | 'mean' (signed) | 'rms' = sqrt(mean(x²))."""
    if reduce == 'mean':
        return x.mean(dim=(-2, -1))
    if reduce == 'rms':
        return torch.sqrt((x * x).mean(dim=(-2, -1)))
    return x.abs().mean(dim=(-2, -1))


def _to_chunk(movie, s, e, dev):
    """Frames [s:e] of the movie → float32 tensor on `dev`. The movie stays on CPU in its native
    dtype (uint8 from the render) — only this chunk is converted + moved, so neither the full movie
    nor a full float32 copy is ever resident (bounds both CPU and GPU memory for long sessions)."""
    sub = movie[s:e]
    if not torch.is_tensor(sub):
        sub = torch.as_tensor(np.asarray(sub))
    return sub.to(device=dev, dtype=torch.float32)


def f_gabor_activity_gpu(movie, specs, reduce='absmean', sigma_scale=0.5, gamma=1.0,
                         chunk_t=2000, device=None, return_spatial=False):
    """Batched, REDUCE-FUSED gabor scan: convolve a batch of gabor `specs` as conv2d out-channels and
    reduce each to the whole-monitor magnitude per frame. STREAMS over time — the movie stays on CPU,
    only a `chunk_t`-frame slice is on the GPU at once, so peak VRAM ≈ chunk_t·2·len(specs)·H·W (does
    NOT grow with session length). Kernels of different wavelengths are zero-padded to a common odd
    size. Returns ts (T, len(specs)) ndarray; with `return_spatial` also the per-filter TIME-MEAN
    magnitude map (len(specs), H, W). `len(specs)`=filt_batch and `chunk_t` are the VRAM knobs."""
    dev = pick_device(device)
    T, H, W = movie.shape[0], movie.shape[1], movie.shape[2]
    pairs = [_spatial_kernels(sp['theta'], sp['wavelength'], sigma_scale, gamma, dev, torch.float32)
             for sp in specs]
    K = max(ev.shape[-1] for ev, _ in pairs)                 # common (odd) kernel size

    def _pad(kern):
        k = kern.shape[-1]; p = (K - k) // 2                 # K-k even (both odd) → symmetric
        return F.pad(kern, (p, K - k - p, p, K - k - p))
    w = torch.stack([t for ev, od in pairs for t in (_pad(ev), _pad(od))]).unsqueeze(1)  # (2n,1,K,K)
    pad = K // 2
    n = len(specs)
    ts = np.empty((T, n), np.float32)
    spat = torch.zeros((n, H, W), device=dev) if return_spatial else None
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        x = _to_chunk(movie, s, e, dev).unsqueeze(1)                         # move ONLY this chunk to GPU
        r = _spatial_conv(x, w, pad).view(e - s, n, 2, H, W)          # (b,n,2,H,W)
        mag = torch.sqrt(r[:, :, 0] ** 2 + r[:, :, 1] ** 2)           # (b,n,H,W)
        ts[s:e] = _reduce_space(mag, reduce).cpu().numpy()
        if return_spatial:
            spat += mag.sum(dim=0)
        del x, r, mag
    return (ts, (spat / T).cpu().numpy()) if return_spatial else ts


def f_meng_activity_gpu(movie, theta, wavelength, tfreqs, reduce='absmean', temporal_support=11,
                        temporal_sigma=None, sigma_scale=0.5, gamma=1.0, output='opponent',
                        chunk_t=2000, device=None, return_spatial=False):
    """Reduce-fused meng scan over `tfreqs` (shared spatial gabor). STREAMS over time with a temporal
    HALO — the movie stays on CPU; each chunk (+halo for the temporal filter) is convolved on the GPU
    and reduced, so peak VRAM is bounded (does NOT hold the full-time response). Recomputes the cheap
    spatial conv per chunk. `temporal_support` = scalar or {tf: support} (per-tf, constant-cycle);
    halo = support//2. Returns {tfreq: ts (T,)} (opponent) or {tfreq: (tsR, tsL)}; with
    `return_spatial` opponent values are (ts, spat (H,W)) = time-mean |opponent|."""
    dev = pick_device(device)
    T, H, W = movie.shape[0], movie.shape[1], movie.shape[2]
    ev, od = _spatial_kernels(theta, wavelength, sigma_scale, gamma, dev, torch.float32)
    ws = torch.stack([ev, od]).unsqueeze(1); pad_s = ev.shape[-1] // 2
    out = {}
    for tf in np.atleast_1d(tfreqs):
        tf = float(tf)
        sup = int(temporal_support[tf] if isinstance(temporal_support, dict) else temporal_support)
        sig = (temporal_sigma[tf] if isinstance(temporal_sigma, dict) else temporal_sigma) or sup / 4.0
        te, to = _temporal_kernels(tf, sup, sig, dev, torch.float32)
        halo = sup // 2
        tsR = np.empty(T, np.float32)
        tsL = None if output == 'opponent' else np.empty(T, np.float32)
        spat = torch.zeros((H, W), device=dev) if (return_spatial and output == 'opponent') else None
        for s in range(0, T, chunk_t):
            e = min(s + chunk_t, T)
            lo, hi = max(0, s - halo), min(T, e + halo)               # +halo for the temporal conv
            x = _to_chunk(movie, lo, hi, dev).unsqueeze(1)
            r = _spatial_conv(x, ws, pad_s)                           # (hi-lo,2,H,W)
            re, ro = r[:, 0], r[:, 1]
            re_te = _temporal_conv(re, te); re_to = _temporal_conv(re, to)
            ro_te = _temporal_conv(ro, te); ro_to = _temporal_conv(ro, to)
            eR = (re_te - ro_to) ** 2 + (re_to + ro_te) ** 2          # |R|²
            eL = (re_te + ro_to) ** 2 + (ro_te - re_to) ** 2          # |L|²
            sl = slice(s - lo, s - lo + (e - s))                      # valid interior (halo trimmed)
            if output == 'opponent':
                opp = (eR - eL)[sl]
                tsR[s:e] = _reduce_space(opp, reduce).cpu().numpy()
                if spat is not None:
                    spat += opp.abs().sum(dim=0)
            else:
                tsR[s:e] = _reduce_space(eR[sl], reduce).cpu().numpy()
                tsL[s:e] = _reduce_space(eL[sl], reduce).cpu().numpy()
            del x, r, re, ro, re_te, re_to, ro_te, ro_to, eR, eL
        if output == 'opponent':
            out[tf] = (tsR, (spat / T).cpu().numpy()) if return_spatial else tsR
        else:
            out[tf] = (tsR, tsL)
    return out


def _make_reducer_torch(H, W, grid, pool, sample_ds, dev):
    """torch spatial reducer (t,H,W)->(t,ny,nx), byte-parity with f_visual_features._make_reducer.
    'grid' = area-normalized mean-pool over the SAME f_grid_bounds tiling (0/1 assignment matmuls sum the
    exact reduceat segments, ÷ the same per-cell pixel area); 'sample' = centered strided point-sample
    (_sample_idx offsets). Returns (ny, nx, reduce_fn)."""
    if pool == 'grid':
        ys, xs = f_grid_bounds(H, W, grid)               # int cut points; edge cells absorb remainder
        ny, nx = grid
        area = torch.as_tensor((np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32), device=dev)
        Ay = torch.zeros((ny, H), dtype=torch.float32, device=dev)
        for i in range(ny):
            Ay[i, int(ys[i]):int(ys[i + 1])] = 1.0
        Ax = torch.zeros((nx, W), dtype=torch.float32, device=dev)
        for k in range(nx):
            Ax[k, int(xs[k]):int(xs[k + 1])] = 1.0

        def _reduce(x):                                  # (t,H,W) -> (t,ny,nx) mean per cell
            g = torch.einsum('yh,thw->tyw', Ay, x)       # sum rows in each y-segment
            p = torch.einsum('xw,tyw->tyx', Ax, g)       # sum cols in each x-segment
            return p / area
        return ny, nx, _reduce
    if pool == 'sample':
        ds_h, ds_w = ((int(sample_ds[0]), int(sample_ds[1])) if isinstance(sample_ds, (tuple, list))
                      else (int(sample_ds), int(sample_ds)))
        rows, cols = _sample_idx(H, ds_h), _sample_idx(W, ds_w)
        oh, ow = int(rows[0]), int(cols[0])
        ny, nx = len(rows), len(cols)

        def _reduce(x):                                  # centered strided point-sample
            return x[:, oh::ds_h, ow::ds_w]
        return ny, nx, _reduce
    raise ValueError(f"pool {pool!r} (expected 'grid' | 'sample')")


def _gab_stat_torch(tok, re, ro, mag):
    """Per-channel gabor stat in torch (parity with f_gabor_features' numpy `chan` lambdas)."""
    if tok == 'M':                                       # magnitude
        return mag
    if tok == 'Pc':                                      # phase cos = re/mag (0 where mag<=1e-6)
        return torch.where(mag > 1e-6, re / mag.clamp_min(1e-6), torch.zeros_like(re))
    if tok == 'Ps':                                      # phase sin = ro/mag
        return torch.where(mag > 1e-6, ro / mag.clamp_min(1e-6), torch.zeros_like(ro))
    if tok == 'Qe':                                      # signed even (no magnitude taken)
        return re
    if tok == 'Qo':                                      # signed odd
        return ro
    if tok == 'Ev':                                      # even = |re|
        return re.abs()
    raise ValueError(f'unknown gabor stat token {tok!r}')


def f_gabor_pooled_gpu(movie, thetas, wavelengths, tokens, pool, grid, sample_ds,
                       sigma_scale=0.5, gamma=1.0, chunk_t=2000, device=None, lum=False):
    """GPU gabor for the feature-BUILD path (drop-in for f_visual_features.f_gabor_features' inner
    conv+pool). Streams time-chunks, convolves even/odd of every (orientation × wavelength) on the GPU
    (batched as conv2d out-channels), then computes each stat token (`tokens` ⊆ {'M','Pc','Ps','Ev'};
    `lum` adds 'Lum') AND the spatial pool (`pool`/`grid`/`sample_ds`) ON THE GPU, transferring only the
    small pooled (t,ny,nx) result — the full-res (t,H,W) response never crosses to CPU (the old per-filter
    .cpu() + numpy stat+pool was the speed killer). Byte-parity with the CPU numpy path (~1e-6). `lum=True`
    also convolves the carrier-free Gaussian envelope per SF → √2·|proj|, stored under (0, isf, 'Lum').
    Returns {(io,isf,tok): (T,ny,nx) float32}."""
    dev = pick_device(device)
    T, H, W = movie.shape[0], movie.shape[1], movie.shape[2]
    ny, nx, reduce_t = _make_reducer_torch(H, W, grid, pool, sample_ds, dev)
    thetas = np.atleast_1d(thetas); wavelengths = np.atleast_1d(wavelengths)
    idx = [(io, isf) for io in range(len(thetas)) for isf in range(len(wavelengths))]
    pooled = {}
    w = None
    if tokens:                                               # skip the gabor conv when only luminance
        pairs = [_spatial_kernels(thetas[io], wavelengths[isf], sigma_scale, gamma, dev, torch.float32)
                 for io, isf in idx]
        K = max(ev.shape[-1] for ev, _ in pairs)             # common (odd) kernel size (zero-pad)

        def _pad(kern):
            k = kern.shape[-1]; p = (K - k) // 2
            return F.pad(kern, (p, K - k - p, p, K - k - p))
        w = torch.stack([t for ev, od in pairs for t in (_pad(ev), _pad(od))]).unsqueeze(1)  # (2nf,1,K,K)
        padw = K // 2
        pooled.update({(io, isf, tok): np.empty((T, ny, nx), np.float32)
                       for io, isf in idx for tok in tokens})
    wl = None
    if lum:                                                  # env kernels per SF (isotropic Gaussian)
        envs = [torch.as_tensor(_gauss_env(sigma_scale * float(wavelengths[isf]), gamma),
                                dtype=torch.float32, device=dev) for isf in range(len(wavelengths))]
        Kl = max(en.shape[-1] for en in envs)

        def _padl(kern):
            k = kern.shape[-1]; p = (Kl - k) // 2
            return F.pad(kern, (p, Kl - k - p, p, Kl - k - p))
        wl = torch.stack([_padl(en) for en in envs]).unsqueeze(1)   # (n_sf,1,Kl,Kl)
        padl = Kl // 2
        pooled.update({(0, isf, 'Lum'): np.empty((T, ny, nx), np.float32)
                       for isf in range(len(wavelengths))})
    nf = len(idx)
    # phase (re/mag) is gate-sensitive to FFT rounding over black regions → keep it on direct conv
    _sb = 'direct' if (not FFT_WITH_PHASE and any(t in ('Pc', 'Ps') for t in tokens)) else 'auto'
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        x = _to_chunk(movie, s, e, dev).unsqueeze(1)
        if w is not None:
            r = _spatial_conv(x, w, padw, backend=_sb).view(e - s, nf, 2, H, W)
            for fi, (io, isf) in enumerate(idx):
                re, ro = r[:, fi, 0], r[:, fi, 1]            # stay on GPU
                mag = torch.sqrt(re * re + ro * ro)
                for tok in tokens:                           # torch stat + torch pool → transfer only pooled
                    pooled[(io, isf, tok)][s:e] = reduce_t(_gab_stat_torch(tok, re, ro, mag)).cpu().numpy()
            del r
        if wl is not None:
            rl = _spatial_conv(x, wl, padl)                  # (t, n_sf, H, W)
            for isf in range(len(wavelengths)):
                proj = rl[:, isf]
                pooled[(0, isf, 'Lum')][s:e] = reduce_t(np.sqrt(2.0) * proj.abs()).cpu().numpy()
            del rl
        del x
    return pooled


def f_meng_pooled_gpu(movie, thetas, wavelengths, temporal_freqs, pool, grid, sample_ds,
                      temporal_support=11, temporal_sigma=None, sigma_scale=0.5, gamma=1.0,
                      output='opponent', chunk_t=1000, device=None):
    """GPU motion energy for the feature-BUILD path (drop-in for f_motion_energy_features' inner
    conv+pool). Per (orientation × wavelength × tfreq): spatial gabor + complex temporal filter →
    opponent |R|²−|L|² (or both directions); the directional energy AND the spatial pool
    (`pool`/`grid`/`sample_ds`) run ON THE GPU, transferring only the pooled (t,ny,nx) result. Streams
    with a temporal HALO. Byte-parity with the CPU numpy path (~1e-6). Returns {(io,isf,itf,d):
    (T,ny,nx) float32} (d=0 for opponent; d∈{0,1} for output='both')."""
    dev = pick_device(device)
    T, H, W = movie.shape[0], movie.shape[1], movie.shape[2]
    ny, nx, reduce_t = _make_reducer_torch(H, W, grid, pool, sample_ds, dev)
    thetas = np.atleast_1d(thetas); wavelengths = np.atleast_1d(wavelengths)
    temporal_freqs = np.atleast_1d(temporal_freqs)
    n_out = 1 if output == 'opponent' else 2
    tsig = temporal_sigma if temporal_sigma else temporal_support / 4.0
    skern = {(io, isf): _spatial_kernels(thetas[io], wavelengths[isf], sigma_scale, gamma, dev, torch.float32)
             for io in range(len(thetas)) for isf in range(len(wavelengths))}
    tkern = {itf: _temporal_kernels(float(tf), int(temporal_support), tsig, dev, torch.float32)
             for itf, tf in enumerate(temporal_freqs)}
    halo = int(temporal_support) // 2
    store = {(io, isf, itf, d): np.empty((T, ny, nx), np.float32)
             for io in range(len(thetas)) for isf in range(len(wavelengths))
             for itf in range(len(temporal_freqs)) for d in range(n_out)}
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        lo, hi = max(0, s - halo), min(T, e + halo)
        x = _to_chunk(movie, lo, hi, dev).unsqueeze(1)
        vs, ve = s - lo, s - lo + (e - s)                    # valid interior (halo trimmed)
        for (io, isf), (ev, od) in skern.items():
            ws = torch.stack([ev, od]).unsqueeze(1); pad_s = ev.shape[-1] // 2
            r = _spatial_conv(x, ws, pad_s); re, ro = r[:, 0], r[:, 1]
            for itf, (te, to) in tkern.items():
                re_te = _temporal_conv(re, te); re_to = _temporal_conv(re, to)
                ro_te = _temporal_conv(ro, te); ro_to = _temporal_conv(ro, to)
                eR = ((re_te - ro_to) ** 2 + (re_to + ro_te) ** 2)[vs:ve]      # |R|² (stay on GPU)
                eL = ((re_te + ro_to) ** 2 + (ro_te - re_to) ** 2)[vs:ve]      # |L|²
                if output == 'opponent':                     # torch pool → transfer only pooled
                    store[(io, isf, itf, 0)][s:e] = reduce_t(eR - eL).cpu().numpy()
                else:
                    store[(io, isf, itf, 0)][s:e] = reduce_t(eR).cpu().numpy()
                    store[(io, isf, itf, 1)][s:e] = reduce_t(eL).cpu().numpy()
            del r, re, ro
        del x
    return store


def _spec_maps_gpu(movie, spec, sigma_scale=0.5, gamma=1.0, chunk_t=4000, device=None):
    """GPU analogue of f_conv_repo._spec_maps for ONE filter spec → list of (stat_token, (T,H,W)
    float32 ndarray). Used as the GPU backend for the dense conv_repo build (results moved to CPU)."""
    if spec['kind'] == 'gabor':
        m = f_gabor_map_gpu(movie, spec['theta'], spec['wavelength'], sigma_scale, gamma,
                            chunk_t=chunk_t, device=device)
        return [('gabM', np.asarray(m, np.float32))]
    out = spec.get('output', 'opponent')
    res = f_meng_maps_gpu(movie, spec['theta'], spec['wavelength'], [spec['tfreq']],
                          spec.get('temporal_support', 11), spec.get('temporal_sigma'),
                          sigma_scale, gamma, out, chunk_t=chunk_t, device=device)[float(spec['tfreq'])]
    if out == 'opponent':
        return [('mengopp', np.asarray(res, np.float32))]
    return [('mengd0', np.asarray(res[0], np.float32)), ('mengd1', np.asarray(res[1], np.float32))]
